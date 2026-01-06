# "estimate_param_simulation_from_mrc.py"
# "estimate_param_simulation_from_mrc.py"
"""
Comprehensive Cryo-EM Particle Stack Analyzer for Simulation Parameter Estimation

This script reads a particle stack and performs key analyses to estimate 
a full set of parameters required for generating realistic synthetic cryo-EM data.
 
"""

# ============================================================================
# 1. COMMON IMPORTS AND UTILITIES
# ============================================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.optimize import curve_fit
import pandas as pd
import time
import sys

try:
    import mrcfile
except ImportError:
    print("Install mrcfile: pip install mrcfile")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator

class ParticleStackReader:
    """Read particle stacks from MRC/MRCS files"""
    def __init__(self, stack_path):
        self.stack_path = Path(stack_path)

    def read_stack(self, max_particles=None):
        if self.stack_path.suffix in ['.mrc', '.mrcs']:
            with mrcfile.open(self.stack_path, permissive=True) as mrc:
                data = mrc.data
                if data.ndim == 2: particles = data[np.newaxis, :, :]
                elif data.ndim == 3: particles = data
                else: raise ValueError(f"Unexpected data dimensions: {data.ndim}")
                if max_particles is not None: particles = particles[:max_particles]
                print(f"Read {len(particles)} particles of size {particles.shape[1]}x{particles.shape[2]}")
                return particles
        else: raise ValueError(f"Unsupported file format: {self.stack_path.suffix}")

def create_masks_gpu(size, signal_radius_px, bg_inner_px, bg_outer_px, device):
    """
    Creates hard-edged signal and background masks
    """
    # 1. Create a coordinate grid pre-centered at (0,0).
    grid = torch.linspace(-0.5 * (size - 1), 0.5 * (size - 1), size, device=device)
    
    # 2. Calculate squared distance from the center for every pixel.
    r_squared = grid[None, :] ** 2 + grid[:, None] ** 2
    
    # 3. Create masks with an exclusive boundary condition (< radius**2).
    signal_mask = (r_squared < signal_radius_px**2)
    background_mask = ((r_squared > bg_inner_px**2) & (r_squared < bg_outer_px**2))
    
    return signal_mask, background_mask


# ============================================================================
# 2. ICE POWER SPECTRUM ANALYSIS MODULE
# ============================================================================
class GPUIcePowerSpectrumAnalyzer:
    def __init__(self, pixel_size=1.0, device=None):
        self.pixel_size = pixel_size
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Ice Spectrum Analyzer using device: {self.device}")

    def compute_radial_power_spectrum_gpu(self, image_batch, mask=None):
        batch_size, size, _ = image_batch.shape
        if mask is not None: image_batch = image_batch * mask.unsqueeze(0)
        
        if mask is not None:
            n_pixels = mask.sum()
            if n_pixels > 0:
                mean_val = image_batch.sum(dim=(1, 2), keepdim=True) / n_pixels
                image_batch = image_batch - (mean_val * mask.unsqueeze(0))
        else: image_batch = image_batch - image_batch.mean(dim=(1, 2), keepdim=True)
        
        fft_2d = torch.fft.fft2(image_batch)
        power_2d = torch.abs(fft_2d)**2
        if not hasattr(self, '_freq_grid'):
            freq_1d = torch.fft.fftfreq(size, d=self.pixel_size, device=self.device)
            freq_y, freq_x = torch.meshgrid(freq_1d, freq_1d, indexing='ij')
            self._freq_grid = torch.sqrt(freq_x**2 + freq_y**2)
        power_2d_shifted = torch.fft.fftshift(power_2d, dim=(1, 2))
        freq_grid_shifted = torch.fft.fftshift(self._freq_grid)
        n_bins = size // 2
        max_freq = freq_grid_shifted.max().item()
        freq_bins = torch.linspace(0, max_freq, n_bins, device=self.device)
        power_radial = torch.zeros(batch_size, n_bins - 1, device=self.device)
        for i in range(n_bins - 1):
            mask_ring = (freq_grid_shifted >= freq_bins[i]) & (freq_grid_shifted < freq_bins[i + 1])
            if mask_ring.any():
                masked_power = power_2d_shifted * mask_ring.unsqueeze(0)
                power_radial[:, i] = masked_power.sum(dim=(1, 2)) / mask_ring.sum()
        freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
        return freq_centers.cpu().numpy(), power_radial

    def analyze_stack_gpu(self, particles, background_mask, batch_size=32):
        n_particles = len(particles)
        print(f"Analyzing ice spectrum for {n_particles} particles on GPU (batch_size={batch_size})...")
        all_power_spectra = []
        n_batches = (n_particles + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Computing Power Spectra"):
                start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, n_particles)
                batch_tensor = torch.from_numpy(particles[start_idx:end_idx]).float().to(self.device)
                freq, power_batch = self.compute_radial_power_spectrum_gpu(batch_tensor, background_mask)
                all_power_spectra.append(power_batch.cpu().numpy())
                del batch_tensor, power_batch
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        all_power_spectra = np.vstack(all_power_spectra)
        return freq, np.mean(all_power_spectra, axis=0), np.std(all_power_spectra, axis=0), all_power_spectra

    def fit_ice_model(self, freq, power, freq_range=(0.01, 0.3)):
        mask = (freq > freq_range[0]) & (freq < freq_range[1]) & (power > 0)
        freq_fit, power_fit = freq[mask], power[mask]
        if len(freq_fit) < 10: return None, None
        def ice_model(f, amp, slope, f_falloff, white_noise): return amp / (1.0 + (f / f_falloff)**slope) + white_noise
        p0, bounds = [np.max(power_fit)*0.1, 2.0, 0.05, np.min(power_fit)], ([0, 0.5, 0.001, 0], [np.inf, 5.0, 0.5, np.inf])
        try:
            popt, pcov = curve_fit(ice_model, freq_fit, power_fit, p0=p0, bounds=bounds, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            params = {'amplitude': popt[0], 'slope': popt[1], 'falloff_freq': popt[2], 'white_noise': popt[3]}
            params_std = {'amplitude': perr[0], 'slope': perr[1], 'falloff_freq': perr[2], 'white_noise': perr[3]}
            return params, params_std
        except Exception: return None, None

def plot_power_spectrum_analysis(freq, power_mean, power_std, all_power_spectra, fitted_params=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5)); ax = axes[0]
    n_show = min(50, len(all_power_spectra))
    for i in range(n_show): ax.loglog(freq, all_power_spectra[i], 'gray', alpha=0.1, linewidth=0.5)
    valid = (freq > 0) & (power_mean > 0)
    ax.loglog(freq[valid], power_mean[valid], 'b-', linewidth=3, label='Mean')
    ax.fill_between(freq[valid], np.maximum(1e-10, power_mean[valid] - power_std[valid]), power_mean[valid] + power_std[valid], alpha=0.3, color='blue', label='± 1 std')
    if fitted_params:
        def ice_model(f, amplitude, slope, falloff_freq, white_noise): return amplitude / (1.0 + (f / falloff_freq)**slope) + white_noise
        model_power = ice_model(freq, **fitted_params)
        ax.loglog(freq[valid], model_power[valid], 'r--', linewidth=2, label='Fit to Mean')
        param_text = f"Slope: {fitted_params['slope']:.2f}\nFalloff: {fitted_params['falloff_freq']:.4f} Å⁻¹"
        ax.text(0.05, 0.05, param_text, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Spatial Frequency (Å⁻¹)'); ax.set_ylabel('Power'); ax.set_title('Ice Power Spectrum Analysis'); ax.legend(); ax.grid(True, alpha=0.3, which='both'); ax.set_xlim(0.01, 0.5)
    ax = axes[1]
    if fitted_params:
        model_power = ice_model(freq, **fitted_params)
        residual = (power_mean - model_power) / (power_std + 1e-10)
        valid_res = (freq > 0.01) & (freq < 0.3)
        ax.plot(freq[valid_res], residual[valid_res], 'o-', markersize=3)
        ax.axhline(0, color='red', linestyle='--'); ax.axhline(2, color='orange', linestyle=':'); ax.axhline(-2, color='orange', linestyle=':')
        ax.set_xlabel('Spatial Frequency (Å⁻¹)'); ax.set_ylabel('Residual (σ)'); ax.set_title('Fit Residuals to Mean'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================================
# 3. SNR ANALYSIS MODULE
# ============================================================================
class GPUSNRAnalyzer:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"SNR Analyzer using device: {self.device}")

    def compute_snr_batch(self, image_batch, signal_mask, background_mask):
        signal_mask_exp, bg_mask_exp = signal_mask.unsqueeze(0), background_mask.unsqueeze(0)
        n_signal, n_background = signal_mask.sum(), background_mask.sum()
        if n_signal == 0 or n_background == 0:
            nan_tensor = torch.full((image_batch.shape[0],), float('nan'), device=self.device)
            return nan_tensor, nan_tensor, nan_tensor, nan_tensor
        signal_mean = (image_batch * signal_mask_exp).sum(dim=(1, 2)) / n_signal
        signal_variance = ((image_batch - signal_mean.view(-1, 1, 1))**2 * signal_mask_exp).sum(dim=(1, 2)) / n_signal
        background_mean = (image_batch * bg_mask_exp).sum(dim=(1, 2)) / n_background
        background_variance = ((image_batch - background_mean.view(-1, 1, 1))**2 * bg_mask_exp).sum(dim=(1, 2)) / n_background
        protein_variance = torch.clamp(signal_variance - background_variance, min=1e-10)
        snr_values = protein_variance / (background_variance + 1e-10)
        return snr_values, torch.sqrt(signal_variance), torch.sqrt(background_variance), torch.sqrt(protein_variance)

    def analyze_stack_gpu(self, particles, signal_mask, background_mask, batch_size=32):
        n_particles = len(particles)
        print(f"Analyzing SNR for {n_particles} particles on GPU (batch_size={batch_size})...")
        results = {'snr': [], 'signal_std': [], 'background_std': [], 'protein_std': []}
        n_batches = (n_particles + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Computing SNR"):
                start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, n_particles)
                batch_tensor = torch.from_numpy(particles[start_idx:end_idx]).float().to(self.device)
                snr, sig_std, bg_std, prot_std = self.compute_snr_batch(batch_tensor, signal_mask, background_mask)
                results['snr'].append(snr.cpu().numpy())
                results['signal_std'].append(sig_std.cpu().numpy())
                results['background_std'].append(bg_std.cpu().numpy())
                results['protein_std'].append(prot_std.cpu().numpy())
                del batch_tensor
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        for key in results: results[key] = np.concatenate(results[key])
        results.update({'n_particles': n_particles, 'image_size': particles.shape[1]})
        return results

def plot_snr_analysis(results, particles=None):
    snr = results['snr'][np.isfinite(results['snr'])]
    if len(snr) == 0:
        print("Warning: No valid SNR values to plot.")
        return plt.figure()
    fig = plt.figure(figsize=(16, 8)); gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.logspace(np.log10(max(snr.min(), 1e-5)), np.log10(snr.max()), 50)
    ax1.hist(snr, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(snr), color='red', linestyle='--', label=f'Mean: {np.mean(snr):.4f}')
    ax1.axvline(np.median(snr), color='orange', linestyle='--', label=f'Median: {np.median(snr):.4f}')
    ax1.set_xscale('log'); ax1.set_xlabel('SNR (var(protein)/var(ice)) [log scale]'); ax1.set_ylabel('Count'); ax1.set_title('SNR Distribution'); ax1.legend(); ax1.grid(True, alpha=0.3, which='both')
    ax2 = fig.add_subplot(gs[0, 1])
    protein_var, ice_var = results['protein_std']**2, results['background_std']**2
    valid_vars = np.isfinite(protein_var) & np.isfinite(ice_var)
    ax2.scatter(ice_var[valid_vars], protein_var[valid_vars], alpha=0.3, s=10)
    ax2.set_xlabel('Ice Variance'); ax2.set_ylabel('Protein Variance'); ax2.set_title('Protein vs Ice Variance')
    xlim = ax2.get_xlim(); x_diag = np.linspace(xlim[0], xlim[1], 100)
    for snr_line in [0.01, 0.05, 0.10]: ax2.plot(x_diag, x_diag * snr_line, '--', alpha=0.5, label=f'SNR={snr_line:.2f}')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    if particles is not None:
        indices = {'Low SNR': np.argmin(np.abs(snr - np.percentile(snr, 10))), 'Medium SNR': np.argmin(np.abs(snr - np.median(snr))), 'High SNR': np.argmin(np.abs(snr - np.percentile(snr, 90)))}
        for i, (label, idx) in enumerate(indices.items()):
            ax = fig.add_subplot(gs[1, i]); ax.imshow(particles[idx], cmap='gray'); ax.set_title(f'{label}\nSNR = {snr[idx]:.4f}'); ax.axis('off')
    plt.suptitle(f'SNR Analysis: {results["n_particles"]} Particles\nSNR = var(protein)/var(ice) where var(protein)=var(center)-var(ice)'); plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ============================================================================
# 4. PHYSICAL PARAMETER ESTIMATION MODULE
# ============================================================================
def estimate_physical_parameters(image_stack, signal_mask, background_mask, lowpass_filter_sigma: float = 2.0, device: str = None) -> dict:
    print("\nEstimating physical parameters from real data...")
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Physical Param Estimator using device: {dev}")

    with torch.no_grad():
        stack_tensor = torch.from_numpy(image_stack).float().to(dev)
        signal_mask_dev, background_mask_dev = signal_mask.to(dev), background_mask.to(dev)
        ice_pixels = stack_tensor[:, background_mask_dev]
        estimated_ice_mean = ice_pixels.mean().item()
        filtered_stack_np = ndimage.gaussian_filter(image_stack, sigma=(0, lowpass_filter_sigma, lowpass_filter_sigma))
        filtered_stack = torch.from_numpy(filtered_stack_np).to(dev)
        estimated_ice_std = filtered_stack[:, background_mask_dev].std().item()
        var_measured_protein_region = torch.var(stack_tensor[:, signal_mask_dev])
        var_measured_ice_region = torch.var(ice_pixels)
        var_protein_estimated = var_measured_protein_region - var_measured_ice_region
        estimated_protein_std = torch.sqrt(var_protein_estimated).item() if var_protein_estimated > 0 else 0.0

    return {"ice_mean": estimated_ice_mean, "ice_std": estimated_ice_std, "protein_std": estimated_protein_std}

# ============================================================================
# 5. MAIN ANALYSIS PIPELINE AND REPORTING
# ============================================================================
def print_final_simulation_summary(results: dict):
    print("\n" + "="*80); print("      FINAL SIMULATION PARAMETER SUMMARY"); print("="*80)
    print("Use these parameters to generate realistic synthetic cryo-EM data.")
    phys_params = results.get('physical_params', {}); print("\n--- Physical Properties ---")
    print(f"  Ice Mean (background level):          {phys_params.get('ice_mean', 'N/A'):.4f}")
    print(f"  Ice Std (texture amplitude):          {phys_params.get('ice_std', 'N/A'):.4f}")
    print(f"  Protein Contrast Std (signal strength): {phys_params.get('protein_std', 'N/A'):.4f}")
    spec_results = results.get('spectrum_results', {}); params = spec_results.get('params'); params_std = spec_results.get('params_std')
    slope_rec, falloff_rec = None, None; print("\n--- Ice Texture Shape (Power Spectrum) ---")
    if params:
        if params_std and params_std['slope'] > 0 and params_std['falloff_freq'] > 0:
            range_method = "based on ±2σ of fit uncertainty"
            slope_rec = (max(0.5, params['slope'] - 2*params_std['slope']), min(5.0, params['slope'] + 2*params_std['slope']))
            falloff_rec = (max(0.001, params['falloff_freq'] - 2*params_std['falloff_freq']), min(0.5, params['falloff_freq'] + 2*params_std['falloff_freq']))
        else:
            range_method = "based on ±20% of mean value"
            slope_rec = (max(0.5, params['slope'] * 0.8), min(5.0, params['slope'] * 1.2))
            falloff_rec = (max(0.001, params['falloff_freq'] * 0.8), min(0.5, params['falloff_freq'] * 1.2))
        print(f"  (Ranges are {range_method})")
        print(f"  Slope Range:                          ({slope_rec[0]:.3f}, {slope_rec[1]:.3f})")
        print(f"  Falloff Frequency Range (Å⁻¹):        ({falloff_rec[0]:.4f}, {falloff_rec[1]:.4f})")
    else: print("  Spectrum analysis was not run or fitting failed.")
    snr_results = results.get('snr_results', {}); snr = snr_results.get('snr'); snr_robust_range = None
    print("\n--- Final Image Quality (SNR) ---")
    if snr is not None:
        snr = snr[np.isfinite(snr)]
        if len(snr) > 0:
            snr_rec_lower = max(0.001, np.percentile(snr, 10)); snr_rec = (snr_rec_lower, np.percentile(snr, 90))
            snr_q1, snr_q3 = np.percentile(snr, 25), np.percentile(snr, 75); snr_iqr = snr_q3 - snr_q1
            snr_robust_lower = max(0.001, snr_q1 - 1.5 * snr_iqr); snr_robust_upper = snr_q3 + 1.5 * snr_iqr
            snr_robust_range = (snr_robust_lower, snr_robust_upper)
            print("  (This is the final power SNR after all effects)")
            print(f"  Typical (Median):                     {np.median(snr):.4f}")
            print(f"  Target Range (10-90 percentile):      ({snr_rec[0]:.4f}, {snr_rec[1]:.4f})")
            print(f"  Robust Range (IQR-based):             ({snr_robust_range[0]:.4f}, {snr_robust_range[1]:.4f})")
        else: print("  SNR analysis yielded no valid values.")
    else: print("  SNR analysis was not run.")
    print("\n" + "-"*80); print("COPY-PASTE PARAMETERS FOR SIMULATOR:"); print("-"*80)
    if all([phys_params, slope_rec, falloff_rec, snr_robust_range]):
        print(f"simulation_params = {{")
        print(f"    'ice_mean': {phys_params['ice_mean']:.4f},")
        print(f"    'ice_std': {phys_params['ice_std']:.4f},")
        print(f"    'protein_contrast_std': {phys_params['protein_std']:.4f},")
        print(f"    'ice_slope_range': ({slope_rec[0]:.3f}, {slope_rec[1]:.3f}),")
        print(f"    'ice_falloff_freq_range': ({falloff_rec[0]:.4f}, {falloff_rec[1]:.4f}),")
        print(f"    'target_snr_range': ({snr_robust_range[0]:.4f}, {snr_robust_range[1]:.4f}),")
        print(f"}}")
    else: print("Could not generate complete parameter block as one or more analyses were skipped or failed.")
    print("="*80)

def run_comprehensive_analysis(args):
    print("[1/5] Reading particle stack...")
    reader = ParticleStackReader(args.input)
    particles = reader.read_stack(max_particles=args.max_particles)
    image_size = particles.shape[1]
    all_results = {}

    print("[2/5] Resolving mask radii...")
    pixel_args_provided = all(v is not None for v in [args.signal_radius_px, args.background_inner_px, args.background_outer_px])
    pixel_args_partially_provided = any(v is not None for v in [args.signal_radius_px, args.background_inner_px, args.background_outer_px])
    if pixel_args_partially_provided and not pixel_args_provided:
        sys.exit("ERROR: If using pixel-based radii, you must provide all three: --signal_radius_px, --background_inner_px, and --background_outer_px.")
    if pixel_args_provided:
        print("Using user-provided absolute pixel radii for masks.")
        signal_radius_px, bg_inner_px, bg_outer_px = args.signal_radius_px, args.background_inner_px, args.background_outer_px
    else:
        print("Calculating pixel radii from fractional arguments.")
        signal_radius_px, bg_inner_px, bg_outer_px = int(image_size * args.signal_radius), int(image_size * args.background_inner), int(image_size * args.background_outer)
    print(f"  Signal Radius: {signal_radius_px} px"); print(f"  Background Annulus: {bg_inner_px} px to {bg_outer_px} px")

    # Create masks once and pass them to all functions
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    signal_mask, background_mask = create_masks_gpu(image_size, signal_radius_px, bg_inner_px, bg_outer_px, device)
    
    # Store masks in results for potential visualization/debugging
    all_results['masks'] = {'signal': signal_mask.cpu().numpy(), 'background': background_mask.cpu().numpy()}

    if not args.no_spectrum:
        print("\n[3/5] Starting Ice Power Spectrum Analysis...")
        start_spec = time.time()
        spec_analyzer = GPUIcePowerSpectrumAnalyzer(pixel_size=args.pixel_size, device=device)
        freq, p_mean, p_std, all_p = spec_analyzer.analyze_stack_gpu(particles, background_mask, batch_size=args.batch_size)
        params, params_std = spec_analyzer.fit_ice_model(freq, p_mean)
        all_results['spectrum_results'] = {'params': params, 'params_std': params_std}
        if args.output:
            fig_spec = plot_power_spectrum_analysis(freq, p_mean, p_std, all_p, params)
            fig_spec.savefig(args.output / 'power_spectrum_analysis.png', dpi=150)
            np.savez(args.output / 'power_spectrum_data.npz', freq=freq, power_mean=p_mean, power_std=p_std)
        print(f"  Spectrum analysis took {time.time() - start_spec:.2f} seconds.")
    else: print("\n[3/5] Skipping Ice Power Spectrum Analysis.")
        
    if not args.no_snr:
        print("\n[4/5] Starting SNR and Physical Parameter Analysis...")
        start_snr = time.time()
        snr_analyzer = GPUSNRAnalyzer(device=device)
        snr_results = snr_analyzer.analyze_stack_gpu(particles, signal_mask, background_mask, batch_size=args.batch_size)
        all_results['snr_results'] = snr_results
        phys_params = estimate_physical_parameters(particles, signal_mask, background_mask, device=device)
        all_results['physical_params'] = phys_params
        if args.output:
            fig_snr = plot_snr_analysis(snr_results, particles)
            fig_snr.savefig(args.output / 'snr_analysis.png', dpi=150)
            df = pd.DataFrame({'snr': snr_results['snr']}); df.to_csv(args.output / 'snr_data.csv', index=False)
        print(f"  SNR & Physical Param analysis took {time.time() - start_snr:.2f} seconds.")
    else: print("\n[4/5] Skipping SNR and Physical Parameter Analysis.")

    print("\n[5/5] Generating Final Summary...")
    print_final_simulation_summary(all_results)
    
    if args.show_plots: print("Displaying plots..."); plt.show()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Cryo-EM Particle Stack Analyzer for Simulation Parameter Estimation.', formatter_class=argparse.RawTextHelpFormatter)
    g_core = parser.add_argument_group('Core Parameters'); g_core.add_argument('--input', '-i', required=True, help='Particle stack path (.mrc, .mrcs)')
    g_core.add_argument('--output', '-o', type=Path, default=Path("./sim_params"), help='Output directory to save plots and data files')
    g_core.add_argument('--pixel_size', '-p', type=float, default=1.0, help='Pixel size (Å/pixel) for spectrum analysis')
    g_core.add_argument('--max_particles', '-n', type=int, default=None, help='Max particles to analyze')
    g_core.add_argument('--device', '-d', type=str, default=None, help='Device: cuda or cpu (default: auto-detect)')
    g_core.add_argument('--batch_size', '-b', type=int, default=128, help='GPU batch size for all analyses')
    
    g_mask_frac = parser.add_argument_group('Masking (as fraction of image size -- used if pixel arguments are not provided)')
    g_mask_frac.add_argument('--signal_radius', type=float, default=0.5, help='SNR signal region radius fraction. (Default: 0.5 to match simulators)')
    g_mask_frac.add_argument('--background_inner', type=float, default=0.6, help='SNR/Ice background annulus inner radius fraction. (Default: 0.6)')
    g_mask_frac.add_argument('--background_outer', type=float, default=0.9, help='SNR/Ice background annulus outer radius fraction. (Default: 0.9)')
    
    g_mask_pix = parser.add_argument_group('Masking (in absolute pixels -- OVERRIDES fractional arguments)')
    g_mask_pix.add_argument('--signal_radius_px', type=int, default=None, help='SNR signal region radius in pixels.')
    g_mask_pix.add_argument('--background_inner_px', type=int, default=None, help='SNR/Ice background annulus inner radius in pixels.')
    g_mask_pix.add_argument('--background_outer_px', type=int, default=None, help='SNR/Ice background annulus outer radius in pixels.')

    g_ctrl = parser.add_argument_group('Execution Control')
    g_ctrl.add_argument('--no-snr', action='store_true', help="Skip SNR and Physical Parameter analysis.")
    g_ctrl.add_argument('--no-spectrum', action='store_true', help="Skip Ice Power Spectrum analysis.")
    g_ctrl.add_argument('--show-plots', action='store_true', help="Display generated plots at the end (in addition to saving them).")
    
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(exist_ok=True, parents=True)
        print(f"Results will be saved to: {args.output.resolve()}")
    run_comprehensive_analysis(args)
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
