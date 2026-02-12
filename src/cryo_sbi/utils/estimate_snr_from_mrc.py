# "analyze_mrc_snr.py"
"""
Cryo-EM Particle Stack SNR Analyzer

This script reads a particle stack, calculates the Signal-to-Noise Ratio (SNR) for
each particle, and provides a statistical summary and plots of the results.

It uses a memory-mapped approach to handle arbitrarily large particle stacks without
loading them fully into RAM.
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
import pandas as pd
import time
import sys
import gc
from contextlib import contextmanager

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

# =======================================================================================
# 2. ROBUST MRC FILE HANDLING
#    Provides memory-efficient and resilient MRC file reading.
# =======================================================================================

def check_mrc_file_size(filepath):
    """Check MRC file size in bytes and GB."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    file_size_gb = file_size / (1024**3)
    return file_size, file_size_gb

def validate_mrc_data(data):
    """Validate MRC data after reading."""
    if data is None or data.size == 0 or data.ndim not in [2, 3]:
        return False, f"Invalid data shape or type: {data.shape if hasattr(data, 'shape') else 'None'}"
    try:
        # For memmap, only check first particle to avoid loading all
        test_data = data[0] if data.ndim == 3 else data
        if np.all(test_data == 0): return False, "All data is zero"
        if np.any(np.isnan(test_data)): return False, "Data contains NaN"
        if np.any(np.isinf(test_data)): return False, "Data contains inf"
        if np.std(test_data) == 0: return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def read_mrc_header_raw(filepath):
    """Read MRC header manually."""
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(1024)
            if len(header_bytes) < 1024: return None
            import struct
            nx, ny, nz = struct.unpack('iii', header_bytes[0:12])
            mode = struct.unpack('i', header_bytes[12:16])[0]
            return {'nx': nx, 'ny': ny, 'nz': nz, 'mode': mode}
    except:
        return None

def get_dtype_from_mode(mode):
    """Convert MRC mode to numpy dtype."""
    dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    return dtype_map.get(mode, np.float32)

def validate_mrc_dimensions(nx, ny, nz):
    """Check if dimensions are reasonable."""
    if nx <= 0 or ny <= 0 or nz <= 0: return False, f"Non-positive: {nz}×{ny}×{nx}"
    if nx > 8192 or ny > 8192: return False, f"Too large: {ny}×{nx}"
    return True, "Valid"

@contextmanager
def open_mrc_memmap(filepath):
    """Context manager for opening MRC as memmap (never loads into RAM)."""
    filepath = Path(filepath)
    memmap_obj = None
    try:
        if not filepath.exists():
            yield None, False, "File not found"
            return
        # Try standard mrcfile first (but don't load data)
        try:
            with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
                nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
                dtype = mrc.data.dtype
            memmap_obj = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            if validate_mrc_data(memmap_obj)[0]:
                yield memmap_obj, True, "Memmap via mrcfile header"
                return
        except Exception:
            pass # Fallback to manual reading
        # Fallback: force-read header
        header_info = read_mrc_header_raw(filepath)
        if header_info:
            nx, ny, nz, mode = header_info['nx'], header_info['ny'], header_info['nz'], header_info['mode']
            if not validate_mrc_dimensions(nx, ny, nz)[0]:
                yield None, False, f"Invalid dimensions from raw header: {nz}x{ny}x{nx}"
                return
            dtype = get_dtype_from_mode(mode)
            memmap_obj = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            if validate_mrc_data(memmap_obj)[0]:
                yield memmap_obj, True, "Memmap via manual header read"
                return
        yield None, False, "All MRC opening methods failed"
    finally:
        if memmap_obj is not None: del memmap_obj
        gc.collect()

# ============================================================================
# 3. PARTICLE STACK READER
# ============================================================================
class ParticleStackReader:
    """Read particle stacks from MRC/MRCS files using a robust, memory-mapped approach."""
    def __init__(self, stack_path):
        self.stack_path = Path(stack_path)

    def read_stack(self, max_particles=None):
        """Reads particle stack using memory-mapping. Returns a numpy.memmap object."""
        if self.stack_path.suffix not in ['.mrc', '.mrcs']:
            raise ValueError(f"Unsupported file format: {self.stack_path.suffix}")
        file_size, file_size_gb = check_mrc_file_size(self.stack_path)
        print(f"Input file size: {file_size_gb:.2f} GB. Using memory-mapped I/O.")
        with open_mrc_memmap(self.stack_path) as (data, success, msg):
            if not success:
                raise IOError(f"Failed to read MRC file '{self.stack_path}': {msg}")
            print(f"✓ Successfully opened stack via memory-map ({msg})")
            particles = data[np.newaxis, :, :] if data.ndim == 2 else data
            if max_particles is not None:
                particles = particles[:max_particles]
            print(f"Analyzing {len(particles)} particles of size {particles.shape[1]}x{particles.shape[2]}")
            return particles

# ============================================================================
# 4. ANALYSIS MASKING
# ============================================================================
def create_masks_gpu(size, signal_radius_px, bg_inner_px, bg_outer_px, device):
    """Creates hard-edged signal and background masks."""
    grid = torch.linspace(-0.5 * (size - 1), 0.5 * (size - 1), size, device=device)
    r_squared = grid[None, :] ** 2 + grid[:, None] ** 2
    signal_mask = (r_squared < signal_radius_px**2)
    background_mask = ((r_squared > bg_inner_px**2) & (r_squared < bg_outer_px**2))
    return signal_mask, background_mask

# ============================================================================
# 5. SNR ANALYSIS MODULE
# ============================================================================
class GPUSNRAnalyzer:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"SNR Analyzer using device: {self.device}")

    def compute_snr_batch(self, image_batch, signal_mask, background_mask):
        n_signal, n_background = signal_mask.sum(), background_mask.sum()
        if n_signal == 0 or n_background == 0:
            nan_tensor = torch.full((image_batch.shape[0],), float('nan'), device=self.device)
            return nan_tensor, nan_tensor, nan_tensor, nan_tensor
        signal_mean = (image_batch * signal_mask).sum(dim=(1, 2)) / n_signal
        signal_variance = ((image_batch - signal_mean.view(-1, 1, 1))**2 * signal_mask).sum(dim=(1, 2)) / n_signal
        background_mean = (image_batch * background_mask).sum(dim=(1, 2)) / n_background
        background_variance = ((image_batch - background_mean.view(-1, 1, 1))**2 * background_mask).sum(dim=(1, 2)) / n_background
        protein_variance = torch.clamp(signal_variance - background_variance, min=1e-10)
        snr_values = protein_variance / (background_variance + 1e-10)
        return snr_values, torch.sqrt(signal_variance), torch.sqrt(background_variance), torch.sqrt(protein_variance)

    def analyze_stack_gpu(self, particles, signal_mask, background_mask, batch_size=32):
        n_particles = len(particles)
        results = {'snr': [], 'signal_std': [], 'background_std': [], 'protein_std': []}
        n_batches = (n_particles + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Computing SNR"):
                batch_tensor = torch.from_numpy(particles[i*batch_size:min((i+1)*batch_size, n_particles)].astype(np.float32)).to(self.device)
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
    valid = np.isfinite(protein_var) & np.isfinite(ice_var)
    ax2.scatter(ice_var[valid], protein_var[valid], alpha=0.3, s=10)
    ax2.set_xlabel('Ice Variance'); ax2.set_ylabel('Protein Variance'); ax2.set_title('Protein vs Ice Variance')
    xlim = ax2.get_xlim(); x_diag = np.linspace(xlim[0], xlim[1], 100)
    for snr_line in [0.01, 0.05, 0.10]: ax2.plot(x_diag, x_diag * snr_line, '--', alpha=0.5, label=f'SNR={snr_line:.2f}')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    if particles is not None:
        indices = {'Low SNR': np.argmin(np.abs(snr - np.percentile(snr, 10))), 'Medium SNR': np.argmin(np.abs(snr - np.median(snr))), 'High SNR': np.argmin(np.abs(snr - np.percentile(snr, 90)))}
        for i, (label, idx) in enumerate(indices.items()):
            ax = fig.add_subplot(gs[1, i]); ax.imshow(particles[idx], cmap='gray', vmin=-4, vmax=4); ax.set_title(f'{label}\nSNR = {snr[idx]:.4f}'); ax.axis('off')
    plt.suptitle(f'SNR Analysis: {results["n_particles"]} Particles\nSNR = var(protein)/var(ice) where var(protein)=var(center)-var(ice)'); plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ============================================================================
# HIGH SNR IMAGES VIEWER
# ============================================================================
def plot_high_snr_grid(particles, snr_values, n_images=20, grid_shape=(4, 5)):
    """
    Create a grid visualization of the highest SNR images.
    
    Args:
        particles: Array of particle images
        snr_values: Array of SNR values corresponding to particles
        n_images: Number of highest SNR images to display (default: 20)
        grid_shape: Tuple (rows, cols) for the grid layout (default: 4x5)
    
    Returns:
        matplotlib.figure.Figure: Figure containing the grid of images
    """
    # Get indices of particles with valid SNR values
    valid_mask = np.isfinite(snr_values)
    valid_indices = np.where(valid_mask)[0]
    valid_snr = snr_values[valid_mask]
    
    # Get top n_images by SNR
    top_indices = valid_indices[np.argsort(valid_snr)[-n_images:]][::-1]  # Sort descending
    
    rows, cols = grid_shape
    fig = plt.figure(figsize=(16, 12))
    
    for i, particle_idx in enumerate(top_indices):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(particles[particle_idx], cmap='gray', vmin=-4, vmax=4)
        ax.set_title(f'SNR: {snr_values[particle_idx]:.4f}', fontsize=9)
        ax.axis('off')
    
    plt.suptitle(f'Top {n_images} Particles by SNR', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig


def save_pdf_multi_page(output_path, figures, pdf_filename='snr_analysis.pdf'):
    """
    Save multiple matplotlib figures to a single PDF file with multiple pages.
    
    Args:
        output_path: Path to output directory
        figures: List of matplotlib figures
        pdf_filename: Name of the output PDF file
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = output_path / pdf_filename
    
    with PdfPages(str(pdf_path)) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
    
    print(f"✓ Saved multi-page PDF to {pdf_path}")


# ============================================================================
# 6. MAIN ANALYSIS PIPELINE AND REPORTING
# ============================================================================
def print_snr_summary(snr_results: dict):
    print("\n" + "="*80); print("               SNR ANALYSIS SUMMARY"); print("="*80)
    
    snr = snr_results.get('snr')
    if snr is not None and len(snr_valid := snr[np.isfinite(snr)]) > 0:
        print(f"  Analyzed {len(snr_valid)} particles with valid SNR values.")
        print(f"  Mean SNR:                           {np.mean(snr_valid):.4f}")
        print(f"  Median SNR:                         {np.median(snr_valid):.4f}")
        print(f"  Std Dev of SNR:                     {np.std(snr_valid):.4f}")
        print(f"  Min SNR:                            {np.min(snr_valid):.4f}")
        print(f"  Max SNR:                            {np.max(snr_valid):.4f}")
        print(f"  Range (10-90 percentile):           ({np.percentile(snr_valid, 10):.4f}, {np.percentile(snr_valid, 90):.4f})")
    else:
        print("  SNR analysis was not run or yielded no valid values.")
    print("="*80)

def run_analysis(args):
    print("[1/3] Reading particle stack...")
    reader = ParticleStackReader(args.input)
    particles = reader.read_stack(max_particles=args.max_particles)
    image_size = particles.shape[1]

    print("\n[2/3] Resolving mask radii...")
    print("Calculating pixel radii from fractional arguments.")
    signal_radius_px = int(image_size * args.signal_radius)
    bg_inner_px = int(image_size * args.background_inner)
    bg_outer_px = int(image_size * args.background_outer)
    print(f"  Signal Radius: {signal_radius_px} px\n  Background Annulus: {bg_inner_px} px to {bg_outer_px} px")

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    signal_mask, background_mask = create_masks_gpu(image_size, signal_radius_px, bg_inner_px, bg_outer_px, device)
    
    print("\n[3/3] Starting SNR Analysis...")
    start_time = time.time()
    snr_analyzer = GPUSNRAnalyzer(device=device)
    snr_results = snr_analyzer.analyze_stack_gpu(particles, signal_mask, background_mask, batch_size=args.batch_size)
    
    if args.output:
        # Generate first page: SNR analysis plots
        fig_snr = plot_snr_analysis(snr_results, particles)
        
        # Generate second page: Grid of top 20 SNR images
        fig_grid = plot_high_snr_grid(particles, snr_results['snr'], n_images=20, grid_shape=(4, 5))
        
        # Save as multi-page PDF
        save_pdf_multi_page(args.output, [fig_snr, fig_grid], 'snr_analysis.pdf')
        
        # Save data to CSV
        pd.DataFrame(snr_results).to_csv(args.output / 'snr_data_all.csv', index=False)
        
        plt.close('all')
    
    print(f"  Analysis took {time.time() - start_time:.2f} seconds.")

    print_snr_summary(snr_results)
    
    del particles; gc.collect()
    if args.show_plots: print("Displaying plots..."); plt.show()