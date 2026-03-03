#!/usr/bin/env python3
"""
calculate_nps_ctf_zeros.py (v2.4)

Calculates the 2D Noise Power Spectrum (NPS) from a cryo-EM particle stack
and its corresponding STAR file by sampling the power spectrum at the zeros
of the Contrast Transfer Function (CTF).
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import argparse
import numpy as np
import torch
import starfile
import mrcfile
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import ndimage
import warnings
import sys

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mrcfile')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

# ============================================================================
# 2. MRC FILE HANDLING (UPGRADED FOR ROBUSTNESS AND MEMORY EFFICIENCY)
# ============================================================================
def check_mrc_file_size(filepath):
    """Check MRC file size in bytes and GB."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    return file_size, file_size / (1024**3)

def validate_mrc_data(data):
    """
    Validate MRC data after reading. This version is 'memmap-aware' to avoid
    loading entire large files into memory for validation.
    """
    if data is None or data.size == 0 or data.ndim not in [2, 3]:
        return False, f"Invalid data shape or type: {data.shape if hasattr(data, 'shape') else 'None'}"
    try:
        # For memmap, only check the first particle to avoid loading all data.
        if isinstance(data, np.memmap):
            test_data = data[0] if data.ndim == 3 else data
        else:
            test_data = data
        if np.all(test_data == 0): return False, "All data is zero"
        if np.any(np.isnan(test_data)): return False, "Data contains NaN"
        if np.any(np.isinf(test_data)): return False, "Data contains inf"
        if np.std(test_data) == 0: return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def read_mrc_header_raw(filepath):
    """Read MRC header manually if standard methods fail."""
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
    if nz > 50000000: return False, f"Stack too large: {nz}"
    return True, "Valid"

def open_mrc_robust(filepath, max_size_gb=None):
    """
    Robustly open MRC file with fallback methods, prioritizing memory-mapping
    to avoid loading large files into RAM.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None, False, "File not found"
    
    file_size, file_size_gb = check_mrc_file_size(filepath)
    if max_size_gb is not None and file_size_gb > max_size_gb:
        return None, False, f"Too large: {file_size_gb:.2f} GB"
    
    try:
        header_info = read_mrc_header_raw(filepath)
        if header_info is not None:
            nx, ny, nz, mode = header_info['nx'], header_info['ny'], header_info['nz'], header_info['mode']
            is_valid, msg = validate_mrc_dimensions(nx, ny, nz)
            if not is_valid:
                return None, False, msg
            
            dtype = get_dtype_from_mode(mode)
            data = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            
            is_valid, msg = validate_mrc_data(data)
            if is_valid:
                return data, True, f"Memmap via manual header read"
    except Exception as e:
        return None, False, f"Failed: {str(e)[:100]}"
    
    return None, False, "All MRC opening methods failed"

# ============================================================================
# 3. PARAMETER EXTRACTION
# ============================================================================
def extract_star_info(star_file):
    """Reads STAR file and extracts dataframe and other info."""
    print("Extracting CTF parameters from STAR file...")
    try:
        data = starfile.read(star_file)
    except Exception as e:
        sys.exit(f"❌ Error reading STAR file: {e}")

    particles = data.get('particles', data)
    num_particles = len(particles)
    print(f"✓ Found {num_particles:,} parameter sets in STAR file.")

    star_image_size = None
    if 'rlnImageSize' in particles.columns:
        star_image_size = int(particles['rlnImageSize'].values[0])

    return particles, star_image_size

def extract_ctf_parameters(star_file):
    data = starfile.read(star_file)
    particles = data.get('particles', data)
    params = {}
    if 'rlnVoltage' in particles.columns:
        params['voltage'] = float(particles['rlnVoltage'].values[0])
    else: params['voltage'] = 300.0
    if 'rlnSphericalAberration' in particles.columns:
        params['spherical_aberration'] = float(particles['rlnSphericalAberration'].values[0])
    else: params['spherical_aberration'] = 2.7
    return params

def extract_amplitude_contrast(star_file):
    data = starfile.read(star_file)
    particles = data.get('particles', data) if 'particles' in data else data
    if 'rlnAmplitudeContrast' in particles.columns:
        return float(particles['rlnAmplitudeContrast'].values[0])
    return 0.1

# ============================================================================
# 4. CORE NPS CALCULATION LOGIC & PLOTTING
# ============================================================================
def analytical_nps_model(q, a, b, c):
    """Analytical model for camera MTF + shot noise: A*exp(-B*q^2) + C."""
    return a * np.exp(-b * q**2) + c

def calculate_ctf_2d_torch(params, freq_sq, angle_grid, device):
    defocus_angle_rad = torch.deg2rad(torch.tensor(params['defocus_angle'], device=device))
    defocus_avg = (params['defocus_u'] + params['defocus_v']) / 2.0
    defocus_dev = (params['defocus_u'] - params['defocus_v']) / 2.0
    defocus_astig = defocus_avg + defocus_dev * torch.cos(2 * (angle_grid - defocus_angle_rad))
    lambda_ = 12.2643247 / np.sqrt(params['voltage'] * 1000 + 0.978466 * (params['voltage'] * 1000)**2 / 1e6)
    gamma = 2 * np.pi * (-0.5 * defocus_astig * lambda_ * freq_sq + 0.25 * params['cs'] * (lambda_**3) * (freq_sq**2))
    if 'phase_shift' in params: gamma += torch.deg2rad(torch.tensor(params['phase_shift'], device=device))
    ctf = -(torch.sqrt(torch.tensor(1 - params['amp_contrast']**2, device=device)) * torch.sin(gamma) - params['amp_contrast'] * torch.cos(gamma))
    return ctf

def running_average_smooth(data, window_size):
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_data, weights, mode='valid')
    return smoothed

def reconstruct_2d_nps_from_1d(nps_profile_1d, output_size):
    freq_y_px = np.fft.fftfreq(output_size) * output_size
    freq_x_px = np.fft.fftfreq(output_size) * output_size
    fy_grid_px, fx_grid_px = np.meshgrid(freq_y_px, freq_x_px, indexing='ij')
    radial_freq_px = np.sqrt(fx_grid_px**2 + fy_grid_px**2).astype(int)
    max_index = len(nps_profile_1d) - 1
    radial_freq_px[radial_freq_px > max_index] = max_index
    symmetric_nps_grid = nps_profile_1d[radial_freq_px]
    return symmetric_nps_grid

def plot_nps_profile(raw_radii, raw_nps, final_radii, final_nps, output_path):
    if not PLOTTING_ENABLED: print("⚠️ Plotting disabled. `matplotlib` and `seaborn` are required."); return
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(raw_radii, raw_nps, s=15, alpha=0.4, label='Raw Binned Data', color='cornflowerblue')
    ax.plot(final_radii, final_nps, 'r-', lw=2.5, label='Final Analytical Model (A*exp(-Bq²)+C)')
    ax.set_title('1D Noise Power Spectrum Profile', fontsize=16)
    ax.set_xlabel('Spatial Frequency (pixels)', fontsize=12)
    ax.set_ylabel('Average Noise Power', fontsize=12)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(); ax.grid(True, which="both", ls="--")
    if len(raw_radii) > 1: ax.set_xlim(left=max(0.5, raw_radii[1] / 2))
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Diagnostic plot saved to: {output_path}")
    except Exception as e: print(f"❌ Failed to save plot: {e}")
    plt.close(fig)

# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Calculate NPS by sampling at CTF zeros from a particle stack and STAR file.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_stack', required=True, help="Input particle stack (.mrcs)")
    parser.add_argument('--in_star', required=True, help="Input STAR file with CTF parameters")
    parser.add_argument('--output_nps', '-o', required=True, help="Output path for the symmetric NPS grid (.mrc)")
    parser.add_argument('--pixel_size', type=float, default=None, help="Override pixel size (Å/px). Highest priority.")
    parser.add_argument('--zero_threshold', type=float, default=0.08, help="Threshold to define CTF zeros (default: 0.08)")
    parser.add_argument('--device', default=None, help="Computation device ('cuda' or 'cpu'). Defaults to auto-detect.")
    parser.add_argument('--stride', type=int, default=1, help="Sample the STAR file with a stride. For the i-th MRC particle, the (i * stride)-th STAR entry\nis used. The STAR file must be large enough to accommodate this. Default is 1 (one-to-one).")
    parser.add_argument('--plot', action='store_true', help="Generate a diagnostic plot of the 1D NPS profile.")
    parser.add_argument('--exclude_first', type=int, default=0,
                        help="Exclude the first N data points from the analytical fit by giving them zero weight. (Default: 0)")
    args = parser.parse_args()
    
    print("\n[1/5] Reading files and extracting parameters...")
    
    particles_data, success, method = open_mrc_robust(args.in_stack)
    if not success: sys.exit(f"❌ Failed to read MRC file: {method}")
    print(f"✓ Particle stack opened successfully ({method})")
    input_size = particles_data.shape[-1]
    print(f"✓ Image size from MRC data is the definitive source: {input_size}x{input_size} pixels")

    particles_df, star_image_size = extract_star_info(args.in_star)
    ctf_params = extract_ctf_parameters(args.in_star)
    amp_contrast = extract_amplitude_contrast(args.in_star)
    mic_params = { 'voltage': ctf_params['voltage'], 'cs': ctf_params['spherical_aberration'], 'amp_contrast': amp_contrast }
    print(f"✓ Using Microscope Parameters: Voltage={mic_params['voltage']}kV, Cs={mic_params['cs']}mm, AmpContrast={mic_params['amp_contrast']:.2f}")

    if star_image_size is not None and star_image_size != input_size:
        print(f"⚠️ Warning: Image size in STAR file ({star_image_size}px) does not match "
              f"image size in MRC stack ({input_size}px). Using the MRC size as the source of truth.")

    if args.pixel_size is not None:
        apix = args.pixel_size
        print(f"✓ Using user-provided pixel size: {apix:.3f} Å/px")
    else:
        with mrcfile.open(args.in_stack, permissive=True) as mrc:
            apix = float(mrc.voxel_size.x)
        print(f"✓ Pixel size from MRC header: {apix:.3f} Å/px")
    
    if apix is None or apix < 0.1: sys.exit("❌ Could not determine a valid pixel size.")
    print(f"✓ Using final image parameters: Box Size={input_size}px, Pixel Size={apix:.3f} Å/px")

    print("\n[2/5] Setting up for processing...")
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"  - Using device: {device}")
    
    num_mrc_particles = len(particles_data)
    num_star_entries = len(particles_df)

    if args.stride == 1:
        if num_mrc_particles != num_star_entries:
            sys.exit(f"❌ Particle count mismatch (stride=1).\n"
                     f"   - MRC stack has {num_mrc_particles:,} particles.\n"
                     f"   - STAR file has {num_star_entries:,} entries.\n"
                     f"   - These must be equal when stride is 1.")
    else:
        print(f"✓ Using stride: {args.stride}. For MRC particle `i`, STAR entry `i * {args.stride}` will be used.")
        last_required_star_index = (num_mrc_particles - 1) * args.stride
        if num_star_entries < last_required_star_index + 1:
            sys.exit(f"❌ STAR file is too small for the given stride.\n"
                     f"   - MRC stack has {num_mrc_particles:,} particles.\n"
                     f"   - To process the last particle (index {num_mrc_particles-1:,}) with stride={args.stride}, we need to access STAR entry at index {last_required_star_index:,}.\n"
                     f"   - This requires the STAR file to have at least {last_required_star_index+1:,} entries.\n"
                     f"   - However, the STAR file only has {num_star_entries:,} entries.")

    indices_to_process = range(num_mrc_particles)
    num_particles_processed = num_mrc_particles

    print(f"  -> Total particles in stack: {num_mrc_particles:,}")
    print(f"  -> Particles to be processed: {num_particles_processed:,}")

    print("\n[3/5] Pre-calculating static frequency grids for GPU...")
    freq_A = torch.fft.fftfreq(input_size, d=apix, device=device)
    fy_A, fx_A = torch.meshgrid(freq_A, freq_A, indexing='ij')
    freq_sq_grid = fx_A**2 + fy_A**2
    angle_grid = torch.atan2(fy_A, fx_A)
    freq_px = torch.fft.fftfreq(input_size, d=1.0, device=device) * input_size
    fy_px, fx_px = torch.meshgrid(freq_px, freq_px, indexing='ij')
    radial_freq_px_grid = torch.sqrt(fy_px**2 + fx_px**2)
    
    print("\n[4/5] Sampling power spectrum at CTF zeros...")

    max_radius = int(np.ceil(input_size * np.sqrt(2)))
    sum_power_accumulator = np.zeros(max_radius + 1, dtype=np.float64)
    counts_accumulator = np.zeros(max_radius + 1, dtype=np.int64)
    total_samples_collected = 0

    with torch.no_grad():
        for i in tqdm(indices_to_process, desc="Processing particles"):
            particle_image = torch.from_numpy(particles_data[i].astype(np.float32)).to(device)
            star_idx = i * args.stride
            per_particle_params = mic_params.copy()
            per_particle_params.update({
                'defocus_u': particles_df.loc[star_idx, 'rlnDefocusU'], 'defocus_v': particles_df.loc[star_idx, 'rlnDefocusV'],
                'defocus_angle': particles_df.loc[star_idx, 'rlnDefocusAngle'],
            })
            if 'rlnPhaseShift' in particles_df.columns:
                per_particle_params['phase_shift'] = particles_df.loc[star_idx, 'rlnPhaseShift']
            power_2d = torch.abs(torch.fft.fft2(particle_image))**2
            ctf_2d = calculate_ctf_2d_torch(per_particle_params, freq_sq_grid, angle_grid, device)
            zero_indices = torch.where(torch.abs(ctf_2d) < args.zero_threshold)
            if zero_indices[0].numel() > 0:
                sampled_power = power_2d[zero_indices].cpu().numpy()
                sampled_radii = radial_freq_px_grid[zero_indices].cpu().numpy().astype(int)
                total_samples_collected += len(sampled_power)
                sum_power_accumulator += np.bincount(sampled_radii, weights=sampled_power, minlength=len(sum_power_accumulator))
                counts_accumulator += np.bincount(sampled_radii, minlength=len(counts_accumulator))

    del particles_data, particle_image, power_2d, ctf_2d

    print("\n[5/5] Binning, fitting, and reconstructing NPS...")
    if total_samples_collected == 0: sys.exit("❌ No power samples collected. Try increasing --zero_threshold.")

    valid_bins = counts_accumulator > 0
    radii_bins = np.arange(len(counts_accumulator))
    radii_bins_valid = radii_bins[valid_bins]
    
    nps_profile_1d_valid = np.zeros_like(radii_bins_valid, dtype=np.float64)
    nps_profile_1d_valid = sum_power_accumulator[valid_bins] / counts_accumulator[valid_bins]
    nps_profile_1d_valid /= num_particles_processed
    print(f"✓ Binned {total_samples_collected:,} samples into {len(radii_bins_valid)} radial bins.")
    
    output_size = input_size
    final_radii = np.arange(int(np.ceil(output_size * np.sqrt(2))) + 1)
    
    print("  - Preparing data for analytical fit...")
    if args.exclude_first < 0:
        sys.exit("❌ Error: --exclude_first must be a non-negative integer.")

    num_valid_points = len(radii_bins_valid)
    
    # Create a sigma array for weighted least squares fitting.
    # Points to be excluded are given an infinite sigma, which corresponds to zero weight.
    sigma_for_fit = np.ones(num_valid_points)

    if args.exclude_first > 0:
        sigma_for_fit[:args.exclude_first] = np.inf
        print(f"  - Excluding the first {args.exclude_first} points from the fit.")

    # Automatically exclude points beyond Nyquist frequency
    nyquist_px = input_size / 2.0
    nyquist_indices_to_exclude = np.where(radii_bins_valid > nyquist_px)[0]
    if len(nyquist_indices_to_exclude) > 0:
        sigma_for_fit[nyquist_indices_to_exclude] = np.inf
        print(f"  - Automatically excluding {len(nyquist_indices_to_exclude)} data points beyond Nyquist ({nyquist_px:.1f} pixels).")
    
    # Validate that enough points remain for fitting
    num_points_for_fit = np.sum(np.isfinite(sigma_for_fit))
    if num_points_for_fit < 3: # Need at least 3 points to fit 3 parameters
         sys.exit(f"❌ Not enough data points ({num_points_for_fit}) left for fitting after exclusions. Try reducing --exclude_first.")
    
    print("  - Fitting with analytical model: A * exp(-B*q^2) + C")
    p0 = [
        nps_profile_1d_valid[0] - nps_profile_1d_valid[-1],
        1e-4,
        nps_profile_1d_valid[-1]
    ]
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(
            analytical_nps_model,
            radii_bins_valid,
            nps_profile_1d_valid,
            p0=p0,
            bounds=bounds,
            sigma=sigma_for_fit
        )
        print(f"✓ Analytical fit successful. Parameters: A={popt[0]:.2e}, B={popt[1]:.2e}, C={popt[2]:.2e}")
        nps_profile_1d_final = analytical_nps_model(final_radii, *popt)
    except RuntimeError as e:
        sys.exit(f"❌ Analytical fit failed: {e}. Cannot proceed.")

    nps_profile_1d_final[nps_profile_1d_final < 0] = 0
    nps_profile_1d_final[0] = 0.0 
    
    symmetric_nps_grid = reconstruct_2d_nps_from_1d(nps_profile_1d_final, output_size)
    print(f"✓ Reconstructed NPS on a {output_size}x{output_size} grid.")

    print("\n[Last Step] Saving final NPS grid and generating plot...")
    output_path = Path(args.output_nps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(symmetric_nps_grid.astype(np.float32))
        mrc.voxel_size = 1.0 / (apix * output_size)
    print(f"\n✓ NPS calculation complete. Output saved to: {output_path.resolve()}")

    if args.plot:
        plot_output_path = output_path.with_suffix('.png')
        plot_nps_profile(
            raw_radii=radii_bins_valid,
            raw_nps=nps_profile_1d_valid,
            final_radii=final_radii,
            final_nps=nps_profile_1d_final,
            output_path=plot_output_path
        )

if __name__ == "__main__":
    main()
