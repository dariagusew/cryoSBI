#!/usr/bin/env python3
"""
calculate_nps_from_mrc_background_mask_v4.py

Calculates the 2D Noise Power Spectrum (NPS) from a cryo-EM particle stack
by applying a real-space mask to isolate the background (ice) region.

This script is a direct merger of two user-provided scripts:
1. MRC File Handling & Mask Definitions are taken EXACTLY from
   `estimate_snr_from_mrc.py`. The mask parameters are REQUIRED. This version
   corrects a bug by ensuring the pixel size is also read by this robust method.
2. The NPS processing pipeline (radial averaging, smoothing, interpolation,
   and reconstruction) is taken from `calculate_nps_from_mrc_CTF_zeros_v2.2.py`.

It does NOT require a STAR file.
"""

# ============================================================================
# 1. COMMON IMPORTS AND UTILITIES
# ============================================================================
import numpy as np
import torch
import mrcfile
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
import warnings
import sys
import gc
import argparse
from contextlib import contextmanager

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mrcfile')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

# =======================================================================================
# 2. ROBUST MRC FILE HANDLING
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
        try:
            with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
                nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
                dtype = mrc.data.dtype
            memmap_obj = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            if validate_mrc_data(memmap_obj)[0]:
                yield memmap_obj, True, "Memmap via mrcfile header"
                return
        except Exception:
            pass
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

class ParticleStackReader:
    """Read particle stacks using a robust, memory-mapped approach."""
    def __init__(self, stack_path):
        self.stack_path = Path(stack_path)
        self.apix = None

    def read_stack(self, max_particles=None):
        """Reads particle stack using memory-mapping and extracts pixel size."""
        if self.stack_path.suffix not in ['.mrc', '.mrcs']:
            raise ValueError(f"Unsupported file format: {self.stack_path.suffix}")
        
        try:
            with mrcfile.open(self.stack_path, permissive=True, header_only=True) as mrc:
                self.apix = float(mrc.voxel_size.x)
        except Exception:
            print("⚠️ Could not read pixel size from MRC header during initial check.")
            self.apix = None

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
# 3. ANALYSIS MASKING AND NPS UTILITIES
# ============================================================================
def create_masks_gpu(size, signal_radius_px, bg_inner_px, bg_outer_px, device):
    """Creates hard-edged signal and background masks."""
    grid = torch.linspace(-0.5 * (size - 1), 0.5 * (size - 1), size, device=device)
    r_squared = grid[None, :] ** 2 + grid[:, None] ** 2
    signal_mask = (r_squared < signal_radius_px**2)
    background_mask = ((r_squared > bg_inner_px**2) & (r_squared < bg_outer_px**2))
    return signal_mask, background_mask

def get_radial_indices_gpu(size, device):
    """Pre-calculates a grid of integer radial distances for binning."""
    freq_px = torch.fft.fftfreq(size, d=1.0) * size
    fy_px, fx_px = torch.meshgrid(freq_px, freq_px, indexing='ij')
    radial_freq_px_grid = torch.sqrt(fy_px**2 + fx_px**2)
    return radial_freq_px_grid.to(device).int()

def running_average_smooth(data, window_size):
    """Smooths 1D data using a running average."""
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_data, weights, mode='valid')
    return smoothed

def reconstruct_2d_nps_from_1d(nps_profile_1d, output_size):
    """Reconstructs a 2D rotationally symmetric grid from a 1D profile."""
    freq_y_px = np.fft.fftfreq(output_size) * output_size
    freq_x_px = np.fft.fftfreq(output_size) * output_size
    fy_grid_px, fx_grid_px = np.meshgrid(freq_y_px, freq_x_px, indexing='ij')
    radial_freq_px = np.sqrt(fx_grid_px**2 + fy_grid_px**2).astype(int)
    max_index = len(nps_profile_1d) - 1
    radial_freq_px[radial_freq_px > max_index] = max_index
    symmetric_nps_grid = nps_profile_1d[radial_freq_px]
    return symmetric_nps_grid

def plot_nps_profile(raw_radii, raw_nps, anchor_radii, anchor_nps, final_radii, final_nps, output_path):
    """Generates and saves a diagnostic plot of the 1D NPS profile."""
    if not PLOTTING_ENABLED:
        print("⚠️ Plotting disabled. `matplotlib` and `seaborn` are required.")
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(raw_radii, raw_nps, s=15, alpha=0.4, label='Raw Binned Data', color='cornflowerblue')
    ax.plot(final_radii, final_nps, 'r-', lw=2.5, label='Final Interpolated Model')
    ax.scatter(anchor_radii, anchor_nps, marker='x', color='black', s=50, label='Smoothed Anchor Points', zorder=10)
    ax.set_title('1D Noise Power Spectrum Profile (from Background Mask)', fontsize=16)
    ax.set_xlabel('Spatial Frequency (pixels)', fontsize=12)
    ax.set_ylabel('Average Noise Power Density', fontsize=12)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(); ax.grid(True, which="both", ls="--")
    if len(raw_radii) > 1:
        ax.set_xlim(left=max(0.5, raw_radii[1] / 2))
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Diagnostic plot saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save plot: {e}")
    plt.close(fig)

# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Calculate NPS by masking the particle and analyzing the background.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_stack', required=True, help="Input particle stack (.mrc, .mrcs)")
    parser.add_argument('--signal_radius', type=float, default=0.5, help="Radius of central signal mask as fraction of box size")
    parser.add_argument('--background_inner', type=float, default=0.6, help="Inner radius of background annulus as fraction of box size")
    parser.add_argument('--background_outer', type=float, default=0.9, help="Outer radius of background annulus as fraction of box size")
    parser.add_argument('--output_nps', '-o', required=True, help="Output path for the symmetric NPS grid (.mrc)")
    parser.add_argument('--pixel_size', type=float, default=None, help="Pixel size (Å/px). Overrides value from MRC header.")
    parser.add_argument('--fit_window_fraction', type=float, default=0.02, help="Window size for running average pre-smoothing as a fraction of the number of bins (default: 0.02).")
    parser.add_argument('--device', default=None, help="Computation device ('cuda' or 'cpu'). Defaults to auto-detect.")
    parser.add_argument('--plot', action='store_true', help="Generate a diagnostic plot of the 1D NPS profile.")
    parser.add_argument('--batch_size', type=int, default=32, help="Number of particles to process per batch on the GPU (default: 32).")
    args = parser.parse_args()
    
    print("\n[1/5] Reading particle stack...")
    reader = ParticleStackReader(args.in_stack)
    particles_data = reader.read_stack()
    num_particles, image_size, _ = particles_data.shape

    if args.pixel_size is not None:
        apix = args.pixel_size
        print(f"✓ Using user-provided pixel size: {apix:.3f} Å/px")
    else:
        apix = reader.apix
        if apix is not None:
             print(f"✓ Pixel size from MRC header: {apix:.3f} Å/px")
    
    if apix is None or apix < 0.1:
        sys.exit("❌ Could not determine a valid pixel size. Please provide it using --pixel_size.")

    print("\n[2/5] Setting up masks and GPU environment...")
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"  - Using device: {device}")

    signal_radius_px = int(image_size * args.signal_radius)
    bg_inner_px = int(image_size * args.background_inner)
    bg_outer_px = int(image_size * args.background_outer)
    print(f"  - Signal Radius (to exclude): {signal_radius_px} px")
    print(f"  - Background Annulus (to analyze): {bg_inner_px} px to {bg_outer_px} px")
    
    _, nps_mask = create_masks_gpu(image_size, signal_radius_px, bg_inner_px, bg_outer_px, device)
    n_mask_pixels = nps_mask.sum().item()
    if n_mask_pixels == 0:
        sys.exit("❌ The specified background mask is empty! Check your radius fractions.")
    print(f"  - Final mask for analysis contains {n_mask_pixels:,.0f} pixels.")

    print("\n[3/5] Pre-calculating static frequency grids for GPU...")
    radial_indices_gpu = get_radial_indices_gpu(image_size, device)
    max_radius = radial_indices_gpu.max().item()
    counts_per_bin_gpu = torch.bincount(radial_indices_gpu.flatten())
    counts_per_bin_cpu = counts_per_bin_gpu.cpu().numpy().astype(np.int64)

    print("\n[4/5] Processing particles and accumulating power spectra...")
    sum_power_accumulator = np.zeros(max_radius + 1, dtype=np.float64)
    n_batches = (num_particles + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Processing Batches"):
            batch_start = i * args.batch_size
            batch_end = min((i + 1) * args.batch_size, num_particles)
            batch_tensor = torch.from_numpy(particles_data[batch_start:batch_end].astype(np.float32)).to(device)
            masked_batch = batch_tensor * nps_mask
            power_batch = torch.abs(torch.fft.fft2(masked_batch))**2
            summed_power_2d = power_batch.sum(dim=0)
            binned_power = torch.bincount(radial_indices_gpu.flatten(), weights=summed_power_2d.flatten(), minlength=max_radius + 1)
            sum_power_accumulator += binned_power.cpu().numpy()
            del batch_tensor, masked_batch, power_batch, summed_power_2d
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    del particles_data; gc.collect()

    print("\n[5/5] Binning, fitting, and reconstructing NPS...")
    valid_bins = counts_per_bin_cpu > 0
    radii_bins_valid = np.arange(len(counts_per_bin_cpu))[valid_bins]
    avg_power_per_bin = sum_power_accumulator[valid_bins] / (counts_per_bin_cpu[valid_bins] * num_particles)
    nps_profile_1d_valid = avg_power_per_bin / n_mask_pixels
    print(f"✓ Binned power into {len(radii_bins_valid)} radial bins.")
    
    window_size = max(3, int(len(radii_bins_valid) * args.fit_window_fraction))
    if window_size % 2 == 0: window_size += 1
    print(f"  - Applying running average pre-smoothing with window size: {window_size} bins")
    anchor_points_y = running_average_smooth(nps_profile_1d_valid, window_size)

    output_size = image_size
    final_radii = np.arange(int(np.ceil(output_size / 2 * np.sqrt(2))) + 1)
    
    plateau_low = anchor_points_y[0]
    plateau_high = anchor_points_y[-1]
    fit_function = interp1d(radii_bins_valid, anchor_points_y, kind='linear',
                            bounds_error=False, fill_value=(plateau_low, plateau_high))
    
    nps_profile_1d_final = fit_function(final_radii)
    nps_profile_1d_final[nps_profile_1d_final < 0] = 0
    nps_profile_1d_final[0] = nps_profile_1d_final[1]
    
    symmetric_nps_grid = reconstruct_2d_nps_from_1d(nps_profile_1d_final, output_size)
    final_nps_grid_shifted = np.fft.fftshift(symmetric_nps_grid)
    print(f"✓ Reconstructed NPS on a {output_size}x{output_size} grid.")

    output_path = Path(args.output_nps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(final_nps_grid_shifted.astype(np.float32))
        mrc.voxel_size = apix
    print(f"\n✓ NPS calculation complete. Output saved to: {output_path.resolve()}")

    if args.plot:
        plot_output_path = output_path.with_suffix('.png')
        plot_nps_profile(
            raw_radii=radii_bins_valid,
            raw_nps=nps_profile_1d_valid,
            anchor_radii=radii_bins_valid,
            anchor_nps=anchor_points_y,
            final_radii=final_radii,
            final_nps=nps_profile_1d_final,
            output_path=plot_output_path
        )

if __name__ == "__main__":
    main()
