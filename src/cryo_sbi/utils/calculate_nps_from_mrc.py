#!/usr/bin/env python
# compute_and_symmetrize_nps.py
"""
End-to-end Noise Power Spectrum (NPS) Calculator for Cryo-EM Particle Stacks.

This script performs a two-stage process:
1.  Calculates the raw 2D NPS from the background (ice) region of a particle stack.
2.  Optionally, it makes the resulting 2D NPS radially symmetric via azimuthal averaging
    to produce a clean, 1D noise profile represented as a 2D image.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import time

# --- Dependency Checks ---
try:
    import mrcfile
except ImportError:
    sys.exit("This script requires 'mrcfile'. Please install it: pip install mrcfile")

try:
    from scipy import ndimage
except ImportError:
    sys.exit("This script requires 'scipy'. Please install it: pip install scipy")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs): return iterator

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: 'matplotlib' not found. Plotting will be disabled.")
    plt = None


# ============================================================================
# 2. UTILITY AND I/O MODULES (from calculate_nps_from_mrc.py)
# ============================================================================
class ParticleStackReader:
    # ... (code from calculate_nps_from_mrc.py is unchanged)
    """Read particle stacks from MRC/MRCS files"""
    def __init__(self, stack_path):
        self.stack_path = Path(stack_path)
        if not self.stack_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.stack_path}")

    def read_stack(self, max_particles=None):
        if self.stack_path.suffix in ['.mrc', '.mrcs']:
            with mrcfile.open(self.stack_path, permissive=True) as mrc:
                data = mrc.data
                if data.ndim == 2:
                    particles = data[np.newaxis, :, :]
                elif data.ndim == 3:
                    particles = data
                else:
                    raise ValueError(f"Unexpected data dimensions: {data.ndim}")
                
                if max_particles is not None:
                    particles = particles[:max_particles]
                    
                print(f"Read {len(particles)} particles of size {particles.shape[1]}x{particles.shape[2]}")
                return particles
        else:
            raise ValueError(f"Unsupported file format: {self.stack_path.suffix}")

def create_masks_gpu(size, signal_radius_px, bg_inner_px, bg_outer_px, device):
    # ... (code from calculate_nps_from_mrc.py is unchanged)
    """
    Creates hard-edged signal and background masks on the specified device.
    For NPS calculation, the background_mask is the primary mask of interest.
    """
    # Create a coordinate grid pre-centered at (0,0).
    grid = torch.linspace(-0.5 * (size - 1), 0.5 * (size - 1), size, device=device)
    
    # Calculate squared distance from the center for every pixel.
    r_squared = grid[None, :] ** 2 + grid[:, None] ** 2
    
    # Create masks with an exclusive boundary condition (< radius**2).
    signal_mask = (r_squared < signal_radius_px**2)
    background_mask = ((r_squared > bg_inner_px**2) & (r_squared < bg_outer_px**2))
    
    return signal_mask, background_mask

# ============================================================================
# 3. CORE CALCULATION MODULES (from both scripts)
# ============================================================================
def calculate_nps_stack_gpu(particles, background_mask, device, batch_size=32):
    # ... (code from calculate_nps_from_mrc.py is almost unchanged)
    # This version returns a PyTorch tensor instead of a NumPy array
    """
    Calculates the average 2D Noise Power Spectrum from a particle stack on the GPU.
    """
    n_particles, size, _ = particles.shape
    print(f"Calculating raw 2D NPS for {n_particles} particles on {device} (batch_size={batch_size})...")

    total_power_spectrum_2d = torch.zeros((size, size), dtype=torch.float32, device=device)
    n_bg_pixels = background_mask.sum()
    if n_bg_pixels == 0:
        raise ValueError("The provided background mask is empty. Check your radii.")
    
    background_mask_exp = background_mask.unsqueeze(0)

    n_batches = (n_particles + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Computing 2D Power Spectra"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_particles)
        
        batch_tensor = torch.from_numpy(particles[start_idx:end_idx]).float().to(device)
        
        mean_vals = (batch_tensor * background_mask_exp).sum(dim=(1, 2), keepdim=True) / n_bg_pixels
        masked_batch = (batch_tensor - mean_vals) * background_mask_exp

        fft_2d = torch.fft.fft2(masked_batch, dim=(-2, -1))
        power_2d = torch.abs(fft_2d)**2
        
        total_power_spectrum_2d += power_2d.sum(dim=0)

        del batch_tensor, masked_batch, fft_2d, power_2d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    average_nps = total_power_spectrum_2d / n_particles
    
    return average_nps # Return the tensor on its current device


def symmetrize_nps_torch(nps_tensor: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    # ... (code from simmetrize_NPS.py is unchanged)
    """
    Takes a 2D NPS tensor and makes it radially symmetric via azimuthal averaging.
    """
    if nps_tensor.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, but got {nps_tensor.ndim} dimensions.")

    original_device = nps_tensor.device
    nps_grid_np = nps_tensor.cpu().numpy()
    h, w = nps_grid_np.shape

    freq_y = np.fft.fftfreq(h)
    freq_x = np.fft.fftfreq(w)
    fx_grid, fy_grid = np.meshgrid(freq_x, freq_y, indexing='ij')
    radial_freq_px = np.sqrt((fx_grid * h)**2 + (fy_grid * w)**2)
    
    radial_bins = radial_freq_px.astype(int)
    
    max_bin = np.max(radial_bins)
    radii = np.arange(max_bin + 1)
    nps_profile_1d = ndimage.mean(nps_grid_np, labels=radial_bins, index=radii)
    
    valid_indices = ~np.isnan(nps_profile_1d)
    radii = radii[valid_indices]
    nps_profile_1d = nps_profile_1d[valid_indices]

    full_profile = np.zeros(max_bin + 1)
    full_profile[radii] = nps_profile_1d
    
    symmetric_nps_grid_np = full_profile[radial_bins]
    
    symmetric_nps_tensor = torch.from_numpy(symmetric_nps_grid_np).to(
        dtype=nps_tensor.dtype, device=original_device
    )
    
    return symmetric_nps_tensor, (radii, nps_profile_1d)

# ============================================================================
# 4. MAIN ANALYSIS PIPELINE
# ============================================================================
def save_mrc(data_tensor, output_path_str):
    """Helper function to save a torch tensor to an MRC file."""
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data_tensor.cpu().numpy().astype(np.float32))
    print(f"✓ Saved grid to: {output_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate and optionally symmetrize the 2D NPS from a cryo-EM particle stack.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    g_io = parser.add_argument_group('Input/Output Parameters')
    g_io.add_argument('--input', '-i', required=True, help='Input particle stack path (.mrc, .mrcs)')
    g_io.add_argument('--output_raw', type=str, default=None, help='(Optional) Output path for the raw, non-symmetric NPS grid (.mrc).')
    g_io.add_argument('--output', '-o', required=True, help='Output path for the final, symmetric NPS grid (.mrc).')
    
    g_core = parser.add_argument_group('Core Calculation Parameters')
    g_core.add_argument('--max_particles', '-n', type=int, default=None, help='Max number of particles to analyze (default: all)')
    g_core.add_argument('--device', '-d', type=str, default=None, help='Computation device: "cuda" or "cpu" (default: auto-detect)')
    g_core.add_argument('--batch_size', '-b', type=int, default=128, help='GPU batch size for processing (default: 128)')
    g_core.add_argument('--plot', action='store_true', help='Show a diagnostic plot of the raw vs. symmetric grids and the 1D profile.')

    # Masking arguments are unchanged
    g_mask_frac = parser.add_argument_group('Masking (as fraction of image size -- used if pixel arguments are not provided)')
    g_mask_frac.add_argument('--signal_radius', type=float, default=0.5, help='Radius of central signal region to EXCLUDE (default: 0.5)')
    g_mask_frac.add_argument('--background_inner', type=float, default=0.6, help='Inner radius of background annulus to INCLUDE (default: 0.6)')
    g_mask_frac.add_argument('--background_outer', type=float, default=0.9, help='Outer radius of background annulus to INCLUDE (default: 0.9)')
    
    g_mask_pix = parser.add_argument_group('Masking (in absolute pixels -- OVERRIDES fractional arguments)')
    g_mask_pix.add_argument('--signal_radius_px', type=int, default=None, help='Radius of central signal region to EXCLUDE, in pixels.')
    g_mask_pix.add_argument('--background_inner_px', type=int, default=None, help='Inner radius of background annulus to INCLUDE, in pixels.')
    g_mask_pix.add_argument('--background_outer_px', type=int, default=None, help='Outer radius of background annulus to INCLUDE, in pixels.')

    args = parser.parse_args()
    start_time = time.time()
    
    # --- 1. Load Data ---
    print("[1/5] Reading particle stack...")
    reader = ParticleStackReader(args.input)
    particles = reader.read_stack(max_particles=args.max_particles)
    image_size = particles.shape[1]

    # --- 2. Create Masks ---
    print("\n[2/5] Resolving mask radii...")
    # (Logic to determine pixel radii is unchanged)
    pixel_args_provided = all(v is not None for v in [args.signal_radius_px, args.background_inner_px, args.background_outer_px])
    if pixel_args_provided:
        signal_radius_px, bg_inner_px, bg_outer_px = args.signal_radius_px, args.background_inner_px, args.background_outer_px
    else:
        signal_radius_px = int(image_size * args.signal_radius)
        bg_inner_px = int(image_size * args.background_inner)
        bg_outer_px = int(image_size * args.background_outer)
    print(f"  Signal Radius: {signal_radius_px} px | Background Annulus: {bg_inner_px} px to {bg_outer_px} px")
    
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    _, background_mask = create_masks_gpu(image_size, signal_radius_px, bg_inner_px, bg_outer_px, device)
    
    # --- 3. Calculate Raw 2D NPS ---
    print("\n[3/5] Starting Raw NPS Calculation...")
    raw_nps_tensor = calculate_nps_stack_gpu(particles, background_mask, device, args.batch_size)
    
    if args.output_raw:
        print("\n[4/5] Saving raw NPS grid...")
        save_mrc(raw_nps_tensor, args.output_raw)
    else:
        print("\n[4/5] Skipping raw NPS save.")
    
    # --- 4. Symmetrize NPS ---
    print("\n[5/5] Symmetrizing NPS and saving final result...")
    symmetric_nps_tensor, diagnostics = symmetrize_nps_torch(raw_nps_tensor)
    save_mrc(symmetric_nps_tensor, args.output)
        
    total_time = time.time() - start_time
    print(f"\n✓ NPS processing complete in {total_time:.2f} seconds.")

    # --- 5. Optional Plotting ---
    if args.plot and plt is not None:
        radii, profile_1d = diagnostics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(np.log(np.fft.fftshift(raw_nps_tensor.cpu().numpy()) + 1e-9), cmap='viridis')
        axes[0].set_title('Raw 2D NPS')
        fig.colorbar(im1, ax=axes[0], label='Log(Power)')
        
        im2 = axes[1].imshow(np.log(np.fft.fftshift(symmetric_nps_tensor.cpu().numpy()) + 1e-9), cmap='viridis')
        axes[1].set_title('Symmetrized 2D NPS')
        fig.colorbar(im2, ax=axes[1], label='Log(Power)')
        
        axes[2].semilogy(radii, profile_1d, '.-')
        axes[2].set_title('1D Radial Profile')
        axes[2].set_xlabel('Radial Frequency (pixels)')
        axes[2].set_ylabel('Averaged Power (log scale)')
        axes[2].grid(True, which="both", ls="--")
        if len(radii) > 1: axes[2].set_xlim(left=0)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
