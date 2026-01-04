#!/usr/bin/env python
# calculate_nps_from_mrc.py
"""
Noise Power Spectrum (NPS) Calculator for Cryo-EM Particle Stacks

This script reads a particle stack, masks out the central signal region for each
particle, and computes the average 2D Noise Power Spectrum (NPS) of the 
remaining background (ice) region.

The resulting 2D NPS is saved as a standard MRC file, following the standard
FFT convention where the zero-frequency component is at the top-left corner
(index 0,0).
"""

# ============================================================================
# 1. COMMON IMPORTS AND UTILITIES
# ============================================================================
import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import time

try:
    import mrcfile
except ImportError:
    print("This script requires the 'mrcfile' package.")
    print("Please install it: pip install mrcfile")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, create a dummy function
    def tqdm(iterator, **kwargs):
        return iterator

class ParticleStackReader:
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
# 2. CORE NPS CALCULATION MODULE
# ============================================================================
def calculate_nps_stack_gpu(particles, background_mask, device, batch_size=32):
    """
    Calculates the average 2D Noise Power Spectrum from a particle stack on the GPU.

    Args:
        particles (np.ndarray): The particle stack (N, H, W).
        background_mask (torch.Tensor): A 2D boolean tensor (H, W) indicating the noise region.
        device (torch.device): The device to perform computations on.
        batch_size (int): Number of particles to process in each batch.

    Returns:
        np.ndarray: The final, average 2D NPS grid (DC component at corner).
    """
    n_particles, size, _ = particles.shape
    print(f"Calculating NPS for {n_particles} particles on {device} (batch_size={batch_size})...")

    # This tensor will accumulate the sum of all power spectra
    total_power_spectrum_2d = torch.zeros((size, size), dtype=torch.float32, device=device)
    
    # Pre-calculate the number of pixels in the mask for mean calculation
    n_bg_pixels = background_mask.sum()
    if n_bg_pixels == 0:
        raise ValueError("The provided background mask is empty. Check your radii.")
    
    background_mask_exp = background_mask.unsqueeze(0) # For broadcasting with batch

    n_batches = (n_particles + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Computing 2D Power Spectra"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_particles)
        
        # 1. Load batch to GPU
        batch_tensor = torch.from_numpy(particles[start_idx:end_idx]).float().to(device)
        
        # 2. Isolate background and make it zero-mean
        mean_vals = (batch_tensor * background_mask_exp).sum(dim=(1, 2), keepdim=True) / n_bg_pixels
        masked_batch = (batch_tensor - mean_vals) * background_mask_exp

        # 3. Compute 2D power spectrum for the batch
        fft_2d = torch.fft.fft2(masked_batch, dim=(-2, -1))
        power_2d = torch.abs(fft_2d)**2
        
        # 4. Add the sum of power spectra from this batch to the total
        total_power_spectrum_2d += power_2d.sum(dim=0)

        # 5. Clean up GPU memory
        del batch_tensor, masked_batch, fft_2d, power_2d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 6. Calculate the average NPS across all particles
    average_nps = total_power_spectrum_2d / n_particles
    
    return average_nps.cpu().numpy()


# ============================================================================
# 3. MAIN ANALYSIS PIPELINE
# ============================================================================
def run_nps_calculation(args):
    """Main function to orchestrate the NPS calculation process."""
    start_time = time.time()
    
    print("[1/4] Reading particle stack...")
    reader = ParticleStackReader(args.input)
    particles = reader.read_stack(max_particles=args.max_particles)
    image_size = particles.shape[1]

    print("[2/4] Resolving mask radii...")
    pixel_args_provided = all(v is not None for v in [args.signal_radius_px, args.background_inner_px, args.background_outer_px])
    pixel_args_partially_provided = any(v is not None for v in [args.signal_radius_px, args.background_inner_px, args.background_outer_px])

    if pixel_args_partially_provided and not pixel_args_provided:
        sys.exit("ERROR: If using pixel-based radii, you must provide all three: --signal_radius_px, --background_inner_px, and --background_outer_px.")

    if pixel_args_provided:
        print("Using user-provided absolute pixel radii for masks.")
        signal_radius_px = args.signal_radius_px
        bg_inner_px = args.background_inner_px
        bg_outer_px = args.background_outer_px
    else:
        print("Calculating pixel radii from fractional arguments.")
        signal_radius_px = int(image_size * args.signal_radius)
        bg_inner_px = int(image_size * args.background_inner)
        bg_outer_px = int(image_size * args.background_outer)
    
    print(f"  Signal Radius (to be excluded): {signal_radius_px} px")
    print(f"  Background Annulus (for NPS): {bg_inner_px} px to {bg_outer_px} px")
    
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    _, background_mask = create_masks_gpu(image_size, signal_radius_px, bg_inner_px, bg_outer_px, device)
    
    print("\n[3/4] Starting NPS Calculation...")
    nps_grid = calculate_nps_stack_gpu(particles, background_mask, device, args.batch_size)

    print("\n[4/4] Saving NPS grid to MRC file...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(nps_grid.astype(np.float32))
        print(f"Successfully saved un-shifted NPS grid to: {output_path.resolve()}")
        
    total_time = time.time() - start_time
    print(f"\n✓ NPS calculation complete in {total_time:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate the 2D Noise Power Spectrum (NPS) from a cryo-EM particle stack.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    g_core = parser.add_argument_group('Core Parameters')
    g_core.add_argument('--input', '-i', required=True, help='Input particle stack path (.mrc, .mrcs)')
    g_core.add_argument('--output', '-o', required=True, help='Output path for the NPS grid (.mrc file)')
    g_core.add_argument('--max_particles', '-n', type=int, default=None, help='Max number of particles to analyze from the stack (default: all)')
    g_core.add_argument('--device', '-d', type=str, default=None, help='Computation device: "cuda" or "cpu" (default: auto-detect)')
    g_core.add_argument('--batch_size', '-b', type=int, default=128, help='GPU batch size for processing (default: 128)')
    
    g_mask_frac = parser.add_argument_group('Masking (as fraction of image size -- used if pixel arguments are not provided)')
    g_mask_frac.add_argument('--signal_radius', type=float, default=0.5, help='Radius of central signal region to EXCLUDE, as fraction of image size (default: 0.5)')
    g_mask_frac.add_argument('--background_inner', type=float, default=0.6, help='Inner radius of background annulus to INCLUDE, as fraction of image size (default: 0.6)')
    g_mask_frac.add_argument('--background_outer', type=float, default=0.9, help='Outer radius of background annulus to INCLUDE, as fraction of image size (default: 0.9)')
    
    g_mask_pix = parser.add_argument_group('Masking (in absolute pixels -- OVERRIDES fractional arguments)')
    g_mask_pix.add_argument('--signal_radius_px', type=int, default=None, help='Radius of central signal region to EXCLUDE, in pixels.')
    g_mask_pix.add_argument('--background_inner_px', type=int, default=None, help='Inner radius of background annulus to INCLUDE, in pixels.')
    g_mask_pix.add_argument('--background_outer_px', type=int, default=None, help='Outer radius of background annulus to INCLUDE, in pixels.')

    args = parser.parse_args()
    
    if not args.output.lower().endswith('.mrc'):
        print("Warning: Output file does not have a .mrc extension. It will be saved in MRC format regardless.")

    run_nps_calculation(args)

if __name__ == "__main__":
    main()
