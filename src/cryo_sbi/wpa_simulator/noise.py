import torch
from typing import Optional
from cryo_sbi.wpa_simulator.image_tools import circular_mask 
import math


def add_Gaussian_noise(
    image: torch.Tensor, 
    snr: torch.Tensor,
    mask_radius: Optional[float] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Adds Gaussian white noise to images based on power SNR.
    
    SNR definition: SNR = signal_variance / noise_variance
    Therefore: noise_std = signal_std / sqrt(SNR)
    
    Args:
        image: Image tensor of shape (batch, height, width)
        snr: Signal-to-noise ratio (power SNR)
             Typical values: 0.001 - 0.1 for single particles
        mask_radius: Radius for signal calculation. If None, uses image_size//2
        seed: Random seed for reproducibility
        
    Returns:
        Noisy image with same shape as input
    """
    
    if seed is not None:
        torch.manual_seed(seed)
    
    device = image.device
    n_pixels = image.shape[-1]
    
    # Create mask for signal region
    if mask_radius is None:
        mask_radius = 0.5 * n_pixels
    mask = circular_mask(n_pixels, mask_radius, device=device)
    
    # Calculate signal standard deviation within mask
    # mean image values are automatically subtracted by pytorch
    signal_std = torch.std(image[:, mask], dim=[-1])  # (batch,)
    
    # Calculate noise standard deviation from power SNR
    # SNR = σ²_signal / σ²_noise → σ_noise = σ_signal / sqrt(SNR) [batch, 1, 1]
    # Now we sample the correct prior distribution:
    # Default Log-Uniform (Jeffreys) - Uniform with USE_UNIFORM_SNR=true
    noise_std = signal_std.reshape(-1, 1, 1) / torch.sqrt(snr)
    
    # Generate noise map [batch, npixels, npixels]
    noise = torch.randn_like(image)
    noise = noise * noise_std
    # Final noisy image [batch, npixels, npixels] 
    image_noise = image + noise
    
    return image_noise


def add_Poisson_noise(
    image: torch.Tensor,
    target_snr: torch.Tensor,
    simulation_param: dict,
    mtf: Optional[torch.Tensor] = None,
    nps: Optional[torch.Tensor] = None,
    mask_radius: Optional[float] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Adds Poisson and readout noise to images.

    Args:
        image: Image tensor. Can be a clean signal or a composite of signal and
               structural noise. Shape: (batch, height, width).
        target_snr: Signal-to-noise target ratio (power SNR).
        simulation_param: Dictionary of simulation parameters.
        mtf (optional): MTF grid.
        nps (optional): NPS excess noise grid.
        mask_radius (optional): Radius for signal calculation.
        seed (optional): Random seed for reproducibility.

    Returns:
        Noisy image with the same shape as the input.
    """
    # 1. Setup and Parameter Extraction
    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size = simulation_param["pixel_size"].item() 
    else:
        pixel_size = simulation_param["pixel_size"]

    if seed is not None:
        torch.manual_seed(seed)
    
    device = image.device
    n_pixels = image.shape[-1]
    
    # 2. Calculate Signal Variance
    if mask_radius is None:
        mask_radius = 0.5 * n_pixels
    mask = circular_mask(n_pixels, mask_radius, device=device)
    
    # Calculate variance from the input image within the mask.
    signal_var = torch.var(image[:, mask], dim=[-1]).view(-1, 1, 1)

    # 3. Core Noise Simulation
    # Convert dose from e/Å² to e/pixel
    mean_electron_dose = simulation_param["dose"] * pixel_size**2

    # 4. Define quantum efficieny
    # With "Poisson" noise: use qe = DQE(0) = simulation_param["qe"]
    # With "Poisson-MTF": set qe=1.0 -> it will be taken care by MTF/DQE
    if mtf is None and nps is None:
       qe = simulation_param["qe"]
    else:
       qe = 1.0

    # 5. Determine the contrast scale for each image based on its target SNR
    # Deal with zero variance corner cases
    eps = torch.finfo(signal_var.dtype).eps
    contrast_scale = torch.where(
          signal_var > eps,
          torch.sqrt(target_snr / (mean_electron_dose * qe * signal_var)),
          torch.zeros_like(signal_var)
    )

    # 6. Determine the mean electron count per pixel
    mean_counts_per_pixel = qe * mean_electron_dose * (1.0 + contrast_scale * image)

    # 7. Generate Poisson shot noise
    # First, clamp mean_counts_per_pixel to be non-negative
    mean_counts_per_pixel = torch.clamp(mean_counts_per_pixel, min=0)
    image_noise = torch.poisson(mean_counts_per_pixel)

    # 8. Adding MTF/NPS corrections
    if mtf is not None and nps is not None:
       # 8.1 Apply Modulation Transfer Function 
       image_fft = torch.fft.fft2(image_noise)
       image_fft_mtf = image_fft * mtf
       image_noise = torch.fft.ifft2(image_fft_mtf).real

       # 8.2 Apply Detective Quantum Efficiency correction by adding excess noise
       # Generate white noise in Fourier space
       white_noise_fft = torch.fft.fft2(torch.randn_like(image_noise))
       
       # Scale by sqrt(NPS) to color the noise
       colored_noise_fft = white_noise_fft * torch.sqrt(nps)
       
       # Inverse FFT to get the noise in real space and add it
       colored_noise = torch.fft.ifft2(colored_noise_fft).real
       image_noise = image_noise + colored_noise 

    # 9. Add Gaussian Readout Noise
    readout_noise = torch.randn_like(image_noise) * simulation_param["readout_std"]
    final_image = image_noise + readout_noise

    return final_image
