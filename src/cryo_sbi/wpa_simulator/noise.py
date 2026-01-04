import torch
from typing import Optional


def add_Gaussian_noise(
    image: torch.Tensor,
    snr: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Adds Gaussian white noise to images based on power SNR.
    
    SNR definition: SNR = signal_variance / noise_variance
    Therefore: noise_std = signal_std / sqrt(SNR)
    
    Args:
        image (torch.Tensor): Image tensor of shape (batch, height, width)
        snr (torch.Tensor): Signal-to-noise ratio (power SNR)
        mask (torch.Tensor): Signal mask
        
    Returns:
        Noisy image with same shape as input
    """
    # Calculate signal standard deviation within mask
    signal_std = torch.std(image[:, mask], dim=[-1])  # (batch,)
    
    # Calculate noise standard deviation from power SNR
    # SNR = σ²_signal / σ²_noise → σ_noise = σ_signal / sqrt(SNR) [batch, 1, 1]
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
    nps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Adds Poisson and readout noise to images.

    Args:
        image (torch.Tensor): Image tensor of shape (batch, height, width) 
        target_snr (torch.Tensor): Signal-to-noise target ratio (power SNR)
        simulation_param (dict): Dictionary of simulation parameters
        mtf (optional, torch.Tensor): MTF grid
        nps (optional, torch.Tensor): NPS excess noise grid

    Returns:
        Noisy image with the same shape as the input.
    """
    # 1. Setup and Parameter Extraction
    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size = simulation_param["pixel_size"].item() 
    else:
        pixel_size = simulation_param["pixel_size"]

    # 2. Get mask 
    mask = simulation_param["mask"]
    
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
    contrast_scale = torch.where(
          signal_var > 0,
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


def add_noise_from_nps(
    image: torch.Tensor,
    snr: torch.Tensor,
    nps: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Adds colored noise to images based on a Noise Power Spectrum (NPS) file.

    This function generates a noise realization with the spectral characteristics
    defined in the NPS, then scales it to achieve a target power SNR.

    SNR definition: SNR = signal_variance / noise_variance

    Args:
        image (torch.Tensor): Image tensor of shape (batch, height, width)
        snr (torch.Tensor): Signal-to-noise ratio (power SNR)
        nps (torch.Tensor): NPS from mrc file
        mask (torch.Tensor): Signal mask

    Returns:
        torch.Tensor: Noisy image with the same shape as the input.
    """
    # 1. Create a realization of colored noise
    # Start with white noise in real space for each image in the batch
    white_noise_real = torch.randn_like(image)

    # Go to Fourier space
    white_noise_fft = torch.fft.fft2(white_noise_real)

    # Color the noise by multiplying with the NPS amplitude spectrum.
    colored_noise_fft = white_noise_fft * nps.unsqueeze(0)

    # Go back to real space to get the unscaled noise realization
    unscaled_colored_noise = torch.fft.ifft2(colored_noise_fft).real

    # 2. Calculate signal and noise variances
    # Signal variance
    signal_var = torch.var(image[:, mask], dim=[-1]) # Shape: (batch,)

    # Noise variance
    noise_var = torch.var(unscaled_colored_noise, dim=[-2, -1]) # Shape: (batch,)

    # 4. Rescale noise to match the target SNR
    # We want: SNR = signal_var / var(scaled_noise)
    # Let scaled_noise = C * unscaled_colored_noise
    # Then var(scaled_noise) = C^2 * var(unscaled_colored_noise)
    # So, SNR = signal_var / (C^2 * noise_var)
    # Solving for C: C = sqrt(signal_var / (SNR * noise_var))

    # Add a small epsilon to prevent division by zero
    epsilon = 1e-9
    snr_squeezed = snr.squeeze() # From (batch, 1, 1) to (batch,)

    scaling_factor_sq = torch.where(
          signal_var > 0,
          signal_var / (snr_squeezed * noise_var + epsilon), 
          torch.ones_like(signal_var)
    )
    scaling_factor = torch.sqrt(torch.clamp(scaling_factor_sq, min=0))

    # Reshape for broadcasting over the image dimensions
    scaling_factor = scaling_factor.view(-1, 1, 1)

    # 5. Add scaled noise to the image
    scaled_noise = unscaled_colored_noise * scaling_factor
    final_image = image + scaled_noise

    return final_image
