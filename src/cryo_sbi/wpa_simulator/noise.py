import torch
from typing import Optional


def circular_mask(n_pixels: int, radius: int, device: str = "cpu") -> torch.Tensor:
    """
    Creates a circular mask of radius centered in the image.
    
    Args:
        n_pixels: Number of pixels along image side
        radius: Radius of the mask in pixels
        device: Device to create mask on
        
    Returns:
        Boolean mask of shape (n_pixels, n_pixels)
    """
    grid = torch.linspace(
        -0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels, device=device
    )
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius**2
    
    return mask


def add_noise(
    image: torch.Tensor, 
    snr: torch.Tensor,
    mask_radius: Optional[int] = None,
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
        mask_radius = n_pixels // 2
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
