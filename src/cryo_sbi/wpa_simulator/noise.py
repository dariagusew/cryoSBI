import torch
from typing import Optional
import math

def circular_mask(n_pixels: int, radius: float, device: str = "cpu") -> torch.Tensor:
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
    snr: torch.Tensor,
    simulation_param: dict,
    mask_radius: Optional[float] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Adds Poisson noise to images based on power SNR.
    
    SNR definition: SNR = signal_variance / noise_variance
    
    Args:
        image: Image tensor of shape (batch, height, width)
        snr: Signal-to-noise ratio (power SNR)
             Typical values: 0.001 - 0.1 for single particles
        simulation_param: Dictionary of simulation parameters
        mask_radius: Radius for signal calculation. If None, uses image_size//2
        seed: Random seed for reproducibility
        
    Returns:
        Noisy image with same shape as input
    """

    # Convert pixel_size to float if tensor
    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size_val = simulation_param["pixel_size"].item() 
    else:
        pixel_size_val = simulation_param["pixel_size"]

    if seed is not None:
        torch.manual_seed(seed)
    
    device = image.device
    n_pixels = image.shape[-1]

    # 0. Create mask for signal region
    if mask_radius is None:
        mask_radius = 0.5 * n_pixels
    mask = circular_mask(n_pixels, mask_radius, device=device)

    # 1. Convert experimental dose (e/Å²) to simulation dose (e/pixel)
    mean_electron_dose = simulation_param["dose"] * pixel_size_val**2

    # 2. Calculate signal variance within mask
    signal_var = torch.var(image[:, mask], dim=[-1]).view(-1, 1, 1) # [B, 1, 1]

    # 3. Determine the contrast scale for each image based on its target SNR
    # Empirical MTF boost factor - decrease target SNR - MTF will lowpass filter the image
    mtf_boost_factor = simulation_param["mtf_a"]**0.9 / simulation_param["qe"]
    # Calculate contrast 
    contrast_scale = torch.sqrt(mtf_boost_factor * snr / mean_electron_dose / signal_var) # [B, 1, 1]

    # 4. Create the mean counts per pixel map (Weak Phase Object Approximation)
    mean_counts_per_pixel = mean_electron_dose * (1.0 + contrast_scale * image)

    # 5. Apply Imperfect QE
    # We thin the incoming electron beam before the shot noise occurs
    mean_counts_per_pixel = simulation_param["qe"] * mean_counts_per_pixel

    # 6. Generate the noisy image by sampling from a Poisson distribution.
    # first check all positive
    mean_counts_per_pixel = torch.clamp(mean_counts_per_pixel, min=0)
    image_noise = torch.poisson(mean_counts_per_pixel)

    # 7. Apply detector effects
    image_noise = apply_detector_effects(image_noise, simulation_param["mtf_a"], simulation_param["readout_std"], pixel_size_val)

    return image_noise


def apply_detector_effects(
    image_counts: torch.Tensor,
    mtf_a: float,
    readout_std_dev: float,
    pixel_size: float
) -> torch.Tensor:
    """
    Applies MTF blurring and readout noise to a detected electron image.

    Args:
        image_counts: The image after Poisson sampling (shot noise), shape [..., H, W].
        mtf_a: MTF parameter (dimensionless, mtf at Nyquist). 
               Typical values: 0.15-0.25 for modern detectors.
        readout_std_dev: Standard deviation of Gaussian readout noise (in electrons).
                         Typical values: 1-2 e⁻ for counting mode.
        pixel_size: Pixel size at specimen level (in Å).
    
    Returns:
        Final detector image with MTF blur and readout noise applied.
    
    Physics:
        1. MTF blur: exp(-k² / (2σ²)) where σ = mtf_a × k_Nyquist
        2. Readout noise: Gaussian with std = readout_std_dev
    """

    device = image_counts.device
    num_pixels = image_counts.shape[-1]

    # 1. Create frequency grid (in 1/Å) ---
    freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size, device=device)
    kx, ky = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")
    k2 = kx**2 + ky**2

    # 2. Apply MTF (Modulation Transfer Function)
    # Compute Nyquist frequency in Å⁻¹
    k_nyquist = 1.0 / (2.0 * pixel_size)
    # Solve: exp(-k_nyquist² / (2σ²)) = mtf_a
    # σ² = -k_nyquist² / (2 × ln(target))
    mtf_a_physical = k_nyquist / math.sqrt(-2 * math.log(mtf_a))

    # 3. Gaussian MTF model: exp(-k²/(2σ²))
    mtf = torch.exp(-k2 / (2 * mtf_a_physical**2))

    # 4. Apply MTF in Fourier space
    image_fft = torch.fft.fft2(image_counts.float(), norm='ortho')
    image_fft_mtf = image_fft * mtf
    image_blurred = torch.fft.ifft2(image_fft_mtf, norm='ortho').real

    # 5. Add Readout Noise
    readout_noise = torch.randn_like(image_blurred) * readout_std_dev
    final_image = image_blurred + readout_noise

    return final_image
