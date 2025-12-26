import torch
from typing import Optional
import math

def apply_mtf(
    image: torch.Tensor,
    mtf_a: float,
    pixel_size: torch.Tensor  # in Å 
) -> torch.Tensor:
    """
    Applies MTF blurring to the image in Fourier space 

    Args:
        image: The image after CTF correction, shape [..., H, W].
        mtf_a: MTF parameter (dimensionless, mtf at Nyquist). 
               Typical values: 0.15-0.25 for modern detectors.
        pixel_size: Pixel size at specimen level (in Å).
    
    Returns:
        Image with MTF blur applied.
    
    """

    device = image.device
    num_pixels = image.shape[-1]

    # 1. Create frequency grid (in 1/Å) ---
    freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size.item(), device=device)
    kx, ky = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")
    k2 = kx**2 + ky**2
    k2 = k2.unsqueeze(0)  # [1, num_pixels, num_pixels] - broadcasting will handle batch 

    # 2. Apply MTF (Modulation Transfer Function)
    # Compute Nyquist frequency in Å⁻¹
    k_nyquist = 1.0 / (2.0 * pixel_size.item())
    # Solve: exp(-k_nyquist² / (2σ²)) = mtf_a
    # σ² = -k_nyquist² / (2 × ln(target))
    mtf_a_physical = k_nyquist / math.sqrt(-2 * math.log(mtf_a))

    # 3. Gaussian MTF model: exp(-k²/(2σ²))
    mtf = torch.exp(-0.5 * k2 / mtf_a_physical**2)

    # 4. Apply MTF in Fourier space
    image_fft = torch.fft.fft2(image)
    image_fft_mtf = image_fft * mtf
    image_blurred = torch.fft.ifft2(image_fft_mtf).real

    return image_blurred 
