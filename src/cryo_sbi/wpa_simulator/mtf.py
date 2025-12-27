import torch
from typing import Union
from cryo_sbi.wpa_simulator.image_tools import make_fft_k2_grid
import math

def apply_mtf(
    image: torch.Tensor,
    mtf_a: float,
    pixel_size: Union[float, torch.Tensor]
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

    if isinstance(pixel_size, torch.Tensor):
        pixel_size = pixel_size.item()

    # 1. Create k2 frequency grid
    k2 = make_fft_k2_grid(num_pixels, pixel_size, device) 

    # 2. Apply MTF (Modulation Transfer Function)
    # Compute Nyquist frequency in Å⁻¹
    k_nyquist = 0.5 / pixel_size
    # Solve: exp(-k_nyquist² / (2σ²)) = mtf_a
    # σ² = -k_nyquist² / (2 × ln(mtf_a))
    sigma_k_sq = - 0.5 * k_nyquist**2 / math.log(mtf_a)

    # 3. Gaussian MTF model: exp(-k²/(2σ²))
    mtf = torch.exp(-0.5 * k2 / sigma_k_sq)

    # 4. Apply MTF in Fourier space
    image_fft = torch.fft.fft2(image)
    image_fft_mtf = image_fft * mtf
    image_blurred = torch.fft.ifft2(image_fft_mtf).real

    return image_blurred 
