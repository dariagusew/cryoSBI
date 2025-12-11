import numpy as np
import torch

def apply_ctf(
    image: torch.Tensor, 
    defocus: torch.Tensor,  # in micrometers (positive = underfocus)
    b_factor: torch.Tensor,  # in Å²
    amp: torch.Tensor,  # amplitude contrast (0.07-0.1 typical)
    pixel_size: torch.Tensor,  # in Å
    voltage: float = 300.0,  # in kV
    cs: float = 0.0  # spherical aberration in mm
) -> torch.Tensor:
    """
    Applies the CTF to the image in Fourier space.
    
    Args:
        image: Input image (num_batch, num_pixels, num_pixels)
        defocus: Defocus in micrometers (positive = underfocus)
        b_factor: B-factor in Ų
        amp: Amplitude contrast ratio (typically 0.07-0.1)
        pixel_size: Pixel size in Ångströms
        voltage: Electron voltage in kV
        cs: Spherical aberration in mm
    
    Returns:
        Image with CTF applied
    """

    num_batch, num_pixels, _ = image.shape
    device = image.device
   
    # Electron wavelength (in Angstroms)
    voltage_tensor = torch.tensor(voltage, dtype=torch.float32, device=device)
    wavelength = 12.2643247 / torch.sqrt(
        voltage_tensor * 1000 + 0.978466 * voltage_tensor**2
    )
   
    # Create frequency grid (in 1/Å)
    freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size.item(), device=device)
    kx, ky = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")
    k2 = kx**2 + ky**2  # [num_pixels, num_pixels]
    k2 = k2.unsqueeze(0)  # [1, num_pixels, num_pixels] - broadcasting will handle batch
   
    # Convert units and reshape for broadcasting
    defocus_angstrom = defocus.view(-1, 1, 1) * 1e4  # [num_batch, 1, 1]
    cs_angstrom = cs * 1e7  # scalar
    amp_reshaped = amp.view(-1, 1, 1)  # [num_batch, 1, 1]
    b_factor_reshaped = b_factor.view(-1, 1, 1)  # [num_batch, 1, 1]
   
    # Phase aberration function [num_batch, num_pixels, num_pixels]
    # cs not included in cryoSBI - zero by default here
    gamma = (
        torch.pi * wavelength * defocus_angstrom * k2
        - 0.5 * torch.pi * cs_angstrom * wavelength**3 * k2**2
    )
   
    # Create imaginary component to force complex dtype
    imag = torch.zeros_like(gamma, device=device) * 1j
   
    # CTF = -√(1-A²)·sin(γ) - A·cos(γ) [num_batch, num_pixels, num_pixels]
    ctf = (
        -torch.sqrt(1 - amp_reshaped**2) * torch.sin(gamma)
        - amp_reshaped * torch.cos(gamma)
        + imag
    )
   
    # Apply envelope function [num_batch, num_pixels, num_pixels]
    # unlike cryoSBI, we divide Bfactor by 4 as it is the standard for the envelope term modulating an "amplitude"
    envelope = torch.exp(-b_factor_reshaped * k2 * 0.25)
    # Division by amp as in cryoSBI removed - images are normalized after
    ctf = ctf * envelope
   
    # Apply CTF in Fourier space
    image_fft = torch.fft.fft2(image)
    image_fft_ctf = image_fft * ctf
    image_ctf = torch.fft.ifft2(image_fft_ctf).real

    return image_ctf
