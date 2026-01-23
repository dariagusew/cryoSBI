import torch


def apply_ctf(
    image: torch.Tensor,
    defocus: torch.Tensor,  # Defocus in µm (positive = underfocus)
    b_factor: torch.Tensor,  # in Å²
    amp: torch.Tensor,  # amplitude contrast (0.07-0.1 typical)
    simulation_param: dict # simulation parameters
) -> torch.Tensor:
    """
    Applies the CTF to the image in Fourier space.
    
    Args:
        image: Input image (num_batch, num_pixels, num_pixels)
        defocus: Defocus in micrometers. Can be a tensor of shape [batch_size, 1] 
                 (for no astigmatism) or [batch_size, 3] to include astigmatism
                 in the format [defocus, defocus_astig, defocus_astig_angle].
        b_factor: B-factor
        amp: Amplitude contrast ratio (typically 0.07-0.1)
        simulation_param: Simulation parameters
    
    Returns:
        Image with CTF applied
    """

    # Get parameters
    device = simulation_param["device"]

    if isinstance(simulation_param["num_pixels"], torch.Tensor):
        num_pixels = int(simulation_param["num_pixels"].item())
    else:
        num_pixels = int(simulation_param["num_pixels"])

    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size = simulation_param["pixel_size"].item()
    else:
        pixel_size = simulation_param["pixel_size"]
   
    # Electron wavelength (in Angstroms)
    voltage_tensor = torch.tensor(simulation_param["voltage"], dtype=torch.float32, device=device)
    wavelength = 12.2643247 / torch.sqrt(
        voltage_tensor * 1000 + 0.978466 * voltage_tensor**2
    )
   
    # Get frequency grid
    k2 = simulation_param["k2"]

    # Handle different defocus input formats
    if simulation_param["astigmatism"]:
        # ASTIGMATIC CASE: Defocus map is azimuthally dependent
        defocus_angstrom = defocus[:, 0].view(-1, 1, 1) * 1e4
        defocus_astig_angstrom = defocus[:, 1].view(-1, 1, 1) * 1e4
        angle_astig_rad = torch.deg2rad(defocus[:, 2]).view(-1, 1, 1)
        
        # We need the angle (phi) for each point in the Fourier grid
        freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size, device=device)
        kx, ky = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")
        phi = torch.atan2(ky, kx).unsqueeze(0)
    
        # Defocus(phi) = Mean_Defocus + Astig_Defocus * cos(2 * (phi - angle_astig))
        defocus_map_angstrom = (
            defocus_angstrom +
            defocus_astig_angstrom * torch.cos(2 * (phi - angle_astig_rad))
        )

    elif defocus.ndim == 2 and defocus.shape[1] == 3:
        # NON-ASTIGMATIC CASE with defocus prior from star file
        defocus_map_angstrom = defocus[:, 0].view(-1, 1, 1) * 1e4
 
    else:
        # NON-ASTIGMATIC CASE with uniform defocus prior
        defocus_map_angstrom = defocus.view(-1, 1, 1) * 1e4

    # Convert units and reshape for broadcasting
    cs_angstrom = simulation_param["cs"] * 1e7  # scalar
    amp_reshaped = amp.view(-1, 1, 1)  # [num_batch, 1, 1]
    b_factor_reshaped = b_factor.view(-1, 1, 1)  # [num_batch, 1, 1]

    # Phase aberration function [num_batch, num_pixels, num_pixels]
    gamma = (
        torch.pi * wavelength * defocus_map_angstrom * k2
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
    envelope = torch.exp(-b_factor_reshaped * k2 * 0.25)
    # Division by amp as in cryoSBI removed - images are normalized after
    ctf = ctf * envelope
   
    # Apply CTF in Fourier space
    image_fft = torch.fft.fft2(image)
    image_fft_ctf = image_fft * ctf
    image_ctf = torch.fft.ifft2(image_fft_ctf).real

    return image_ctf
