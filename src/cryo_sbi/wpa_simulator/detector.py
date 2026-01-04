import torch
from typing import Tuple
from cryo_sbi.wpa_simulator.image_tools import make_fft_k2_grid
import math


def get_mtf_nps_grids(
    simulation_param: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-calculate MTF/NPS grids using Gaussian fits of MTF and DQE.

    Args:
        simulation_param: Dictionary of simulation parameters 
    
    Returns:
        MTF/NPS grids and scaling factor for target SNR calculation.
    
    """
    # 0. Extract Python numbers from scalar tensors if necessary
    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size = simulation_param["pixel_size"].item()
    else:
        pixel_size = simulation_param["pixel_size"]

    if isinstance(simulation_param["num_pixels"], torch.Tensor):
        num_pixels = int(simulation_param["num_pixels"].item())
    else:
        num_pixels = int(simulation_param["num_pixels"])

    # 1. Create k2 grid for the radially symmetric Gaussian function
    k2 = make_fft_k2_grid(num_pixels, pixel_size, device=simulation_param["device"])
    # Compute Nyquist frequency in Å⁻¹
    k_nyquist = 0.5 / pixel_size
    
    # 2. Calculate sigma for MTF Gaussian fit
    sigma_k_sq = -0.5 * k_nyquist**2 / math.log(simulation_param["mtf_n"])
    # Gaussian MTF model: exp(-k²/(2σ²))
    mtf = torch.exp(-0.5 * k2 / sigma_k_sq)

    # 3. Now we calculate the DQE grid
    qe_zero = simulation_param["qe"] # DQE(0)
    qe_nyq = simulation_param["qe_n"] # DQE(NYQ)
    sigma_k_sq = -0.5 * k_nyquist**2 / math.log(qe_nyq/qe_zero)
    # Gaussian DQE model: qe_zero * exp(-k²/(2σ²))
    dqe = qe_zero * torch.exp(-0.5 * k2 / sigma_k_sq)
    # Clamp DQE to a small positive number to avoid division by zero or overflow.
    dqe_safe = torch.clamp(dqe, min=1e-8)

    # 4. Finally we calculate the excess noise power to add
    Deff = simulation_param["dose"] * pixel_size**2
    nps = Deff * mtf * mtf * ( 1.0 / dqe_safe - 1.0 )

    # 5. Calculate SNR scaling factor
    scaling_factor = torch.mean(mtf**2 / dqe_safe)

    return mtf, nps, scaling_factor
