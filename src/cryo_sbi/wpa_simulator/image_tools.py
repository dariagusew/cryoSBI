import torch
from typing import Union
import torchvision.transforms as transforms


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


def make_fft_k2_grid(
    num_pixels: Union[float, torch.Tensor], 
    pixel_size: Union[float, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Create a 2D frequency grid in 1/Angstrom

    Args:
        num_pixels: The number of pixels for the grid (e.g., 128).
        pixel_size: The pixel size in Angstrom.
        device: The torch device to place the tensor on ('cpu' or 'cuda').

    Returns:
        A torch.Tensor of shape (1, size, size) representing frequency magnitudes.
    """
    # Extract Python numbers from scalar tensors if necessary.
    if isinstance(num_pixels, torch.Tensor):
        num_pixels = int(num_pixels.item())
    if isinstance(pixel_size, torch.Tensor):
        pixel_size = pixel_size.item()

    # Create 1D frequency grid (in 1/Å) ---
    freq_pix_1d = torch.fft.fftfreq(num_pixels, d=pixel_size, device=device)

    # Create 2D frequency space
    kx, ky = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")

    # Calculate square
    k2 = kx**2 + ky**2
   
    # Add batch dimension
    k2 = k2.unsqueeze(0)  # [1, num_pixels, num_pixels] - broadcasting will handle batch

    return k2


def gaussian_normalize_image(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize an images by subtracting the mean and dividing by the standard deviation.

    Args:
        image (torch.Tensor): Image of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).

    Returns:
        normalized (torch.Tensor): Normalized image.
    """

    mean = images.mean(dim=[1, 2])
    std = images.std(dim=[1, 2])

    return transforms.functional.normalize(images, mean=mean, std=std)
