import numpy as np
import torch

def gen_rot_matrix(quats: torch.Tensor) -> torch.Tensor:
    """
    Generate rotation matrices from quaternions

    Quaternion convention: [qw, qx, qy, qz] where qw is the real/scalar part
    and (qx, qy, qz) is the imaginary/vector part.

    Args:
        quats (torch.Tensor): Quaternions of shape (n_batch, 4)
                             Convention: [qw, qx, qy, qz]

    Returns:
        rot_matrix (torch.Tensor): Rotation matrices of shape (n_batch, 3, 3)
    """
    # Extract quaternion components
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    # Precompute squared terms (each computed once)
    qx2 = qx ** 2
    qy2 = qy ** 2
    qz2 = qz ** 2
    
    # Precompute products
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz
    
    # Build rotation matrix using stack (single operation)
    rot_matrix = torch.stack([
        torch.stack([1 - 2 * (qy2 + qz2), 2 * (qxqy - qwqz), 2 * (qxqz + qwqy)], dim=1),
        torch.stack([2 * (qxqy + qwqz), 1 - 2 * (qx2 + qz2), 2 * (qyqz - qwqx)], dim=1),
        torch.stack([2 * (qxqz - qwqy), 2 * (qyqz + qwqx), 1 - 2 * (qx2 + qy2)], dim=1)
    ], dim=1)
    
    return rot_matrix


def project_density(
    coords: torch.Tensor,
    quats: torch.Tensor,
    sigma: torch.Tensor,
    shift: torch.Tensor,
    num_pixels: torch.Tensor,
    pixel_size: torch.Tensor,
) -> torch.Tensor:
    """
    Generate 2D projections from a set of 3D coordinates

    Projects 3D atomic coordinates onto a 2D plane after rotation, where each atom
    is represented as a Gaussian with standard deviation sigma. The projection is
    computed on a regular grid.

    Args:
        coords (torch.Tensor): Coordinates of shape (num_batch, 3, num_atoms)
        quats (torch.Tensor): Quaternions of shape (num_batch, 4) defining rotations
        sigma (torch.Tensor): Parameters of Gaussian kernel
        shift (torch.Tensor): 2D shift to apply of shape (num_batch, 2)
        num_pixels (torch.Tensor): Number of pixels along one image dimension
        pixel_size (torch.Tensor): Pixel size in Angstrom

    Returns:
        image (torch.Tensor): Projected images of shape (num_batch, num_pixels, num_pixels)
    """
    num_batch, _, num_atoms = coords.shape
    
    # Convert num_pixels to int
    num_pixels = int(num_pixels.item())
   
    # Create grid using linspace
    grid_min = -pixel_size * num_pixels * 0.5
    grid_max = pixel_size * num_pixels * 0.5
    grid = torch.linspace(grid_min, grid_max - pixel_size, num_pixels, 
                         device=coords.device, dtype=coords.dtype)
    
    # Generate rotation matrices
    rot_matrix = gen_rot_matrix(quats)
    
    # Apply rotation to coordinates
    coords_rot = torch.bmm(rot_matrix, coords)
    
    # Apply shift to x and y coordinates
    coords_rot[:, :2, :] = coords_rot[:, :2, :] + shift.unsqueeze(-1)

    # Compute Gaussian in x direction
    # Shape: (num_batch, num_pixels, num_atoms)
    dx = grid.unsqueeze(0).unsqueeze(-1) - coords_rot[:, 0, :].unsqueeze(1)
    gauss_x = sigma[0, :].view(1, 1, -1) * torch.exp(sigma[1, :].view(1, 1, -1) * dx ** 2)
   
    # Compute Gaussian in y direction
    # Shape: (num_batch, num_atoms, num_pixels)
    dy = grid.unsqueeze(0).unsqueeze(0) - coords_rot[:, 1, :].unsqueeze(-1)
    gauss_y = sigma[0, :].view(1, -1, 1) * torch.exp(sigma[1, :].view(1, -1, 1) * dy ** 2)
   
    # Matrix multiplication to get 2D projection
    image = torch.bmm(gauss_x, gauss_y)

    return image
