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
    models: torch.Tensor,
    index: torch.Tensor,
    quats: torch.Tensor,
    sigma: torch.Tensor,
    shift: torch.Tensor,
    num_pixels: torch.Tensor,
    pixel_size: torch.Tensor,
    atom_batch_size: int = 1024
) -> torch.Tensor:
    """
    Generate 2D projections from a set of 3D coordinates using atom-batching
    to reduce memory usage.

    Projects 3D atomic coordinates onto a 2D plane after rotation, where each atom
    is represented as a Gaussian. The projection is computed on a regular grid.

    Args:
        models (torch.Tensor): Models
        index (torch.Tensor): Index of selected models
        quats (torch.Tensor): Quaternions of shape (num_batch, 4) defining rotations
        sigma (torch.Tensor): Parameters of Gaussian kernel of shape (n_models, 2, num_atoms)
        shift (torch.Tensor): 2D shift to apply of shape (num_batch, 2)
        num_pixels (torch.Tensor): Number of pixels along one image dimension
        pixel_size (torch.Tensor): Pixel size in Angstrom
        atom_batch_size (int): The number of atoms to process in each chunk to
                               save memory. Defaults to 2048.

    Returns:
        image (torch.Tensor): Projected images of shape (num_batch, num_pixels, num_pixels)
    """
    # Convert index to integer 
    index = index.round().long()
    # Get coordinates and sigmas of selected models
    coords = models[index.flatten()]
    sigmas = sigma[index.flatten()]

    num_batch, _, num_atoms = coords.shape
    device, dtype = coords.device, coords.dtype

    # Convert num_pixels to int
    num_pixels = int(num_pixels.item())

    # Create grid using linspace
    grid_min = -pixel_size * num_pixels * 0.5
    grid_max =  pixel_size * num_pixels * 0.5
    grid = torch.linspace(grid_min, grid_max - pixel_size, num_pixels,
                         device=device, dtype=dtype)

    # Generate rotation matrices
    rot_matrix = gen_rot_matrix(quats)
    
    # Apply rotation to all coordinates
    coords_rot = torch.bmm(rot_matrix, coords)
    
    # Apply shift to all x and y coordinates
    coords_rot[:, :2, :] = coords_rot[:, :2, :] + shift.unsqueeze(-1)

    # Initialize the final image tensor with zeros
    final_image = torch.zeros((num_batch, num_pixels, num_pixels), device=device, dtype=dtype)

    # Loop over atoms in batches
    for i in range(0, num_atoms, atom_batch_size):
        # Define the slice for the current atom batch
        start_idx = i
        end_idx = min(i + atom_batch_size, num_atoms)
        
        # Slice the rotated coordinates and sigma for the current batch
        coords_rot_batch = coords_rot[:, :, start_idx:end_idx]
        sigma_batch = sigmas[:, :, start_idx:end_idx]

        # Compute Gaussian in x direction for the batch
        # Shape: (batch_size, atom_batch_size, num_pixels)
        dx_batch = grid.unsqueeze(0).unsqueeze(-1) - coords_rot_batch[:, 0, :].unsqueeze(1)
        amplitude_x = sigma_batch[:, 0, :].unsqueeze(1)  # Shape: (batch_size, 1, atom_batch_size)
        width_x = sigma_batch[:, 1, :].unsqueeze(1)      # Shape: (batch_size, 1, atom_batch_size)
        gauss_x_batch = amplitude_x * torch.exp(width_x * dx_batch ** 2)

        # Compute Gaussian in y direction for the batch
        # Shape: (batch_size, atom_batch_size, num_pixels)
        dy_batch = grid.unsqueeze(0).unsqueeze(0) - coords_rot_batch[:, 1, :].unsqueeze(-1)
        amplitude_y = sigma_batch[:, 0, :].unsqueeze(-1)  # Shape: (batch_size, atom_batch_size, 1)
        width_y = sigma_batch[:, 1, :].unsqueeze(-1)      # Shape: (batch_size, atom_batch_size, 1)
        gauss_y_batch = amplitude_y * torch.exp(width_y * dy_batch ** 2)

        # Matrix multiplication to get 2D projection for this batch
        image_batch = torch.bmm(gauss_x_batch, gauss_y_batch)
        
        # Accumulate the result, set nan to zero for missing atoms in that model
        final_image += torch.nan_to_num(image_batch, nan=0.0)

    return final_image
