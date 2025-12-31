import torch
import zuko
import numpy as np
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.utils.data import DataLoader, Dataset, IterableDataset
from scipy.spatial.transform import Rotation as R

class LogTransform(zuko.distributions.Transform):
    r"""
    Transform via the mapping :math:`y = \log(x)`.
    """
    domain = constraints.positive
    codomain = constraints.real
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, LogTransform)

    def _call(self, x):
        return x.log()

    def _inverse(self, y):
        return y.exp()

    def log_abs_det_jacobian(self, x, y):
        # d(log(x))/dx = 1/x
        # log|1/x| = -log(x)
        return -x.log()

def compute_covariance_matrix(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute covariance matrix for a point cloud to find principal axes.
    
    Args:
        coords: [3, Natoms] torch tensor
        
    Returns:
        eigenvalues: Sorted eigenvalues
        eigenvectors: Corresponding eigenvectors
    """
    coords_centered = coords - coords.mean(dim=1, keepdim=True)
    cov = (coords_centered @ coords_centered.T) / coords_centered.shape[1]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    return eigenvalues, eigenvectors


class PreferredOrientationPrior:
    """
    Generate preferred orientations based on molecular shape.
    The direction with smallest variance (thinnest dimension) should be 
    perpendicular to the grid, corresponding to the largest moment of inertia.
    """
    def __init__(self, models: list[torch.Tensor], wobble_angle: float = 15.0, device: str = 'cpu'):
        """
        Args:
            models: List of torch tensors, each of shape [3, Natoms]
            wobble_angle: Degrees of deviation from preferred orientation
            device: torch device
        """
        self.device = device
        self.wobble_angle = wobble_angle

        # Precompute base quaternions for each model (stored as numpy array)
        base_quats_list = []
        for model in models:
            model_cpu = model.cpu() if model.device.type != 'cpu' else model
            eigenvalues, eigenvectors = compute_covariance_matrix(model_cpu)
            # Smallest variance direction should align with z
            preferred_z = eigenvectors[:, 0].numpy()
            
            # Compute base rotation and immediately convert to quaternion
            base_rotation = self._align_vector_to_z(preferred_z)
            base_quats_list.append(base_rotation.as_quat())  # [x, y, z, w]
        
        # Store as numpy array [num_models, 4] for vectorized indexing
        self.base_quaternions = np.stack(base_quats_list)

    def sample(self, shape: tuple, model_indices: torch.Tensor) -> torch.Tensor:
        """
        Sample preferred orientation quaternions.

        Args:
            shape: tuple, batch shape
            model_indices: torch tensor of model indices [batch_size]

        Returns:
            torch tensor of quaternions [batch_size, 4] in [w, x, y, z] format
        """
        batch_size = shape[0]
        model_indices_np = model_indices.cpu().numpy().astype(int).flatten()
        
        # Vectorized random generation
        wobble_angles = np.random.randn(batch_size, 3) * np.radians(self.wobble_angle)
        z_angles = np.random.uniform(0, 2 * np.pi, size=batch_size)
        
        # Create rotation vectors for z-rotation
        z_rotvecs = np.zeros((batch_size, 3))
        z_rotvecs[:, 2] = z_angles
        
        # Batch create wobbles and z_rotations
        wobbles = R.from_euler('xyz', wobble_angles)
        z_rotations = R.from_rotvec(z_rotvecs)
        
        # Vectorized indexing - get base quaternions for selected models
        base_quats = self.base_quaternions[model_indices_np]  # [batch_size, 4]
        
        # Single conversion to batch Rotation object
        base_rots = R.from_quat(base_quats)
        
        # Batch composition: z_rotation * wobble * base_rotation
        final_rotations = z_rotations * wobbles * base_rots
        
        # Batch quaternion extraction
        quats = final_rotations.as_quat()  # [batch_size, 4] in [x, y, z, w]
        
        # Reorder to [w, x, y, z]
        quats_reordered = np.column_stack([quats[:, 3], quats[:, :3]])
        
        # Single device transfer
        return torch.from_numpy(quats_reordered.astype(np.float32)).to(self.device)

    def _align_vector_to_z(self, vector: np.ndarray) -> R:
        """
        Create rotation that aligns vector with z-axis [0, 0, 1].
        
        Args:
            vector: 3D vector to align with z-axis
            
        Returns:
            scipy Rotation object
        """
        z_axis = np.array([0, 0, 1])
        v = np.cross(vector, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(vector, z_axis)

        if s < 1e-6:  # Already aligned
            return R.identity() if c > 0 else R.from_euler('x', 180, degrees=True)

        # Rodrigues' rotation formula
        vx = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        rot_matrix = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
        return R.from_matrix(rot_matrix)


class QuaternionPrior:
    """Random quaternion prior."""
    def __init__(self, device: str):
        self.device = device

    def sample(self, shape: tuple) -> torch.Tensor:
        """
        Sample random orientation quaternions.

        Args:
            shape: tuple, batch shape

        Returns:
            torch tensor of quaternions [batch_size, 4] in [w, x, y, z] format
        """
        # this is batch calculated and uniform
        quats = torch.randn(shape[0], 4, device=self.device)
        quats /= torch.norm(quats, dim=-1, keepdim=True)

        return quats


class QuaternionTestPrior:
    """Fixed quaternion for testing."""
    def __init__(self, quat: list[float], device: str):
        self.device = device
        self.quat = torch.tensor(quat, device=device)

    def sample(self, shape: tuple) -> torch.Tensor:
        quats = torch.stack([self.quat for _ in range(shape[0])], dim=0)
        return quats


class ImagePrior:
    def __init__(
        self,
        index_prior,
        quaternion_prior,
        shift_prior,
        defocus_prior,
        defocus_astig_prior,
        defocus_astig_angle_prior,
        b_factor_prior,
        amp_prior,
        snr_prior,
        device: str,
    ):
        self.priors = [
            index_prior,
            quaternion_prior,
            shift_prior,
            defocus_prior,
            defocus_astig_prior,
            defocus_astig_angle_prior,
            b_factor_prior,
            amp_prior,
            snr_prior,
        ]

    def sample(self, shape: tuple) -> list:
        # Sample index first
        indices = self.priors[0].sample(shape)
        
        # If quaternion prior supports model-specific sampling, pass indices
        if isinstance(self.priors[1], PreferredOrientationPrior):
            quaternions = self.priors[1].sample(shape, model_indices=indices)
        else:
            quaternions = self.priors[1].sample(shape)
        
        # Sample other parameters
        other_samples = [prior.sample(shape) for prior in self.priors[2:]]
        
        return [indices, quaternions] + other_samples


def get_image_priors(
    max_index: int, 
    image_config: dict, 
    models: list[torch.Tensor] = None,
    device: str = "cpu"
) -> ImagePrior:
    """
    Return priors for image generation.
    
    Args:
        max_index: Maximum model index (typically len(models) - 1)
        image_config: Configuration dictionary
        models: List of 3D models [3, Natoms] for computing preferred orientations (optional)
        device: torch device (typically "cpu" for prior generation)
    
    Returns:
        ImagePrior: Combined prior object
    """
    # Shift prior
    shift = image_config["SHIFT"]
    lower=torch.tensor([-shift, -shift], dtype=torch.float32, device=device) 
    upper=torch.tensor([+shift, +shift], dtype=torch.float32, device=device)
    # get prior type
    shift_gauss = image_config.get("SHIFT_GAUSS", False)
    # Gaussian
    if isinstance(shift_gauss, (float, int)):
       loc = torch.tensor([0, 0], dtype=torch.float32, device=device) 
       scale = torch.tensor([shift_gauss, shift_gauss], dtype=torch.float32, device=device)
       shift_prior = zuko.distributions.Truncated(zuko.distributions.Normal(loc, scale), lower=lower, upper=upper)
    # Uniform
    else:
       shift_prior = zuko.distributions.BoxUniform(lower, upper, ndims=1)

    # Defocus prior
    # 1. Average defocus: this corresponds to 0.5 * (DefocusU + DefocusV)
    if isinstance(image_config["DEFOCUS"], list) and len(image_config["DEFOCUS"]) == 2:
        defocus = image_config["DEFOCUS"]
        lower = torch.tensor([[ defocus[0] ]], dtype=torch.float32, device=device)
        upper = torch.tensor([[ defocus[1] ]], dtype=torch.float32, device=device) 
        if lower <= 0.0:
            raise ValueError("DEFOCUS lower bound must be positive")
        if lower > upper:
            raise ValueError(f"DEFOCUS lower bound ({lower.item()}) must be ≤ upper bound ({upper.item()})")
        # check prior type
        defocus_gauss = image_config.get("DEFOCUS_GAUSS", False)
        if isinstance(defocus_gauss, list) and len(defocus_gauss) == 2: 
           # Truncated Gaussian 
           loc = image_config["DEFOCUS_GAUSS"][0] 
           scale = image_config["DEFOCUS_GAUSS"][1]
           defocus_prior = zuko.distributions.Truncated(zuko.distributions.Normal(loc, scale), lower=lower, upper=upper)
        else:
           defocus_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    # 2. Delta defocus: this corresponds to 0.5 * (DefocusU - DefocusV)
    defocus_astig = image_config.get("DEFOCUS_ASTIG", [0, 0])
    lower = torch.tensor([[ defocus_astig[0] ]], dtype=torch.float32, device=device)
    upper = torch.tensor([[ defocus_astig[1] ]], dtype=torch.float32, device=device)
    if lower < 0.0:
        raise ValueError("DEFOCUS_ASTIG lower bound must be positive")
    if lower > upper:
        raise ValueError(f"DEFOCUS_ASTIG lower bound ({lower.item()}) must be ≤ upper bound ({upper.item()})")
    defocus_astig_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    # 3. Astigmatism angle in degrees
    defocus_astig_angle = image_config.get("DEFOCUS_ASTIG_ANGLE", [0, 0])
    lower = torch.tensor([[ defocus_astig_angle[0] ]], dtype=torch.float32, device=device)
    upper = torch.tensor([[ defocus_astig_angle[1] ]], dtype=torch.float32, device=device)
    if lower < 0.0:
        raise ValueError("DEFOCUS_ASTIG_ANGLE lower bound must be positive")
    if lower > upper:
        raise ValueError(f"DEFOCUS_ASTIG_ANGLE lower bound ({lower.item()}) must be ≤ upper bound ({upper.item()})")
    defocus_astig_angle_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    # B-factor prior
    if isinstance(image_config["B_FACTOR"], list) and len(image_config["B_FACTOR"]) == 2:
        b_factor = image_config["B_FACTOR"]
        lower = torch.tensor([[ b_factor[0] ]], dtype=torch.float32, device=device)
        upper = torch.tensor([[ b_factor[1] ]], dtype=torch.float32, device=device)
        if lower < 0.0:
            raise ValueError("B_FACTOR lower bound must be positive")
        if lower > upper:
            raise ValueError(f"B_FACTOR lower bound ({lower.item()}) must be ≤ upper bound ({upper.item()})")
        # check if you want Jeffreys prior, otherwise back to old uniform
        if image_config.get("USE_JEFFREYS_BFACT", False):
           b_factor_prior = zuko.distributions.TransformedUniform(LogTransform(), lower, upper)
        else:
           b_factor_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    # SNR prior
    if isinstance(image_config["SNR"], list) and len(image_config["SNR"]) == 2:
        snr = image_config["SNR"]
        lower = torch.tensor([[ snr[0] ]], dtype=torch.float32, device=device)
        upper = torch.tensor([[ snr[1] ]], dtype=torch.float32, device=device)
        if lower > upper:
            raise ValueError(f"SNR lower bound must be ≤ upper bound")
        # check if you want uniform SNR, otherwise back to old log-uniform/Jeffreys
        if image_config.get("USE_UNIFORM_SNR", False):
           snr_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)
        else:
           snr_prior = zuko.distributions.TransformedUniform(LogTransform(), lower, upper)

    # Amplitude prior
    amp_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor([[ image_config["AMP"] ]], dtype=torch.float32, device=device),
        upper=torch.tensor([[ image_config["AMP"] ]], dtype=torch.float32, device=device),
        ndims=1,
    )

    # Index prior
    index_prior = zuko.distributions.BoxUniform(
        lower=torch.tensor([0], dtype=torch.float32, device=device),
        upper=torch.tensor([max_index], dtype=torch.float32, device=device),
    )
    
    # Quaternion prior
    # Check for preferred orientations
    if models is not None and image_config.get("USE_PREFERRED_ORIENTATIONS", False):
        wobble_angle = image_config.get("WOBBLE_ANGLE", 15.0)  # Default 15 degrees
        quaternion_prior = PreferredOrientationPrior(models, wobble_angle, device)
    # Check for fixed test quaternion
    elif (
        image_config.get("ROTATIONS")
        and isinstance(image_config["ROTATIONS"], list)
        and len(image_config["ROTATIONS"]) == 4
    ):
        test_quat = image_config["ROTATIONS"]
        quaternion_prior = QuaternionTestPrior(test_quat, device)
    # Default to random quaternions
    else:
        quaternion_prior = QuaternionPrior(device)

    return ImagePrior(
        index_prior,
        quaternion_prior,
        shift_prior,
        defocus_prior,
        defocus_astig_prior,
        defocus_astig_angle_prior,
        b_factor_prior,
        amp_prior,
        snr_prior,
        device=device,
    )


class PriorDataset(IterableDataset):
    def __init__(
        self,
        prior: Distribution,
        batch_shape: torch.Size = (),
    ):
        super().__init__()
        self.prior = prior
        self.batch_shape = batch_shape

    def __iter__(self):
        while True:
            theta = self.prior.sample(self.batch_shape)
            yield theta


class PriorLoader(DataLoader):
    def __init__(
        self,
        prior: Distribution,
        batch_size: int = 2**8,  # 256
        **kwargs,
    ):
        super().__init__(
            PriorDataset(prior, batch_shape=(batch_size,)),
            batch_size=None,
            **kwargs,
        )
