import torch
import zuko
import starfile
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


class DefocusPrior:
    """
    Samples defocus triplets (mean defocus, astigmatism, angle)
    from an empirical distribution defined by a RELION STAR file.
    """
    def __init__(self, star_file_path: str, device: str = 'cpu'):
        """
        Args:
            star_file_path: Path to the RELION STAR file (e.g., from CTFFIND).
            device: The torch device to store the parameters on.
        """
        self.device = device
        self.param_triplets = self._load_and_process_star_file(star_file_path)

    def _load_and_process_star_file(self, path: str) -> torch.Tensor:
        """Reads, parses, and converts STAR file data."""
        print(f"Loading defocus parameters from: {path}")
        try:
            data = starfile.read(path)
            df = data['particles'] if 'particles' in data else data 
        except Exception as e:
            raise IOError(f"Failed to read or parse STAR file '{path}'. Error: {e}")

        # Check for required columns
        required_cols = ['rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"STAR file must contain the columns: {required_cols}")

        # Convert from Ångströms (RELION standard) to micrometers
        defocus_u_um = df['rlnDefocusU'].values / 10000.0
        defocus_v_um = df['rlnDefocusV'].values / 10000.0
        angle_deg = df['rlnDefocusAngle'].values

        # Convert to the parameterization used by apply_ctf
        # defocus = 0.5 * (DefocusU + DefocusV)
        # defocus_astig = 0.5 * |DefocusU - DefocusV|
        # defocus_astig_angle = angle
        mean_defocus = 0.5 * (defocus_u_um + defocus_v_um)
        astig_magnitude = 0.5 * abs(defocus_u_um - defocus_v_um)

        # Stack into a single tensor for efficient sampling
        triplets = torch.tensor(
            np.stack([mean_defocus, astig_magnitude, angle_deg], axis=1),
            dtype=torch.float32,
            device=self.device
        )
        print(f"Successfully loaded {len(triplets)} defocus triplets.")
        return triplets

    def sample(self, shape: tuple) -> torch.Tensor:
        """
        Samples a batch of defocus triplets.

        Args:
            shape: tuple, batch shape, e.g., (batch_size,).

        Returns:
            A single tensor of shape [batch_size, 3] where the columns are:
            - 0: defocus (mean)
            - 1: defocus_astig (magnitude)
            - 2: defocus_astig_angle
        """
        batch_size = shape[0]
        num_total_triplets = self.param_triplets.shape[0]

        # Randomly select indices with replacement
        indices = torch.randint(0, num_total_triplets, (batch_size,), device=self.device)

        # Gather the sampled triplets and return directly
        return self.param_triplets[indices]  # Shape: [batch_size, 3]


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
        model_indices_np = np.round(model_indices.cpu().numpy()).astype(int).flatten()
 
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
    shift_gauss = image_config.get("SHIFT_GAUSS", None)

    # Truncated Gaussian prior
    if isinstance(shift_gauss, (float, int)) and shift_gauss>0:
       loc   = torch.tensor([0, 0], dtype=torch.float32, device=device) 
       scale = torch.tensor([shift_gauss, shift_gauss], dtype=torch.float32, device=device)
       shift_prior = zuko.distributions.Truncated(zuko.distributions.Normal(loc, scale), lower=lower, upper=upper)

    # Uniform prior
    else:
       shift_prior = zuko.distributions.BoxUniform(lower, upper, ndims=1)


    # Defocus prior
    defocus = image_config["DEFOCUS"]
    if isinstance(defocus, str):
        # Astigmatism prior
        defocus_prior = DefocusPrior(defocus, device=device)

    # Uniform / Truncated Gaussian prior 
    elif isinstance(defocus, list) and len(defocus) == 2:
        lower = torch.tensor([[ defocus[0] ]], dtype=torch.float32, device=device)
        upper = torch.tensor([[ defocus[1] ]], dtype=torch.float32, device=device) 
        if lower <= 0.0:
            raise ValueError("DEFOCUS lower bound must be positive")
        if lower > upper:
            raise ValueError(f"DEFOCUS lower bound ({lower.item()}) must be ≤ upper bound ({upper.item()})")

        # check prior type
        defocus_gauss = image_config.get("DEFOCUS_GAUSS", None)

        if isinstance(defocus_gauss, list) and len(defocus_gauss) == 2: 
           # Truncated Gaussian prior 
           loc   = defocus_gauss[0]
           scale = defocus_gauss[1] 
           defocus_prior = zuko.distributions.Truncated(zuko.distributions.Normal(loc, scale), lower=lower, upper=upper)
           # Uniform prior
        else:
           defocus_prior = zuko.distributions.BoxUniform(lower=lower, upper=upper, ndims=1)

    # B-factor prior
    b_factor = image_config["B_FACTOR"]
    if isinstance(b_factor, list) and len(b_factor) == 2:
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
    snr = image_config["SNR"]
    if isinstance(snr, list) and len(snr) == 2:
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
        lower=torch.tensor([-0.4999], dtype=torch.float32, device=device),
        upper=torch.tensor([max_index+0.4999], dtype=torch.float32, device=device),
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
