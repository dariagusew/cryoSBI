from typing import Union, Callable
import json
import numpy as np
import torch

from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.mtf import apply_mtf
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_Gaussian_noise, add_Poisson_noise
from cryo_sbi.wpa_simulator.image_tools import gaussian_normalize_image
from cryo_sbi.inference.priors import get_image_priors
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params


# initialize all tensors/parameters for image simulation
def create_simulation_param(image_config: dict, models: torch.Tensor, device: str = "cuda"):
    # initialize dictionary
    simulation_param = {}
    
    # number of models
    natoms = models.shape[2]
 
    # sigma param
    if "TOPOLOGY" in image_config:
        # Load TOPOLOGY from file path
        topology_path = image_config["TOPOLOGY"]
        simulation_param["sigma"] = torch.load(topology_path, map_location=device)

    elif "SIGMA" in image_config:
        sigma_value = image_config["SIGMA"]
       
        # Extract scalar value (or first element if list)
        if isinstance(sigma_value, (list, tuple)):
            sigma_val = torch.as_tensor(sigma_value[0], device=device)
        else:
            sigma_val = torch.as_tensor(sigma_value, device=device)

        # Create sigma tensor [2, natoms] on device
        simulation_param["sigma"] = torch.zeros(2, natoms, device=device)
        simulation_param["sigma"][0, :] = 1.0 / torch.sqrt(natoms * 2 * torch.pi * sigma_val**2)
        simulation_param["sigma"][1, :] = -0.5 / (sigma_val ** 2)
    else:
        raise ValueError("Either TOPOLOGY or SIGMA must be specified in image_config")

    simulation_param["num_pixels"] = torch.tensor(
        image_config["N_PIXELS"], dtype=torch.float32, device=device
    )
    simulation_param["pixel_size"] = torch.tensor(
        image_config["PIXEL_SIZE"], dtype=torch.float32, device=device
    )
    # other microscope parameters (for CTF and noise)
    simulation_param["voltage"] = image_config.get("VOLTAGE", 300.0)
    simulation_param["cs"] = image_config.get("SPHERICAL_ABERRATION", 0.0)
    simulation_param["dose"] = image_config.get("DOSE", 0.0)
    # detector stuff
    simulation_param["qe"] = image_config.get("QUANTUM_EFFICIENCY", 0.8)
    simulation_param["mtf_a"] = image_config.get("MTF_A", 0.3)
    simulation_param["readout_std"] = image_config.get("READOUT_STD", 1.0)

    # noise model
    simulation_param["noise"] = image_config.get("NOISE", "Gaussian")

    # check parameters for Poisson noise
    if (simulation_param["noise"]=="Poisson" or simulation_param["noise"]=="Poisson-MTF"):
       # Dose must be positive
       if (simulation_param["dose"] <= 0):
          raise ValueError("With Poisson noise model DOSE must be specified and positive")

    # Log configuration
    print("\nImage simulation parameters:")
    print(f"  Number of atoms: {natoms:,}")
    print(f"  Image size: {image_config['N_PIXELS']}×{image_config['N_PIXELS']} pixels")
    print(f"  Pixel size: {image_config["PIXEL_SIZE"]:.3f} Å")
    print(f"  Voltage: {simulation_param['voltage']:.1f} kV")
    print(f"  Spherical aberration: {simulation_param['cs']:.2f} mm")
    if "TOPOLOGY" in image_config:
        print(f"  Sigma: variable (from topology)")
        print(f"  Topology file: {topology_path}")
    else:
        print(f"  Sigma: fixed ({sigma_val:.3f} Å)")
    print(f"  Noise model: {simulation_param['noise']}")
    if (simulation_param["noise"]=="Poisson" or simulation_param["noise"]=="Poisson-MTF"):
       print(f"  Dose: {simulation_param['dose']:.1f} e/Å²")
       print(f"  Quantum efficiency: {simulation_param['qe']:.2f}")
       print(f"  Readout std: {simulation_param['readout_std']:.1f} e")
       if simulation_param["noise"]=="Poisson-MTF":
          print(f"  MTF_a at Nyquist: {simulation_param['mtf_a']:.1f}")
    print("="*70)
    
    return simulation_param


def cryo_em_simulator(
    models,
    index,
    quaternion,
    shift,
    defocus,
    b_factor,
    amp,
    snr,
    simulation_param
):
    """
    Simulates a batch of cryo-electron microscopy (cryo-EM) images of a set of given coarse-grained models.

    Args:
        models (torch.Tensor): A tensor of coarse grained models (num_models, 3, num_beads).
        index (torch.Tensor): A tensor of indices to select the models to simulate.
        quaternion (torch.Tensor): A tensor of quaternions to rotate the models.
        shift (torch.Tensor): A tensor of shifts to apply to the models.
        defocus (torch.Tensor): The defocus value of the contrast transfer function (CTF).
        b_factor (torch.Tensor): The B-factor of the CTF.
        amp (torch.Tensor): The amplitude contrast of the CTF.
        snr (torch.Tensor): The signal-to-noise ratio of the simulated image.
        simulation_param  (dict): Dictionary of simulation parameters.

    Returns:
        torch.Tensor: A tensor of the simulated (noisy) cryo-EM image.
        torch.Tensor: A tensor of the simulated (clean) cryo-EM image.
    """
    models_selected = models[index.round().long().flatten()]
    # 1. Project density on 2D plane
    image = project_density(
        models_selected,
        quaternion,
        simulation_param["sigma"],
        shift,
        simulation_param["num_pixels"], 
        simulation_param["pixel_size"]
    )
    # detach and clone the clean image
    image_clean = image.detach().clone()

    # 2. Add CTF
    image = apply_ctf(image, defocus, b_factor, amp, simulation_param["pixel_size"], simulation_param["voltage"], simulation_param["cs"])

    # 3. Add noise
    if simulation_param["noise"]=="Gaussian":
       image = add_Gaussian_noise(image, snr)
    else:
       if simulation_param["noise"]=="Poisson-MTF":
          # apply MTF blurring
          image = apply_mtf(image, simulation_param["mtf_a"], simulation_param["pixel_size"])
       # Poisson + detector noise
       image = add_Poisson_noise(image, snr, simulation_param)

    # 4. Normalize noisy and clean images
    image = gaussian_normalize_image(image)
    image_clean = gaussian_normalize_image(image_clean)
    return image, image_clean


class CryoEmSimulator:
    def __init__(self, config_fname: str, device: str = "cpu"):
        # store device
        self._device = device

        # load parameters from simulation file
        self._load_params(config_fname)

        # load models
        self._load_models()

        # initialize priors
        self._priors = get_image_priors(self.max_index, self._config, models=self._models, device=self._device)

        # get simulation parameters into dictionary 
        self._simulation_param = create_simulation_param(self._config, models=self._models, device=self._device)

 
    def _load_params(self, config_fname: str) -> None:
        """
        Loads the parameters from the config file into a dictionary.

        Args:
            config_fname (str): Path to the configuration file.

        Returns:
            None
        """
        config = json.load(open(config_fname))
        check_image_params(config)
        self._config = config

    def _load_models(self) -> None:
        """
        Loads the models from the model file specified in the config file.

        Returns:
            None

        """
        if self._config["MODEL_FILE"].endswith("npy"):
            models = (
                torch.from_numpy(
                    np.load(self._config["MODEL_FILE"]),
                )
                .to(self._device)
                .to(torch.float32)
            )
        elif self._config["MODEL_FILE"].endswith("pt"):
            models = (
                torch.load(self._config["MODEL_FILE"])
                .to(self._device)
                .to(torch.float32)
            )

        else:
            raise NotImplementedError(
                "Model file format not supported. Please use .npy or .pt."
            )

        self._models = models

        assert self._models.ndim == 3, "Models are not of shape (models, 3, atoms)."
        assert self._models.shape[1] == 3, "Models are not of shape (models, 3, atoms)."

    @property
    def max_index(self) -> int:
        """
        Returns the maximum index of the model file.

        Returns:
            int: Maximum index of the model file.
        """
        return len(self._models) - 1

    def simulate(self, num_sim, indices=None, return_parameters=False, batch_size=None):
        """
        Simulate cryo-EM images using the specified models and prior distributions.

        Args:
            num_sim (int): The number of images to simulate.
            indices (torch.Tensor, optional): The indices of the images to simulate. If None, all images are simulated.
            return_parameters (bool, optional): Whether to return the sampled parameters used for simulation.
            batch_size (int, optional): The batch size to use for simulation. If None, all images are simulated in a single batch.

        Returns:
            torch.Tensor or tuple: The simulated images as a tensor of shape (num_sim, num_pixels, num_pixels),
            and optionally the sampled parameters as a tuple of tensors.
        """

        parameters = self._priors.sample((num_sim,))
        indices = parameters[0] if indices is None else indices
        if indices is not None:
            assert isinstance(
                indices, torch.Tensor
            ), "Indices are not a torch.tensor, converting to torch.tensor."
            assert (
                indices.dtype == torch.float32
            ), "Indices are not a torch.float32, converting to torch.float32."
            assert (
                indices.ndim == 2
            ), "Indices are not a 2D tensor, converting to 2D tensor. With shape (batch_size, 1)."
            parameters[0] = indices

        images = []
        if batch_size is None:
            batch_size = num_sim
        for i in range(0, num_sim, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_parameters = [param[i : i + batch_size] for param in parameters[1:]]
            batch_images, _ = cryo_em_simulator(
                self._models,
                batch_indices,
                *batch_parameters,
                self._simulation_param
            )
            images.append(batch_images.cpu())

        images = torch.cat(images, dim=0)

        if return_parameters:
            return images.cpu(), parameters
        else:
            return images.cpu()
