from typing import Union, Callable
import json
import numpy as np
import torch

from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_noise
from cryo_sbi.wpa_simulator.normalization import gaussian_normalize_image
from cryo_sbi.inference.priors import get_image_priors
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params

def cryo_em_simulator(
    models,
    index,
    quaternion,
    shift,
    defocus,
    b_factor,
    amp,
    snr,
    sigma,
    num_pixels,
    pixel_size,
    voltage,
    cs
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
        sigma (torch.Tensor): Parameters of Gaussian kernel used to project the density.
        num_pixels (torch.Tensor): The number of pixels in the simulated image.
        pixel_size (torch.Tensor): The size of each pixel in the simulated image.
        voltage (float): Electron voltage in kV
        cs (float): Spherical aberration in mm

    Returns:
        torch.Tensor: A tensor of the simulated (noisy) cryo-EM image.
        torch.Tensor: A tensor of the simulated (clean) cryo-EM image.
    """
    models_selected = models[index.round().long().flatten()]
    image = project_density(
        models_selected,
        quaternion,
        sigma,
        shift,
        num_pixels,
        pixel_size,
    )
    # detach and clone the clean image
    image_clean = image.detach().clone()
    # add CTF
    image = apply_ctf(image, defocus, b_factor, amp, pixel_size, voltage, cs)
    # add noise
    image = add_noise(image, snr)
    # normalize noisy and clean images
    image = gaussian_normalize_image(image)
    image_clean = gaussian_normalize_image(image_clean)
    return image, image_clean


class CryoEmSimulator:
    def __init__(self, config_fname: str, device: str = "cpu"):
        self._device = device
        self._load_params(config_fname)
        self._load_models()
        self._priors = get_image_priors(self.max_index, self._config, models=self._models, device=device)
        self._num_pixels = torch.tensor(
            self._config["N_PIXELS"], dtype=torch.float32, device=device
        )
        self._pixel_size = torch.tensor(
            self._config["PIXEL_SIZE"], dtype=torch.float32, device=device
        )
        self._voltage = self._config.get("VOLTAGE", 300.0)
        self._cs = self._config.get("SPHERICAL_ABERRATION", 0.0)
        # sigma stuff 
        natoms = self._models.shape[2]
    
        if "TOPOLOGY" in self._config: 
            # Load TOPOLOGY from file path
            topology_path = self._config["TOPOLOGY"]
            self._sigma = torch.load(topology_path, map_location=device)
    
        elif "SIGMA" in self._config: 
            sigma_value = self._config["SIGMA"]

            # Extract scalar value (or first element if list)
            if isinstance(sigma_value, (list, tuple)):
                sigma_val = torch.as_tensor(sigma_value[0], device=device)
            else:
                sigma_val = torch.as_tensor(sigma_value, device=device)
           
            # Create sigma tensor [2, natoms] on device
            self._sigma = torch.zeros(2, natoms, device=device)
            self._sigma[0, :] = 1.0 / torch.sqrt(natoms * 2 * torch.pi * sigma_val**2)
            self._sigma[1, :] = -0.5 / (sigma_val ** 2)
        else:
            raise ValueError("Either TOPOLOGY or SIGMA must be specified in image_config")

 
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
                self._sigma,
                self._num_pixels,
                self._pixel_size,
                self._voltage,
                self._cs
            )
            images.append(batch_images.cpu())

        images = torch.cat(images, dim=0)

        if return_parameters:
            return images.cpu(), parameters
        else:
            return images.cpu()
