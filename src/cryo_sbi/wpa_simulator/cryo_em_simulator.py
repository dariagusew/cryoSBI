import json
import numpy as np
import torch
import mrcfile
import random
from pathlib import Path
from cryo_sbi.wpa_simulator.ctf import apply_ctf
from cryo_sbi.wpa_simulator.detector import get_mtf_nps_grids 
from cryo_sbi.wpa_simulator.image_generation import project_density
from cryo_sbi.wpa_simulator.noise import add_Gaussian_noise, add_Colored_noise, add_Poisson_noise, add_GAN_ICE_noise
from cryo_sbi.wpa_simulator.noise import add_noise_from_nps, add_real_noise, MRCNoiseDataLoader
from cryo_sbi.wpa_simulator.noise_generator_ice import NoiseGeneratorICE
from cryo_sbi.wpa_simulator.image_tools import gaussian_normalize_image
from cryo_sbi.wpa_simulator.image_tools import circular_mask, make_fft_k2_grid
from cryo_sbi.inference.priors import get_image_priors
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params


# initialize all tensors/parameters for image simulation
# and pre-calculate useful stuff
def create_simulation_param(image_config: dict, models: torch.Tensor, device: str = "cuda"):
    # initialize dictionary
    simulation_param = {}
    
    # store device
    simulation_param["device"] = device

    # number of atoms
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

    # Ensure sigma has shape [num_models, 2, natoms]
    if simulation_param["sigma"].ndim == 2:
        # Old format: [2, natoms] - expand to [num_models, 2, natoms]
        simulation_param["sigma"] = simulation_param["sigma"].unsqueeze(0).expand(models.shape[0], -1, -1)

    simulation_param["num_pixels"] = torch.tensor(
        image_config["N_PIXELS"], dtype=torch.float32, device=device
    )
    simulation_param["pixel_size"] = torch.tensor(
        image_config["PIXEL_SIZE"], dtype=torch.float32, device=device
    )
    # astigmatism
    simulation_param["astigmatism"] = image_config.get("ASTIGMATISM", False)
    # other microscope parameters (for CTF and noise)
    simulation_param["voltage"] = image_config.get("VOLTAGE", 300.0)
    simulation_param["cs"] = image_config.get("SPHERICAL_ABERRATION", 0.0)
    simulation_param["dose"] = image_config.get("DOSE", 0.0)
    # detector stuff - default for K2 Summit
    simulation_param["qe"] = image_config.get("QUANTUM_EFFICIENCY", 0.8)
    simulation_param["qe_n"] = image_config.get("QUANTUM_EFFICIENCY_NYQ", 0.2)
    simulation_param["mtf_n"] = image_config.get("MTF_NYQ", 0.4)
    # detector stuff - default for Falcon 4
    #simulation_param["qe"] = image_config.get("QUANTUM_EFFICIENCY", 0.90)
    #simulation_param["qe_n"] = image_config.get("QUANTUM_EFFICIENCY_NYQ", 0.28)
    #simulation_param["mtf_n"] = image_config.get("MTF_NYQ", 0.09)
    # detector stuff - default for Falcon 4i
    #simulation_param["qe"] = image_config.get("QUANTUM_EFFICIENCY", 0.92)
    #simulation_param["qe_n"] = image_config.get("QUANTUM_EFFICIENCY_NYQ", 0.5)
    #simulation_param["mtf_n"] = image_config.get("MTF_NYQ", 0.09)
    # readout error
    simulation_param["readout_std"] = image_config.get("READOUT_STD", 1.0)

    # noise model (can be a string or a list of strings)
    noise_cfg = image_config.get("NOISE", "Gaussian")
    if isinstance(noise_cfg, str):
        noise_list = [noise_cfg]
    elif isinstance(noise_cfg, (list, tuple)):
        noise_list = list(noise_cfg)
    else:
        raise ValueError("NOISE parameter must be a string or a list of strings")

    supported_noise_models = ["Gaussian", "Colored", "Poisson", "Poisson-MTF", "empirical", "GAN-ICE", "real"]
    for n in noise_list:
        if n not in supported_noise_models:
            raise ValueError(f"Unsupported noise model '{n}', only: {', '.join(supported_noise_models)}")

    simulation_param["noise"] = noise_list

    # check parameters for Poisson noise
    if any(n in ["Poisson", "Poisson-MTF"] for n in noise_list):
       # Dose must be positive
       if (simulation_param["dose"] <= 0):
          raise ValueError("With Poisson noise models DOSE must be specified and positive")

    # prepare Poisson-MTF noise
    if any(n == "Poisson-MTF" for n in noise_list):
        # Pre-compute useful stuff
        mtf, nps, sf = get_mtf_nps_grids(simulation_param)
        # Update the dictionary with the new values.
        simulation_param["mtf"] = mtf
        simulation_param["nps-e"] = nps
        simulation_param["sf"] = sf

    # check parameters for Empirical noise
    if any(n == "empirical" for n in noise_list):
       # get path to mrc file
       mrc_file = image_config.get("NOISE_MRC", None)
       # check that noise_mrc is not None 
       if mrc_file == None:
          raise ValueError("With empirical noise models you must specify NOISE_MRC")
       # precalculate NPS
       with mrcfile.open(mrc_file) as mrc:
            nps_grid = mrc.data.astype(np.float32)
       # convert to torch
       nps_torch = torch.from_numpy(nps_grid).to(device)     
       # We need the amplitude spectrum to color the noise: sqrt(Power)
       # Clamp at zero to handle potential floating point inaccuracies.
       simulation_param["nps"] = torch.sqrt(torch.clamp(nps_torch, min=0))

    # check parameters for GAN-ICE-learned noise
    if any(n == "GAN-ICE" for n in noise_list):
       # get path to checkpoint
       pt_file = image_config.get("ICE_NOISE_PT", None)
       # check that noise_pt is not None
       if pt_file == None:
          raise ValueError("With GAN-ICE noise model you must specify ICE_NOISE_PT")
       # Initialize model generator and load model
       pt_file = Path(pt_file)
       if not pt_file.exists():
          raise FileNotFoundError(f"ICE_NOISE_PT file not found: {pt_file.resolve()}")
       # initialize generator
       simulation_param["noise_generator_ice"] = NoiseGeneratorICE(pt_file=pt_file, device=device)

    # check parameters for real noise
    if any(n == "real" for n in noise_list):
        # get path to mrc file
        mrc_noise_file = image_config.get("MRC_NOISE_FILE", None)
        # check that MRC_NOISE_FILE is not None
        if mrc_noise_file == None:
            raise ValueError("With real noise models you must specify MRC_NOISE_FILE")
        # Store the MRC file path and verify it exists
        import os
        if not os.path.exists(mrc_noise_file):
            raise ValueError(f"MRC_NOISE_FILE does not exist: {mrc_noise_file}")

        # Initialize the memory-efficient noise dataloader
        print(f"  Initializing real noise dataloader...")
        print(f"  Real noise MRC file: {mrc_noise_file}")
        simulation_param["noise_dataloader"] = MRCNoiseDataLoader(mrc_noise_file, device=device)

    # precalculate signal mask (for all noise models)
    num_pixels = int(simulation_param["num_pixels"].item())
    simulation_param["mask"] = circular_mask(num_pixels, device=device)
    # and k2 grid for CTF estimation
    pixel_size = simulation_param["pixel_size"].item()
    simulation_param["k2"] = make_fft_k2_grid(num_pixels, pixel_size, device) 

    # Log configuration
    print("\nImage simulation parameters:")
    print(f"  Number of atoms: {natoms:,}")
    print(f"  Image size: {image_config['N_PIXELS']}×{image_config['N_PIXELS']} pixels")
    print(f"  Pixel size: {image_config['PIXEL_SIZE']:.3f} Å")
    print(f"  Voltage: {simulation_param['voltage']:.1f} kV")
    print(f"  Spherical aberration: {simulation_param['cs']:.2f} mm")
    if "TOPOLOGY" in image_config:
        print(f"  Sigma: variable (from topology)")
        print(f"  Topology file: {topology_path}")
    else:
        print(f"  Sigma: fixed ({sigma_val:.1f} Å)")
    if len(noise_list) == 1:
        print(f"  Noise model: {noise_list[0]}")
    else:
        print(f"  Noise models (mixed): {', '.join(noise_list)}")
       
    if any(n in ["Poisson", "Poisson-MTF"] for n in noise_list):
       print(f"  Dose: {simulation_param['dose']:.1f} e/Å²")
       print(f"  QDE(0): {simulation_param['qe']:.3f}")
       if any(n == "Poisson-MTF" for n in noise_list):
          print(f"  QDE(Nyq): {simulation_param['qe_n']:.3f}")
          print(f"  MTF(Nyq): {simulation_param['mtf_n']:.3f}")
       print(f"  Readout std: {simulation_param['readout_std']:.1f} e")
       
    if any(n == "empirical" for n in noise_list):
       print(f"  NPS noise file: {mrc_file}")
    if any(n == "GAN-ICE" for n in noise_list):
       print(f"  Noise GAN generator loaded from: {pt_file.name}  ")

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
    simulation_param,
    noise_type
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
        noise_type (str or list): noise type(s) to sample from per batch image

    Returns:
        torch.Tensor: A tensor of the simulated (noisy) cryo-EM image.
        torch.Tensor: A tensor of the simulated (clean) cryo-EM image.
    """
    # 1. Project density on 2D plane
    image = project_density(
        models,
        index,
        quaternion,
        simulation_param["sigma"],
        shift,
        simulation_param["num_pixels"], 
        simulation_param["pixel_size"]
    )

    # 2. Add CTF
    image = apply_ctf(image, defocus, b_factor, amp, simulation_param)

    # detach and clone the clean image
    image_clean = image.detach().clone()

    # 3. Add noise
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    batch_noise_types = random.choices(noise_types, k=image.shape[0])
    noisy_image = torch.empty_like(image)
    for n_type in set(batch_noise_types):
        indices = [i for i, t in enumerate(batch_noise_types) if t == n_type]
        sub_image = image[indices]
        sub_snr = snr[indices]

        if n_type == "Gaussian":
            sub_noisy = add_Gaussian_noise(sub_image, sub_snr, simulation_param["mask"])

        elif n_type == "Colored":
            sub_noisy = add_Colored_noise(sub_image, sub_snr, simulation_param["mask"], simulation_param["pixel_size"])

        elif n_type in ["Poisson", "Poisson-MTF"]:
            mtf = simulation_param["mtf"] if n_type == "Poisson-MTF" else None
            nps = simulation_param["nps-e"] if n_type == "Poisson-MTF" else None
            sf = simulation_param["sf"].expand(sub_snr.shape[0], 1, 1) if n_type == "Poisson-MTF" else torch.ones_like(sub_snr)
            target_snr = sf * sub_snr
            sub_noisy = add_Poisson_noise(sub_image, target_snr, simulation_param, mtf, nps)

        elif n_type == "GAN-ICE":
            sub_noisy = add_GAN_ICE_noise(sub_image, simulation_param["noise_generator_ice"], sub_snr, simulation_param["mask"])

        elif n_type == "real":
            sub_noisy = add_real_noise(sub_image, sub_snr, simulation_param["noise_dataloader"], simulation_param["mask"])

        else:
            sub_noisy = add_noise_from_nps(sub_image, sub_snr, simulation_param["nps"], simulation_param["mask"])

        noisy_image[indices] = sub_noisy
    image = noisy_image

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
                self._simulation_param,
                self._simulation_param["noise"]
            )
            images.append(batch_images.cpu())

        images = torch.cat(images, dim=0)

        if return_parameters:
            return images.cpu(), parameters
        else:
            return images.cpu()
