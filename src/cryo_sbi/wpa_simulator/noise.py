import torch
import mrcfile
import numpy as np
from typing import Optional, Tuple


def add_Gaussian_noise(
    image: torch.Tensor,
    snr: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Adds Gaussian white noise to images based on power SNR.
    
    SNR definition: SNR = signal_variance / noise_variance
    Therefore: noise_std = signal_std / sqrt(SNR)
    
    Args:
        image (torch.Tensor): Image tensor of shape (batch, height, width)
        snr (torch.Tensor): Signal-to-noise ratio (power SNR)
        mask (torch.Tensor): Signal mask
        
    Returns:
        Noisy image with same shape as input
    """
    # Calculate signal standard deviation within mask
    signal_std = torch.std(image[:, mask], dim=[-1])  # (batch,)
    
    # Calculate noise standard deviation from power SNR
    # SNR = σ²_signal / σ²_noise → σ_noise = σ_signal / sqrt(SNR) [batch, 1, 1]
    noise_std = signal_std.reshape(-1, 1, 1) / torch.sqrt(snr)
    
    # Generate noise map [batch, npixels, npixels]
    noise = torch.randn_like(image)
    noise = noise * noise_std
    # Final noisy image [batch, npixels, npixels] 
    image_noise = image + noise
    
    return image_noise


def add_Poisson_noise(
    image: torch.Tensor,
    target_snr: torch.Tensor,
    simulation_param: dict,
    mtf: Optional[torch.Tensor] = None,
    nps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Adds Poisson and readout noise to images.

    Args:
        image (torch.Tensor): Image tensor of shape (batch, height, width) 
        target_snr (torch.Tensor): Signal-to-noise target ratio (power SNR)
        simulation_param (dict): Dictionary of simulation parameters
        mtf (optional, torch.Tensor): MTF grid
        nps (optional, torch.Tensor): NPS excess noise grid

    Returns:
        Noisy image with the same shape as the input.
    """
    # 1. Setup and Parameter Extraction
    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size = simulation_param["pixel_size"].item() 
    else:
        pixel_size = simulation_param["pixel_size"]

    # 2. Get mask 
    mask = simulation_param["mask"]
    
    # Calculate variance from the input image within the mask.
    signal_var = torch.var(image[:, mask], dim=[-1]).view(-1, 1, 1)

    # 3. Core Noise Simulation
    # Convert dose from e/Å² to e/pixel
    mean_electron_dose = simulation_param["dose"] * pixel_size**2

    # 4. Define quantum efficieny
    # With "Poisson" noise: use qe = DQE(0) = simulation_param["qe"]
    # With "Poisson-MTF": set qe=1.0 -> it will be taken care by MTF/DQE
    if mtf is None and nps is None:
       qe = simulation_param["qe"]
    else:
       qe = 1.0

    # 5. Determine the contrast scale for each image based on its target SNR
    # Deal with zero variance corner cases
    contrast_scale = torch.where(
          signal_var > 0,
          torch.sqrt(target_snr / (mean_electron_dose * qe * signal_var)),
          torch.zeros_like(signal_var)
    )

    # 6. Determine the mean electron count per pixel
    mean_counts_per_pixel = qe * mean_electron_dose * (1.0 + contrast_scale * image)

    # 7. Generate Poisson shot noise
    # First, clamp mean_counts_per_pixel to be non-negative
    mean_counts_per_pixel = torch.clamp(mean_counts_per_pixel, min=0)
    image_noise = torch.poisson(mean_counts_per_pixel)

    # 8. Adding MTF/NPS corrections
    if mtf is not None and nps is not None:
       # 8.1 Apply Modulation Transfer Function 
       image_fft = torch.fft.fft2(image_noise)
       image_fft_mtf = image_fft * mtf
       image_noise = torch.fft.ifft2(image_fft_mtf).real

       # 8.2 Apply Detective Quantum Efficiency correction by adding excess noise
       # Generate white noise in Fourier space
       white_noise_fft = torch.fft.fft2(torch.randn_like(image_noise))
       
       # Scale by sqrt(NPS) to color the noise
       colored_noise_fft = white_noise_fft * torch.sqrt(nps)
       
       # Inverse FFT to get the noise in real space and add it
       colored_noise = torch.fft.ifft2(colored_noise_fft).real
       image_noise = image_noise + colored_noise 

    # 9. Add Gaussian Readout Noise
    readout_noise = torch.randn_like(image_noise) * simulation_param["readout_std"]
    final_image = image_noise + readout_noise

    return final_image


def add_noise_from_nps(
    image: torch.Tensor,
    snr: torch.Tensor,
    nps: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Adds colored noise to images based on a Noise Power Spectrum (NPS) file.

    This function generates a noise realization with the spectral characteristics
    defined in the NPS, then scales it to achieve a target power SNR.

    SNR definition: SNR = signal_variance / noise_variance

    Args:
        image (torch.Tensor): Image tensor of shape (batch, height, width)
        snr (torch.Tensor): Signal-to-noise ratio (power SNR)
        nps (torch.Tensor): NPS from mrc file
        mask (torch.Tensor): Signal mask

    Returns:
        torch.Tensor: Noisy image with the same shape as the input.
    """
    # 1. Create a realization of colored noise
    # Start with white noise in real space for each image in the batch
    white_noise_real = torch.randn_like(image)

    # Go to Fourier space
    white_noise_fft = torch.fft.fft2(white_noise_real)

    # Color the noise by multiplying with the NPS amplitude spectrum.
    colored_noise_fft = white_noise_fft * nps.unsqueeze(0)

    # Go back to real space to get the unscaled noise realization
    unscaled_colored_noise = torch.fft.ifft2(colored_noise_fft).real

    # 2. Calculate signal and noise variances
    # Signal variance
    signal_var = torch.var(image[:, mask], dim=[-1]) # Shape: (batch,)

    # Noise variance
    noise_var = torch.var(unscaled_colored_noise, dim=[-2, -1]) # Shape: (batch,)

    # 4. Rescale noise to match the target SNR
    # We want: SNR = signal_var / var(scaled_noise)
    # Let scaled_noise = C * unscaled_colored_noise
    # Then var(scaled_noise) = C^2 * var(unscaled_colored_noise)
    # So, SNR = signal_var / (C^2 * noise_var)
    # Solving for C: C = sqrt(signal_var / (SNR * noise_var))

    # Add a small epsilon to prevent division by zero
    epsilon = 1e-9
    snr_squeezed = snr.squeeze() # From (batch, 1, 1) to (batch,)

    scaling_factor_sq = torch.where(
          signal_var > 0,
          signal_var / (snr_squeezed * noise_var + epsilon), 
          torch.ones_like(signal_var)
    )
    scaling_factor = torch.sqrt(torch.clamp(scaling_factor_sq, min=0))

    # Reshape for broadcasting over the image dimensions
    scaling_factor = scaling_factor.view(-1, 1, 1)

    # 5. Add scaled noise to the image
    scaled_noise = unscaled_colored_noise * scaling_factor
    final_image = image + scaled_noise

    return final_image

def add_GAN_noise(
    image:           torch.Tensor,
    noise_generator: "NoiseGenerator",
    snr:             torch.Tensor,
    mask:            torch.Tensor,
) -> torch.Tensor:
    """
    Adds GAN-generated structured ice noise to images based on power SNR.

    The GAN generator produces noise patches that are already normalised
    (mean=0, std=1 per patch).  These are rescaled to match the target
    noise power derived from the SNR, following the same convention as
    add_Gaussian_noise so the two functions are drop-in replacements.

    SNR definition: SNR = signal_variance / noise_variance
    Therefore:      noise_std = signal_std / sqrt(SNR)

    Args:
        image           (torch.Tensor) : clean image, shape (batch, H, W)
        noise_generator (NoiseGenerator): trained GAN noise generator
        snr             (torch.Tensor) : power SNR, shape (batch,) or scalar
        mask            (torch.Tensor) : boolean signal mask, shape (H, W)

    Returns:
        torch.Tensor: noisy image, same shape and device as *image*
    """
    batch, H, W = image.shape

    # ── signal statistics within mask
    signal_std = torch.std(image[:, mask], dim=[-1])          # (batch,)

    # SNR = σ²_signal / σ²_noise  →  σ_noise = σ_signal / √SNR
    noise_std = signal_std.reshape(-1, 1, 1) / torch.sqrt(snr)  # (batch,1,1)

    # ── GAN noise generation ───────────────────────────────────────────────
    # sample() returns (batch, H, W) float32 on noise_generator.device
    # Each patch has mean=0, std=1 by construction (_normalise in sample())
    noise = noise_generator.sample(n=batch, box_size=H)

    # ── rescale to target noise power ─────────────────────────────────────
    # noise has std=1  →  noise * noise_std has std=noise_std
    noise = noise * noise_std

    return image + noise


class MRCNoiseDataLoader:
    """
    Memory-efficient dataloader for large MRC noise files.

    Uses mrcfile.mmap() to memory-map the file. A cache of `cache_size`
    randomly sampled particles is kept on `device` and served as sequential
    batches. When the cache is exhausted a fresh random cache is loaded
    from disk and transferred to the device in one shot, amortising both
    the cost of scattered disk reads and the CPU→device transfer.

    Args:
        mrc_file_path (str)          : Path to the MRC file containing noise data.
        cache_size    (int)          : Number of particles to hold on device at once.
        device        (torch.device) : Target device for the cache (cpu or cuda).
                                       Defaults to CPU if not specified.
    """

    def __init__(
        self,
        mrc_file_path: str,
        cache_size: int = 10240,
        device: Optional[torch.device] = None,
    ):
        self.mrc_file_path = mrc_file_path
        self.cache_size    = cache_size
        self.device        = device or torch.device("cpu")

        self.mrc       = mrcfile.mmap(mrc_file_path, mode="r")
        self.mmap_data = self.mrc.data          # numpy memmap (N, H, W)

        self.num_particles  = self.mmap_data.shape[0]
        self.height         = self.mmap_data.shape[1]
        self.width          = self.mmap_data.shape[2]
        self.particle_shape = (self.height, self.width)

        self.cache     = None
        self.cache_idx = 0
        self._fill_cache()

        # Estimate cache memory footprint
        cache_bytes = self.cache_size * self.height * self.width * 4
        cache_gb    = cache_bytes / (1024 ** 3)

        print(f"  MRC Noise DataLoader initialized:")
        print(f"    Total particles in file : {self.num_particles:,}")
        print(f"    Particle shape          : {self.particle_shape}")
        print(f"    Dtype                   : {self.mmap_data.dtype}")
        print(f"    Cache size              : {self.cache_size:,}")
        print(f"    Cache device            : {self.device}")
        print(f"    Cache memory footprint  : {cache_gb:.2f} GB")
        print(f"    Memory-mapped access    : True")

    def _fill_cache(self):
        """
        Sample cache_size random particles from disk and transfer to device in
        one contiguous copy, amortising both scattered disk reads and the
        CPU→device transfer over the entire cache.
        """
        indices  = np.random.randint(0, self.num_particles, size=self.cache_size)
        cache_np = np.array(self.mmap_data[indices], dtype=np.float32)
        self.cache     = torch.from_numpy(cache_np).to(self.device)
        self.cache_idx = 0

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """
        Return batch_size particles from the on-device cache.
        Refills the cache automatically when exhausted.

        Args:
            batch_size (int): Number of particles to retrieve.

        Returns:
            torch.Tensor: shape (batch_size, height, width), dtype float32,
                          already on self.device.
        """
        if self.cache_idx + batch_size > self.cache_size:
            self._fill_cache()

        batch = self.cache[self.cache_idx : self.cache_idx + batch_size]
        self.cache_idx += batch_size
        return batch

    def __del__(self):
        if hasattr(self, "mrc"):
            self.mrc.close()


def add_real_noise(
    image: torch.Tensor,
    snr: torch.Tensor,
    noise_dataloader: MRCNoiseDataLoader,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Add real cryo-EM ice noise to a batch of images — no Python loops.

    All operations (augmentation, normalisation, SNR scaling) are applied
    across the full batch dimension in parallel. Noise particles are fetched
    from the on-device cache so no CPU→device transfer occurs at call time.

    After Z-score normalisation the noise variance is exactly 1, so the
    scaling factor simplifies to sqrt(signal_var / snr).

    Random D4 augmentation (all 8 square symmetries) is applied via three
    independent binary choices: transpose, vertical flip, horizontal flip.

    Args:
        image            (B, H, W): Input image batch.
        snr              scalar / (1,) / (B,) / (B,1): Power signal-to-noise ratio.
        noise_dataloader           : MRCNoiseDataLoader instance.
        mask             (H, W) bool: Mask selecting signal pixels.

    Returns:
        torch.Tensor: Noisy batch, shape (B, H, W).
    """
    batch_size = image.shape[0]
    device     = image.device

    # Calculate signal standard deviation within mask
    signal_std = torch.std(image[:, mask], dim=[-1])    # (B,)

    # Fetch noise particles — already on device, no transfer cost
    noise_batch = noise_dataloader.get_batch(batch_size)  # (B, H, W)

    # Random D4 augmentation: independent transpose + vertical flip + horizontal flip
    # covers all 8 square symmetries uniformly
    do_T  = (torch.rand(batch_size, device=device) > 0.5).view(batch_size, 1, 1)
    do_ud = (torch.rand(batch_size, device=device) > 0.5).view(batch_size, 1, 1)
    do_lr = (torch.rand(batch_size, device=device) > 0.5).view(batch_size, 1, 1)

    noise_batch = torch.where(do_T,  noise_batch.transpose(1, 2),       noise_batch)
    noise_batch = torch.where(do_ud, torch.flip(noise_batch, dims=[1]), noise_batch)
    noise_batch = torch.where(do_lr, torch.flip(noise_batch, dims=[2]), noise_batch)

    # Per-sample Z-score normalisation (noise variance is exactly 1 after this)
    noise_mean = noise_batch.mean(dim=(1, 2), keepdim=True)             # (B, 1, 1)
    noise_std  = noise_batch.std( dim=(1, 2), keepdim=True)             # (B, 1, 1)
    noise_batch = torch.where(
        noise_std > 0,
        (noise_batch - noise_mean) / noise_std,
        noise_batch,
    )

    # Calculate noise standard deviation from power SNR
    # SNR = σ²_signal / σ²_noise → σ_noise = σ_signal / sqrt(SNR)
    noise_std = signal_std.reshape(-1, 1, 1) / torch.sqrt(snr)          # (B, 1, 1)

    # Scale and add
    return image + noise_batch * noise_std
