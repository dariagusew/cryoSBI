#!/usr/bin/env python3
"""
visualize_gan_output.py (Robust Memory-Efficient Version)

Loads a trained noise GAN model and an MRC file of real noise patches to
generate a side-by-side comparison figure.

This version is optimized to handle very large MRC files by explicitly using
memory-mapping (`mmap`) for all file access, preventing the entire dataset
from being loaded into RAM at any point.
"""

import argparse
import contextlib
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mrcfile
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ===========================================================================
# 1. MODEL & UTILITY DEFINITIONS
# ===========================================================================

_GROUPS = 8

class ResBlockG(nn.Module):
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_GROUPS, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_GROUPS, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class Generator(nn.Module):
    _DILATIONS: List[int] = [1, 2, 4, 8, 4, 2, 1, 1]
    def __init__(self, base_channels: int = 64, n_blocks: int = 8):
        super().__init__()
        ch = base_channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, ch, 7, padding=3), nn.GroupNorm(_GROUPS, ch), nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResBlockG(ch, dilation=self._DILATIONS[i % len(self._DILATIONS)])
            for i in range(n_blocks)
        ])
        self.output_conv = nn.Conv2d(ch, 1, 7, padding=3)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(z)
        for rb in self.res_blocks: h = rb(h)
        return self.output_conv(h)

class EMA:
    def __init__(self, model: nn.Module):
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().float().clone()
            for k, v in model.named_parameters()
        }
    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        backup = {k: v.detach().clone() for k, v in model.named_parameters()}
        for k, p in model.named_parameters():
            p.data.copy_(self.shadow[k].to(p.device))
        try:
            yield
        finally:
            for k, p in model.named_parameters():
                p.data.copy_(backup[k])
    def load_state_dict(self, sd: Dict, device: torch.device) -> None:
        self.shadow = {k: v.to(device) for k, v in sd.items()}

def _normalise_np(x: np.ndarray) -> np.ndarray:
    """Per-image zero mean, unit std on a NumPy array batch."""
    mean = x.mean(axis=(-2, -1), keepdims=True)
    std = x.std(axis=(-2, -1), keepdims=True)
    return (x - mean) / np.where(std > 1e-8, std, 1.0)


# ===========================================================================
# 2. BATCHED ANALYSIS & VISUALIZATION
# ===========================================================================

def _radially_average_psd(
    psd_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to perform radial averaging on a 2D Power Spectrum Density."""
    h, w = psd_2d.shape
    freq_y = np.fft.fftshift(np.fft.fftfreq(h))
    freq_x = np.fft.fftshift(np.fft.fftfreq(w))
    kx, ky = np.meshgrid(freq_x, freq_y)
    radial_freq = np.sqrt(kx**2 + ky**2)
    num_bins = min(h, w) // 2
    freq_bins = np.linspace(0.0, 0.5, num_bins + 1)
    power_sum, _ = np.histogram(
        radial_freq.ravel(), bins=freq_bins, weights=psd_2d.ravel()
    )
    counts, _ = np.histogram(radial_freq.ravel(), bins=freq_bins)
    nps = np.divide(
        power_sum, counts, out=np.zeros_like(power_sum, dtype=float),
        where=(counts != 0)
    )
    freqs = (freq_bins[:-1] + freq_bins[1:]) / 2
    return freqs, nps

def calculate_nps_in_batches(
    image_generator: Generator[np.ndarray, None, None],
    total_images: int,
    description: str = "Calculating NPS"
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates NPS by iterating through batches of images to save memory."""
    total_psd_2d = None
    processed_count = 0
    pbar = tqdm(image_generator, total=total_images, desc=description, unit="img")
    for batch_np in pbar:
        if total_psd_2d is None:
            h, w = batch_np.shape[-2:]
            total_psd_2d = np.zeros((h, w), dtype=np.float64)
        batch_np = _normalise_np(batch_np)
        fourier = np.fft.fft2(batch_np)
        power_spectra_2d = np.abs(fourier) ** 2
        total_psd_2d += power_spectra_2d.sum(axis=0)
        processed_count += len(batch_np)
        pbar.update(len(batch_np))
    pbar.close()
    mean_psd_2d = total_psd_2d / processed_count
    mean_psd_2d_shifted = np.fft.fftshift(mean_psd_2d)
    return _radially_average_psd(mean_psd_2d_shifted)

def generate_real_batches(
    mrc_path: Path, indices: np.ndarray, batch_size: int
) -> Generator[np.ndarray, None, None]:
    """Yields batches of real images from an MRC file using memory-mapping."""
    with mrcfile.mmap(str(mrc_path), mode='r', permissive=True) as mrc:
        for i in range(0, len(indices), batch_size):
            batch_indices = sorted(indices[i : i + batch_size])
            yield mrc.data[batch_indices].astype(np.float32)

def generate_fake_batches(
    G: nn.Module, ema: EMA, total: int, h: int, w: int, batch_size: int, device: torch.device
) -> Generator[np.ndarray, None, None]:
    """Yields batches of synthetic images from the GAN."""
    with ema.applied(G), torch.no_grad():
        for i in range(0, total, batch_size):
            current_batch_size = min(batch_size, total - i)
            z = torch.randn(current_batch_size, 1, h, w, device=device)
            fake_batch = G(z)
            yield fake_batch.squeeze(1).cpu().numpy()

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize trained noise GAN output and compare NPS (memory-efficient).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", required=True, type=Path,
                    help="Path to the trained GAN checkpoint (.pt file).")
    ap.add_argument("--mrc_path", required=True, type=Path,
                    help="Path to an MRC stack of real noise patches.")
    ap.add_argument("--num_images", type=int, default=8,
                    help="Number of example images to show. Should be an even number.")
    ap.add_argument("--nps_count", type=int, default=10000,
                    help="Number of images to use for NPS calculation.")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for processing. Lower this if you get OOM errors.")
    ap.add_argument("--output_path", type=Path, default="gan_visualization.png",
                    help="Path to save the output visualization image.")
    ap.add_argument("--device", type=str, default=None,
                    help="Device to use ('cuda', 'cpu'). Autodetects if None.")
    args = ap.parse_args()

    if args.num_images % 2 != 0:
        print(f"Warning: --num_images is {args.num_images}. For best layout, please use an even number. Adjusting to {args.num_images-1}.")
        args.num_images -= 1

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location="cpu")
    G = Generator(**ckpt["gen_cfg"]).to(device)
    ema = EMA(G)
    ema.load_state_dict(ckpt["ema"], device)
    G.eval()
    print("Model loaded successfully.")

    print(f"Reading header from {args.mrc_path}...")
    with mrcfile.mmap(str(args.mrc_path), mode='r', permissive=True) as mrc:
        total_in_mrc = mrc.data.shape[0]
        h, w = int(mrc.header.ny), int(mrc.header.nx)
        if args.nps_count > total_in_mrc:
            raise ValueError(f"--nps_count cannot be larger than images in MRC file.")
        real_indices = np.random.choice(total_in_mrc, args.nps_count, replace=False)

    print(f"Image size: {h}x{w}")
    print(f"Loading {args.num_images} examples for visualization...")
    vis_indices = real_indices[:args.num_images]
    with mrcfile.mmap(str(args.mrc_path), mode='r', permissive=True) as mrc:
        real_vis_images = mrc.data[sorted(vis_indices)].astype(np.float32)
    real_vis_images = _normalise_np(real_vis_images)

    with ema.applied(G), torch.no_grad():
        z_vis = torch.randn(args.num_images, 1, h, w, device=device)
        fake_vis_images = G(z_vis).squeeze(1).cpu().numpy()
    fake_vis_images = _normalise_np(fake_vis_images)

    print(f"\nCalculating NPS from {args.nps_count} images (batch size: {args.batch_size})...")
    real_gen = generate_real_batches(args.mrc_path, real_indices, args.batch_size)
    real_freqs, real_nps = calculate_nps_in_batches(real_gen, args.nps_count, "Real NPS")
    
    fake_gen = generate_fake_batches(G, ema, args.nps_count, h, w, args.batch_size, device)
    fake_freqs, fake_nps = calculate_nps_in_batches(fake_gen, args.nps_count, "Synthetic NPS")
    
    print(f"\nCreating visualization and saving to {args.output_path}...")
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    
    layout = [["synth_panel", "real_panel"],
              ["nps_panel", "nps_panel"]]
    
    # ----- THE FIX IS HERE -----
    ax_dict = fig.subplot_mosaic(
        layout,
        height_ratios=[1, 1],
        width_ratios=[1, 1]  # Enforces equal width for the top panels
    )

    def plot_image_grid(panel_ax, images, title):
        panel_ax.set_title(title, fontsize=14, weight='bold')
        panel_ax.axis('off')
        
        num_rows = 2
        num_cols = len(images) // num_rows
        gs = gridspec.GridSpecFromSubplotSpec(
            num_rows, num_cols, subplot_spec=panel_ax.get_subplotspec(), 
            wspace=0.05, hspace=0.05
        )
        
        for i, img in enumerate(images):
            ax = fig.add_subplot(gs[i])
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plot_image_grid(ax_dict["synth_panel"], fake_vis_images, "Synthetic Images")
    plot_image_grid(ax_dict["real_panel"], real_vis_images, "Real Images")
    
    ax_nps = ax_dict["nps_panel"]
    
    ax_nps.plot(real_freqs[1:], real_nps[1:], label="Real NPS", color="steelblue", lw=2)
    ax_nps.plot(fake_freqs[1:], fake_nps[1:], label="Synthetic NPS", color="tomato", lw=2, ls="--")
    
    ax_nps.set_yscale("log")
    ax_nps.set_title("Radially Averaged Noise Power Spectrum", fontsize=14, weight='bold')
    ax_nps.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax_nps.set_ylabel("Power (log scale)")
    ax_nps.legend()
    ax_nps.grid(True, which="both", ls="--", alpha=0.5)

    fig.suptitle(f"GAN Noise Model Evaluation: {args.model_path.name}", fontsize=16)
    plt.savefig(str(args.output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Done.")

if __name__ == "__main__":
    main()
