"""
validate_embedding_v7.py

Validates a pretrained image encoder by visualising its embedding space
using UMAP.

Produces a 2×2 figure:
    Row 0 left:  Example synthetic images
    Row 0 right: Example real images (if real MRC provided)
    Row 1 left:  Synthetic UMAP, coloured by conformation index
    Row 1 right: Synthetic + real UMAP; synthetic by conformation, real in orange

Usage:
    python validate_embedding_v7.py \
        --image_config config.json \
        --encoder_weights pretrained_image_embed.pt \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --n_synthetic 2000 \
        --real_data_mrc real_images.mrc \
        --n_real 2000 \
        --output embedding_validation.png
"""

import argparse
import json
import math
import struct
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import umap as umap_lib
except ImportError:
    raise ImportError("umap-learn is required. Install with: pip install umap-learn")

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not installed.")

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import (
    cryo_em_simulator,
    create_simulation_param,
)


# ============================================================================
# ROBUST MRC FILE HANDLING (unchanged from v4)
# ============================================================================

def check_mrc_file_size(filepath):
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    return file_size, file_size / (1024**3)


def validate_mrc_data(data):
    if data is None or data.size == 0 or data.ndim not in [2, 3]:
        return False, f"Invalid data shape or type: {data.shape if hasattr(data, 'shape') else 'None'}"
    try:
        if isinstance(data, np.memmap):
            test_data = data[0] if data.ndim == 3 else data
        else:
            test_data = data
        if np.all(test_data == 0): return False, "All data is zero"
        if np.any(np.isnan(test_data)): return False, "Data contains NaN"
        if np.any(np.isinf(test_data)): return False, "Data contains inf"
        if np.std(test_data) == 0: return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


def read_mrc_header_raw(filepath):
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(1024)
            if len(header_bytes) < 1024: return None
            nx, ny, nz = struct.unpack('iii', header_bytes[0:12])
            mode = struct.unpack('i', header_bytes[12:16])[0]
            return {'nx': nx, 'ny': ny, 'nz': nz, 'mode': mode}
    except:
        return None


def get_dtype_from_mode(mode):
    dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    return dtype_map.get(mode, np.float32)


def validate_mrc_dimensions(nx, ny, nz):
    if nx <= 0 or ny <= 0 or nz <= 0: return False, f"Non-positive: {nz}×{ny}×{nx}"
    if nx > 8192 or ny > 8192: return False, f"Too large: {ny}×{nx}"
    if nz > 50000000: return False, f"Stack too large: {nz}"
    return True, "Valid"


def open_mrc_robust(filepath, max_size_gb=None):
    filepath = Path(filepath)
    if not filepath.exists():
        return None, False, "File not found"
    file_size, file_size_gb = check_mrc_file_size(filepath)
    if max_size_gb is not None and file_size_gb > max_size_gb:
        return None, False, f"Too large: {file_size_gb:.2f} GB"
    try:
        header_info = read_mrc_header_raw(filepath)
        if header_info is not None:
            nx, ny, nz, mode = (
                header_info['nx'], header_info['ny'],
                header_info['nz'], header_info['mode'],
            )
            is_valid, msg = validate_mrc_dimensions(nx, ny, nz)
            if not is_valid:
                return None, False, msg
            dtype = get_dtype_from_mode(mode)
            data = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            is_valid, msg = validate_mrc_data(data)
            if is_valid:
                return data, True, "Memmap via manual header read"
    except Exception as e:
        return None, False, f"Failed: {str(e)[:100]}"
    return None, False, "All MRC opening methods failed"


class RealImageMRCDataset(Dataset):
    """Dataset for loading real images from MRC stack."""

    def __init__(self, mrc_path, cache_size=10000):
        self.mrc_path   = mrc_path
        self.cache_size = cache_size
        print(f"  Opening MRC file: {mrc_path}")
        self.mrc_data, success, method = open_mrc_robust(mrc_path)
        if not success:
            raise RuntimeError(f"Failed to open MRC file: {method}")
        self.n_images   = self.mrc_data.shape[0]
        self.image_shape = self.mrc_data.shape[1:]
        print(f"  Loaded MRC: {self.n_images} images of shape {self.image_shape}")
        print(f"  Loading method: {method}")
        self.cache       = {}
        self.cache_order = deque()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img = self.mrc_data[idx].astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.popleft()
            del self.cache[oldest]
        self.cache[idx] = torch.from_numpy(img.copy())
        self.cache_order.append(idx)
        return self.cache[idx]


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _tile_images(images: np.ndarray, n_cols: int = 4, padding: int = 4) -> np.ndarray:
    """
    Tile [N, H, W] images into a single mosaic with white padding between them.
    Each image is individually z-score normalised for consistent contrast.
    """
    n, h, w  = images.shape
    n_rows   = math.ceil(n / n_cols)
    pad_val  = images.max()  # white padding
    cell_h   = h + padding
    cell_w   = w + padding
    canvas   = np.full(
        (n_rows * cell_h + padding, n_cols * cell_w + padding),
        fill_value=pad_val,
        dtype=np.float32,
    )
    for i, img in enumerate(images[:n_rows * n_cols]):
        r, c     = divmod(i, n_cols)
        img_norm = (img - img.mean()) / (img.std() + 1e-8)
        # Rescale to [pad_val*0, pad_val] so background stays white
        img_norm = (img_norm - img_norm.min()) / (img_norm.ptp() + 1e-8) * pad_val
        row_start = padding + r * cell_h
        col_start = padding + c * cell_w
        canvas[row_start:row_start + h, col_start:col_start + w] = img_norm
    return canvas


# ============================================================================
# DATA GENERATION AND ENCODING
# ============================================================================

def generate_and_encode_synthetic(
    model: nn.Module,
    image_config: dict,
    models: torch.Tensor,
    device: str,
    n_images: int,
    simulation_batch_size: int,
    encode_batch_size: int,
    simulation_param: dict,
    n_example_images: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic cryo-EM images and encode them.

    Returns:
        mus:            [N, D]                   float32 numpy (deterministic)
        zs:             [N, D]                   float32 numpy (stochastic)
        indices:        [N]                       int conformation indices
        example_images: [n_example_images, H, W] float32 numpy
    """
    image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    synthetic_loader = PriorLoader(
        image_prior, batch_size=simulation_batch_size, num_workers=4
    )
    synthetic_iter = iter(synthetic_loader)

    all_mus     = []
    all_zs      = []
    all_indices = []
    example_buf = []
    n_collected = 0

    pbar = tqdm(total=n_images, desc="Generating + encoding synthetic images")

    model.eval()
    with torch.no_grad():
        while n_collected < n_images:
            try:
                parameters = next(synthetic_iter)
            except StopIteration:
                synthetic_iter = iter(synthetic_loader)
                parameters = next(synthetic_iter)

            indices, quaternions, shift, defocus, b_factor, amp, snr = parameters

            images, _ = cryo_em_simulator(
                models,
                indices.to(device,     non_blocking=True),
                quaternions.to(device, non_blocking=True),
                shift.to(device,       non_blocking=True),
                defocus.to(device,     non_blocking=True),
                b_factor.to(device,    non_blocking=True),
                amp.to(device,         non_blocking=True),
                snr.to(device,         non_blocking=True),
                simulation_param,
                simulation_param["noise"],
            )

            # Collect example images from the first batch only
            if len(example_buf) < n_example_images:
                n_grab = min(n_example_images - len(example_buf), len(images))
                example_buf.extend(img.cpu() for img in images[:n_grab])

            batch_mus = []
            batch_zs  = []
            for i in range(0, len(images), encode_batch_size):
                mu, log_var, z = model.forward_vib(images[i:i + encode_batch_size])
                batch_mus.append(mu.cpu())
                batch_zs.append(z.cpu())
            batch_mus = torch.cat(batch_mus, dim=0)
            batch_zs  = torch.cat(batch_zs,  dim=0)

            n_to_keep = min(len(images), n_images - n_collected)
            all_mus.append(batch_mus[:n_to_keep])
            all_zs.append(batch_zs[:n_to_keep])
            all_indices.append(indices[:n_to_keep])

            n_collected += n_to_keep
            pbar.update(n_to_keep)

    pbar.close()

    mus_arr     = torch.cat(all_mus,     dim=0).numpy()
    zs_arr      = torch.cat(all_zs,      dim=0).numpy()
    indices_arr = torch.cat(all_indices, dim=0).numpy().flatten().round().astype(int)
    example_arr = torch.stack(example_buf).numpy() if example_buf else np.zeros((0, 1, 1))

    return mus_arr, zs_arr, indices_arr, example_arr


def load_and_encode_real(
    model: nn.Module,
    mrc_path: str,
    device: str,
    n_images: int,
    encode_batch_size: int,
    n_example_images: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real images from an MRC stack and encode them.

    Returns:
        embeddings:     [N, D]                   float32 numpy
        example_images: [n_example_images, H, W] float32 numpy
    """
    dataset    = RealImageMRCDataset(mrc_path)
    n_to_use   = min(n_images, len(dataset))
    sample_idx = np.random.choice(len(dataset), n_to_use, replace=False)

    # Example images: first n_example_images from the random sample
    n_ex = min(n_example_images, n_to_use)
    example_images = torch.stack([
        dataset[int(j)] for j in sample_idx[:n_ex]
    ]).numpy()

    all_mus = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_to_use, encode_batch_size), desc="Encoding real images"):
            batch_idx = sample_idx[i:i + encode_batch_size]
            batch = torch.stack([dataset[int(j)] for j in batch_idx]).to(device)
            mu, log_var, z = model.forward_vib(batch)
            all_mus.append(mu.cpu())

    return torch.cat(all_mus, dim=0).numpy(), example_images


# ============================================================================
# PLOTTING
# ============================================================================

def make_validation_figure(
    synth_umap:     np.ndarray,
    synth_indices:  np.ndarray,
    combined_umap:  Optional[np.ndarray],
    n_synthetic:    int,
    output_path:    str,
    synth_examples: Optional[np.ndarray] = None,
    real_examples:  Optional[np.ndarray] = None,
):
    """
    2×2 figure:
        [0,0] example synthetic images   [0,1] example real images
        [1,0] UMAP synthetic only        [1,1] UMAP synthetic + real
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 12),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    cmap     = plt.get_cmap("viridis")
    conf_min = int(synth_indices.min())
    conf_max = int(synth_indices.max())
    norm     = plt.Normalize(vmin=conf_min, vmax=conf_max)

    # ---- Row 0, Col 0: example synthetic images ----
    ax = axes[0, 0]
    if synth_examples is not None and len(synth_examples) > 0:
        ax.imshow(_tile_images(synth_examples, n_cols=4), cmap="gray", interpolation="nearest")
        ax.set_title(f"Example synthetic images  (n={len(synth_examples)})", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No synthetic examples",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Example synthetic images", fontsize=12)
    ax.axis("off")

    # ---- Row 0, Col 1: example real images ----
    ax = axes[0, 1]
    if real_examples is not None and len(real_examples) > 0:
        ax.imshow(_tile_images(real_examples, n_cols=4), cmap="gray", interpolation="nearest")
        ax.set_title(f"Example real images  (n={len(real_examples)})", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No real data provided",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Example real images", fontsize=12)
    ax.axis("off")

    # ---- Row 1, Col 0: UMAP synthetic only ----
    ax = axes[1, 0]
    sc = ax.scatter(
        synth_umap[:, 0], synth_umap[:, 1],
        c=synth_indices, s=10, cmap=cmap, norm=norm,
        alpha=0.6, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Conformation index", fontsize=11)
    ax.set_title("Synthetic images\n(colour = conformation index)", fontsize=12)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)

    # ---- Row 1, Col 1: UMAP synthetic + real ----
    ax = axes[1, 1]
    if combined_umap is not None:
        synth_coords = combined_umap[:n_synthetic]
        real_coords  = combined_umap[n_synthetic:]

        sc2 = ax.scatter(
            synth_coords[:, 0], synth_coords[:, 1],
            c=synth_indices, s=10, cmap=cmap, norm=norm,
            alpha=0.5, linewidths=0,
            label=f"Synthetic  (n={n_synthetic:,})", zorder=1,
        )
        ax.scatter(
            real_coords[:, 0], real_coords[:, 1],
            s=10, c="darkorange", alpha=0.7, linewidths=0,
            label=f"Real  (n={len(real_coords):,})", zorder=2,
        )
        cbar2 = fig.colorbar(sc2, ax=ax, pad=0.02)
        cbar2.set_label("Conformation index", fontsize=11)
        ax.set_title(
            "Synthetic vs real images\n(colour = conformation index  |  orange = real)",
            fontsize=12,
        )
    else:
        sc2 = ax.scatter(
            synth_umap[:, 0], synth_umap[:, 1],
            c=synth_indices, s=10, cmap=cmap, norm=norm,
            alpha=0.5, linewidths=0,
            label=f"Synthetic  (n={len(synth_umap):,})",
        )
        cbar2 = fig.colorbar(sc2, ax=ax, pad=0.02)
        cbar2.set_label("Conformation index", fontsize=11)
        ax.set_title("Synthetic images\n(no real data provided)", fontsize=12)

    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Figure saved: {output_path}")
    plt.close(fig)


def make_distribution_figure(
    synth_mu:    np.ndarray,
    synth_z:     np.ndarray,
    real_mu:     Optional[np.ndarray],
    output_path: str,
):
    """
    Mosaic of N plots showing the distribution of synth mu, synth z, and real mu
    for each dimension of the image embedding space.
    """
    n_dim  = synth_mu.shape[1]
    n_cols = min(n_dim, 4)
    n_rows = math.ceil(n_dim / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    all_data = [synth_mu, synth_z]
    if real_mu is not None:
        all_data.append(real_mu)

    global_min = min(d.min() for d in all_data)
    global_max = max(d.max() for d in all_data)
    margin     = (global_max - global_min) * 0.05 if global_max != global_min else 1.0
    x_min      = global_min - margin
    x_max      = global_max + margin

    bins = np.linspace(x_min, x_max, 50)

    color_synth_mu      = "#80B1D3"  # pastel sky blue
    color_synth_mu_edge = "#4F81A3"
    color_synth_z       = "#8DD3C7"  # pastel mint green
    color_synth_z_edge  = "#5B9E93"
    color_real_mu       = "#FDB462"  # pastel orange
    color_real_mu_edge  = "#C88332"

    for i in range(n_dim):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        ax.hist(
            synth_mu[:, i], bins=bins, density=True, alpha=0.4,
            color=color_synth_mu, label=r"Synthetic $\mu$ (deterministic)",
        )
        ax.hist(
            synth_mu[:, i], bins=bins, density=True, histtype="step",
            color=color_synth_mu_edge, linewidth=1.2,
        )

        ax.hist(
            synth_z[:, i], bins=bins, density=True, alpha=0.4,
            color=color_synth_z, label=r"Synthetic $z$ (stochastic)",
        )
        ax.hist(
            synth_z[:, i], bins=bins, density=True, histtype="step",
            color=color_synth_z_edge, linewidth=1.2,
        )

        if real_mu is not None:
            ax.hist(
                real_mu[:, i], bins=bins, density=True, alpha=0.4,
                color=color_real_mu, label=r"Real $\mu$",
            )
            ax.hist(
                real_mu[:, i], bins=bins, density=True, histtype="step",
                color=color_real_mu_edge, linewidth=1.2,
            )

        ax.set_title(f"Dimension {i}", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_dim, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, fontsize=11, frameon=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Distribution figure saved: {output_path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def validate_embedding(
    image_config_path:     str,
    encoder_weights_path:  str,
    embedding_name:        str = "SPATIAL_CRYO",
    embedding_dim:         int = 16,
    device:                str = "cuda",
    n_synthetic:           int = 2000,
    real_data_mrc_path:    Optional[str] = None,
    n_real:                int = 2000,
    simulation_batch_size: int = 512,
    encode_batch_size:     int = 256,
    umap_n_neighbors:      int = 15,
    umap_min_dist:         float = 0.1,
    output_path:           str = "embedding_validation.png",
    n_example_images:      int = 8,
):
    print("\n" + "=" * 70)
    print("EMBEDDING VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Config and conformational models
    # ------------------------------------------------------------------
    image_config = json.load(open(image_config_path))
    image_size   = image_config["N_PIXELS"]

    print("\nLoading conformational models...")
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(device).float()

    n_conformations = len(models)
    print(f"  Conformations: {n_conformations}")
    print(f"  Image size:    {image_size}×{image_size}")

    simulation_param = create_simulation_param(image_config, models, device=device)

    # ------------------------------------------------------------------
    # Load encoder
    # ------------------------------------------------------------------
    print(f"\nLoading encoder: {embedding_name}  (dim={embedding_dim})...")
    ckpt  = torch.load(encoder_weights_path, map_location=device)
    model = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"✅ Encoder loaded — embedding: z = encoder(d)  (mu_head inside)")

    # ------------------------------------------------------------------
    # Generate and encode synthetic images
    # ------------------------------------------------------------------
    print(f"\nGenerating {n_synthetic:,} synthetic images...")
    synth_mus, synth_zs, synth_indices, synth_examples = generate_and_encode_synthetic(
        model, image_config, models, device,
        n_synthetic, simulation_batch_size, encode_batch_size,
        simulation_param, n_example_images,
    )
    synth_embeddings = synth_mus

    print(f"  Embedding shape:    {synth_embeddings.shape}")
    print(f"  Conformation range: [{synth_indices.min()}, {synth_indices.max()}]")

    # ------------------------------------------------------------------
    # Load and encode real images (optional)
    # ------------------------------------------------------------------
    real_embeddings = None
    real_examples   = None
    if real_data_mrc_path is not None:
        print(f"\nLoading real images from: {real_data_mrc_path}")
        real_embeddings, real_examples = load_and_encode_real(
            model, real_data_mrc_path, device, n_real,
            encode_batch_size, n_example_images,
        )
        print(f"  Real embedding shape: {real_embeddings.shape}")

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------
    print(f"\nFitting UMAP on synthetic embeddings (n={len(synth_embeddings):,})...")
    reducer_synth = umap_lib.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=42,
    )
    synth_umap = reducer_synth.fit_transform(synth_embeddings)
    print("✅ UMAP (synthetic) done")

    combined_umap = None
    if real_embeddings is not None:
        combined = np.concatenate([synth_embeddings, real_embeddings], axis=0)
        print(f"\nFitting UMAP on combined embeddings (n={len(combined):,})...")
        reducer_combined = umap_lib.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            n_components=2,
            random_state=42,
        )
        combined_umap = reducer_combined.fit_transform(combined)
        print("✅ UMAP (combined) done")

    # ------------------------------------------------------------------
    # Figure 1: UMAP Validation Figure
    # ------------------------------------------------------------------
    print("\nGenerating figure...")
    make_validation_figure(
        synth_umap=synth_umap,
        synth_indices=synth_indices,
        combined_umap=combined_umap,
        n_synthetic=len(synth_embeddings),
        output_path=output_path,
        synth_examples=synth_examples,
        real_examples=real_examples,
    )

    # ------------------------------------------------------------------
    # Figure 2: Embedding Distributions Figure
    # ------------------------------------------------------------------
    p = Path(output_path)
    dist_output_path = str(p.parent / f"{p.stem}_distributions{p.suffix}")
    print("\nGenerating distribution figure...")
    make_distribution_figure(
        synth_mu=synth_mus,
        synth_z=synth_zs,
        real_mu=real_embeddings,
        output_path=dist_output_path,
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate pretrained image encoder with UMAP visualisation"
    )

    parser.add_argument("--image_config",          required=True,                          help="Path to image config JSON")
    parser.add_argument("--encoder_weights",       required=True,                          help="Path to pretrained encoder weights (.pt)")
    parser.add_argument("--embedding",             default="SPATIAL_CRYO",                 help="Embedding architecture name")
    parser.add_argument("--embedding_dim",         type=int,   default=16,                 help="Encoder output dimension")
    parser.add_argument("--device",                default="cuda",                         help="Compute device (cuda / cpu)")
    parser.add_argument("--n_synthetic",           type=int,   default=2000,               help="Number of synthetic images to generate")
    parser.add_argument("--real_data_mrc",         default=None,                           help="Path to real MRC image stack (optional)")
    parser.add_argument("--n_real",                type=int,   default=2000,               help="Number of real images to use")
    parser.add_argument("--simulation_batch_size", type=int,   default=512,                help="Images generated per simulator call")
    parser.add_argument("--encode_batch_size",     type=int,   default=256,                help="Mini-batch size for encoder forward pass")
    parser.add_argument("--umap_n_neighbors",      type=int,   default=15,                 help="UMAP n_neighbors")
    parser.add_argument("--umap_min_dist",         type=float, default=0.1,                help="UMAP min_dist")
    parser.add_argument("--output",                default="embedding_validation.png",     help="Output figure path")
    parser.add_argument("--n_example_images",      type=int,   default=8,                  help="Number of example images to show per panel (displayed as 2×4 grid)")

    args = parser.parse_args()

    validate_embedding(
        image_config_path     = args.image_config,
        encoder_weights_path  = args.encoder_weights,
        embedding_name        = args.embedding,
        embedding_dim         = args.embedding_dim,
        device                = args.device,
        n_synthetic           = args.n_synthetic,
        real_data_mrc_path    = args.real_data_mrc,
        n_real                = args.n_real,
        simulation_batch_size = args.simulation_batch_size,
        encode_batch_size     = args.encode_batch_size,
        umap_n_neighbors      = args.umap_n_neighbors,
        umap_min_dist         = args.umap_min_dist,
        output_path           = args.output,
        n_example_images      = args.n_example_images,
    )
