#!/usr/bin/env python3
"""
extract_patches_to_mrc.py

Extracts square image patches from a directory of cryo-EM micrographs and
writes them into two MRC image-stack files:

  • Training   (~90 %, configurable) – for training a diffusion model.
  • Validation (~10 %, configurable) – for evaluating the model.

Classes and helpers (MrcReader, extract_and_rescale_patches) are taken
verbatim from estimate_nps_from_ice_v2.py so that both scripts stay
consistent.

Two splitting strategies are supported:
  micrograph (default)
      Whole micrographs are assigned to either train or val *before* any
      patches are extracted.  This is the statistically correct approach:
      no patch from the same physical area can appear in both sets.

  patch
      All patches from all micrographs are pooled, globally shuffled, and
      then split.  Simpler, but patches from the same micrograph may appear
      in both sets.

Typical usage
-------------
python extract_patches_to_mrc.py \\
    --in_dir    /data/micrographs       \\
    --out_train /data/train_patches.mrcs \\
    --out_val   /data/val_patches.mrcs   \\
    --box_size  128  --pixel_size 1.5    \\
    --num_patches 200                    \\
    --normalization zscore               \\
    --split_by micrograph
"""

import gc
import random
import sys
import warnings
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import mrcfile
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mrcfile")


# ============================================================================
# 1.  MRC FILE READER  –  shared with estimate_nps_from_ice_v2.py
# ============================================================================
class MrcReader:
    """
    Minimal reader that opens an MRC/MRCS file, extracts the pixel size from
    the header, and returns the image data as a 2-D float32 numpy array.
    """

    def __init__(self, mrc_path: Path):
        self.mrc_path = mrc_path
        self.apix: Optional[float] = None

    def read_mrc(self) -> np.ndarray:
        with mrcfile.open(self.mrc_path, permissive=True) as mrc:
            try:
                apix = float(mrc.voxel_size.x)
                self.apix = apix if apix > 0 else None
            except (ValueError, TypeError):
                self.apix = None

            data = mrc.data
            if data.ndim == 2:
                return data.astype(np.float32)
            if data.ndim == 3 and data.shape[0] == 1:
                return data[0].astype(np.float32)
            raise ValueError(
                f"Unsupported MRC shape {data.shape} in '{self.mrc_path.name}'. "
                "Expected a single 2-D image."
            )


# ============================================================================
# 2.  PATCH EXTRACTION  –  shared with estimate_nps_from_ice_v2.py
# ============================================================================
def extract_and_rescale_patches(
    micrograph_data: np.ndarray,
    num_patches: int,
    box_size_out: int,
    apix_in: float,
    apix_out: float,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Randomly sample *num_patches* windows from *micrograph_data*, bicubic-
    resize them from the native pixel size (*apix_in*) to the target pixel
    size (*apix_out*), and return a list of (box_size_out × box_size_out)
    float32 tensors placed on *device*.
    """
    mic_h, mic_w = micrograph_data.shape
    box_size_in = int(round(box_size_out / (apix_in / apix_out)))

    if box_size_in > mic_h or box_size_in > mic_w:
        tqdm.write(
            f"  ⚠️  Required patch size ({box_size_in} px) exceeds micrograph "
            f"dimensions ({mic_h}×{mic_w} px) – skipping this micrograph."
        )
        return []

    y_coords = np.random.randint(0, mic_h - box_size_in + 1, size=num_patches)
    x_coords = np.random.randint(0, mic_w - box_size_in + 1, size=num_patches)

    patches: List[torch.Tensor] = []
    for y, x in zip(y_coords, x_coords):
        patch_t = torch.from_numpy(
            micrograph_data[y : y + box_size_in, x : x + box_size_in]
        ).to(device)

        if box_size_in != box_size_out:
            patch_t = F.interpolate(
                patch_t.unsqueeze(0).unsqueeze(0),
                size=(box_size_out, box_size_out),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        patches.append(patch_t)

    return patches


# ============================================================================
# 3.  PER-PATCH NORMALIZATION  (new)
# ============================================================================
_NORM_HELP = (
    "Per-patch normalization applied before saving (default: zscore).\n"
    "  zscore  – zero mean, unit variance     [recommended for diffusion models]\n"
    "  mean    – subtract per-patch mean only\n"
    "  minmax  – rescale pixel values to [0, 1]\n"
    "  none    – keep raw pixel values"
)


def normalize_patch(patch: np.ndarray, method: str) -> np.ndarray:
    """Apply *method* normalization to a single 2-D patch in-place (returns a copy)."""
    if method == "zscore":
        std = patch.std()
        return (patch - patch.mean()) / (std if std > 1e-8 else 1.0)
    if method == "mean":
        return patch - patch.mean()
    if method == "minmax":
        lo, hi = patch.min(), patch.max()
        return (patch - lo) / (hi - lo) if (hi - lo) > 1e-8 else np.zeros_like(patch)
    if method == "none":
        return patch.copy()
    raise ValueError(f"Unknown normalization method: {method!r}")


# ============================================================================
# 4.  MRC STACK WRITER  (new)
# ============================================================================
def write_mrc_stack(
    patches: np.ndarray,
    output_path: Path,
    pixel_size: float,
    label: str,
) -> None:
    """
    Write an (N × H × W) float32 array as an MRC image stack.

    Parameters
    ----------
    patches     : 3-D float32 array, shape (N, H, W).
    output_path : Destination .mrc / .mrcs path.
    pixel_size  : Pixel size (Å/px) stored in the MRC header.
    label       : Human-readable name used in log messages.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w = patches.shape

    print(
        f"\n  [{label}]  {n} patches  ({h}×{w} px)"
        f"  |  min={patches.min():.4f}  max={patches.max():.4f}"
        f"  |  mean={patches.mean():.4f}  std={patches.std():.4f}"
    )
    print(f"    → {output_path}")

    with mrcfile.new(str(output_path), overwrite=True) as mrc:
        mrc.set_data(patches)   # mrcfile auto-selects mode=2 (float32)
        mrc.voxel_size = pixel_size

    print(f"    ✓ Saved to {output_path.resolve()}")


# ============================================================================
# 5.  EXTRACTION HELPER  (new)
# ============================================================================
def _extract_from_filelist(
    mic_files: List[Path],
    args: argparse.Namespace,
    device: torch.device,
    tqdm_label: str,
) -> List[np.ndarray]:
    """
    Iterate over *mic_files*, extract patches from each, normalise them, and
    return a flat list of (H × W) float32 numpy arrays.
    """
    patches: List[np.ndarray] = []

    for mic_path in tqdm(mic_files, desc=f"  {tqdm_label}", ncols=80, unit="mic"):
        try:
            reader = MrcReader(mic_path)
            mic_data = reader.read_mrc()

            # Resolve pixel size
            apix_mic: Optional[float] = (
                args.mic_pixel_size if args.mic_pixel_size is not None else reader.apix
            )
            if apix_mic is None:
                tqdm.write(
                    f"    ⚠️  {mic_path.name}: pixel size unknown "
                    "(no MRC header value and --mic_pixel_size not set) → skipping."
                )
                continue

            tensors = extract_and_rescale_patches(
                mic_data,
                args.num_patches,
                args.box_size,
                apix_mic,
                args.pixel_size,
                device,
            )
            del mic_data
            gc.collect()

            for t in tensors:
                p = normalize_patch(t.cpu().numpy(), args.normalization)
                patches.append(p.astype(np.float32))

            del tensors
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            tqdm.write(f"    ❌  {mic_path.name}: {exc}")

    return patches


# ============================================================================
# 6.  MAIN
# ============================================================================
def main() -> None:

    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Extract cryo-EM patches from a micrograph directory and save them "
            "as train / val MRC stacks for diffusion model training."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument(
        "--in_dir", required=True, type=Path,
        help="Directory containing micrograph files (.mrc, .mrcs).",
    )
    io_grp.add_argument(
        "--out_train", required=True, type=Path,
        help="Output path for the training MRC image stack.",
    )
    io_grp.add_argument(
        "--out_val", required=True, type=Path,
        help="Output path for the validation MRC image stack.",
    )

    geo_grp = parser.add_argument_group("Patch geometry")
    geo_grp.add_argument(
        "--box_size", required=True, type=int,
        help="Side length (px) of each extracted square patch.",
    )
    geo_grp.add_argument(
        "--pixel_size", required=True, type=float,
        help="Target pixel size of the output patches (Å/px).",
    )
    geo_grp.add_argument(
        "--mic_pixel_size", type=float, default=None,
        help=(
            "Pixel size of the input micrographs (Å/px).\n"
            "Overrides the value stored in MRC headers."
        ),
    )

    ext_grp = parser.add_argument_group("Extraction")
    ext_grp.add_argument(
        "--num_patches", type=int, default=200,
        help="Number of patches to extract per micrograph (default: 200).",
    )
    ext_grp.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of data reserved for validation (default: 0.10 = 10%%).",
    )
    ext_grp.add_argument(
        "--split_by", choices=["micrograph", "patch"], default="patch",
        help=(
            "Splitting strategy (default: micrograph).\n"
            "  micrograph – assign whole micrographs to train or val\n"
            "               before any extraction (no data leakage).\n"
            "  patch      – pool all patches, shuffle globally, then split."
        ),
    )

    pre_grp = parser.add_argument_group("Pre-processing")
    pre_grp.add_argument(
        "--normalization", default="zscore",
        choices=["zscore", "mean", "minmax", "none"],
        help=_NORM_HELP,
    )

    rt_grp = parser.add_argument_group("Runtime")
    rt_grp.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    rt_grp.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda' or 'cpu' (default: auto-detect).",
    )

    args = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if not (0.0 < args.val_fraction < 1.0):
        sys.exit("❌  --val_fraction must be strictly between 0 and 1.")

    # ── Setup ─────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Banner ────────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(sep)
    print("  extract_patches_to_mrc.py")
    print(sep)
    print(f"  Device        : {device}")
    print(f"  Box size      : {args.box_size} px")
    print(f"  Pixel size    : {args.pixel_size} Å/px")
    print(f"  Patches/mic   : {args.num_patches}")
    print(f"  Normalization : {args.normalization}")
    print(f"  Split by      : {args.split_by}")
    print(f"  Val fraction  : {args.val_fraction * 100:.0f} %")
    print(f"  Seed          : {args.seed}")
    print(sep)

    # ── Discover micrographs ──────────────────────────────────────────────────
    mic_files: List[Path] = sorted(
        list(args.in_dir.glob("*.mrc")) + list(args.in_dir.glob("*.mrcs"))
    )
    if not mic_files:
        sys.exit(f"❌  No .mrc / .mrcs files found in: {args.in_dir}")
    print(f"\nFound {len(mic_files)} micrograph(s) in '{args.in_dir}'.\n")

    # ── Rough memory estimate ─────────────────────────────────────────────────
    est_total_patches = len(mic_files) * args.num_patches
    est_gb = est_total_patches * (args.box_size ** 2) * 4 / 1e9   # float32
    print(
        f"  Estimated maximum patches : ~{est_total_patches:,}\n"
        f"  Estimated peak RAM usage  : ~{est_gb:.2f} GB  "
        f"(reduce --num_patches if insufficient)\n"
    )

    # =========================================================================
    # STRATEGY A – split by micrograph
    # =========================================================================
    if args.split_by == "micrograph":

        print("[1/3] Assigning micrographs to train / val sets …")

        shuffled_mics = mic_files.copy()
        random.shuffle(shuffled_mics)

        n_val_mics   = max(1, int(round(len(shuffled_mics) * args.val_fraction)))
        val_mics     = shuffled_mics[:n_val_mics]
        train_mics   = shuffled_mics[n_val_mics:]

        print(f"       Train : {len(train_mics)} micrograph(s)")
        print(f"       Val   : {len(val_mics)} micrograph(s)\n")

        print("[2/3] Extracting patches …")
        train_patches = _extract_from_filelist(train_mics, args, device, "train")
        val_patches   = _extract_from_filelist(val_mics,   args, device, "val  ")

    # =========================================================================
    # STRATEGY B – split by patch
    # =========================================================================
    else:   # "patch"

        print("[1/3] Extracting patches from all micrographs …")
        all_patches = _extract_from_filelist(mic_files, args, device, "all")

        if not all_patches:
            sys.exit("❌  No patches were extracted. Aborting.")

        print(f"\n[2/3] Shuffling {len(all_patches):,} patches and splitting …")
        random.shuffle(all_patches)

        n_val         = max(1, int(round(len(all_patches) * args.val_fraction)))
        val_patches   = all_patches[:n_val]
        train_patches = all_patches[n_val:]
        del all_patches
        gc.collect()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not train_patches:
        sys.exit("❌  Training set is empty. Check --val_fraction and your data.")
    if not val_patches:
        sys.exit("❌  Validation set is empty. Check --val_fraction and your data.")

    n_train = len(train_patches)
    n_val   = len(val_patches)
    actual_gb = (n_train + n_val) * (args.box_size ** 2) * 4 / 1e9

    print(
        f"\n  Final split: {n_train:,} train | {n_val:,} val "
        f"({n_val / (n_train + n_val) * 100:.1f}% val) "
        f"| {actual_gb:.2f} GB total"
    )

    # ── Stack and save ─────────────────────────────────────────────────────────
    print("\n[3/3] Stacking patches and writing MRC stacks …")

    train_arr = np.stack(train_patches, axis=0)   # (N_train, H, W) float32
    del train_patches
    gc.collect()
    write_mrc_stack(train_arr, args.out_train, args.pixel_size, "Training")
    del train_arr
    gc.collect()

    val_arr = np.stack(val_patches, axis=0)        # (N_val,   H, W) float32
    del val_patches
    gc.collect()
    write_mrc_stack(val_arr, args.out_val, args.pixel_size, "Validation")
    del val_arr
    gc.collect()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("✓  Done.")
    print(f"   Training   : {args.out_train.resolve()}  [{n_train:,} patches]")
    print(f"   Validation : {args.out_val.resolve()}  [{n_val:,} patches]")
    print(sep)


if __name__ == "__main__":
    main()
