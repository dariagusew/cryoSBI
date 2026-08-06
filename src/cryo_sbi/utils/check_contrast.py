#!/usr/bin/env python3
"""
check_contrast.py

Check whether the particle in an MRC cryo-EM image stack is darker or brighter
than the background, i.e. whether the contrast appears inverted.

Assumes stack shape: [N_images, H, W]
"""
import argparse
import sys
import numpy as np
import torch
import mrcfile


def parse_args():
    p = argparse.ArgumentParser(
        description="Check if the particle signal in an MRC stack is darker "
                    "or brighter than the background (GPU accelerated)."
    )
    p.add_argument("mrc", help="Input MRC stack file")
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA if available")
    p.add_argument("--batch", type=int, default=64,
                   help="Number of images per GPU batch")
    p.add_argument("--center-radius", type=float, default=0.25,
                   help="Radius of central particle mask as fraction of half-box")
    p.add_argument("--bg-inner", type=float, default=0.35,
                   help="Inner radius of background annulus (fraction of half-box)")
    p.add_argument("--bg-outer", type=float, default=0.45,
                   help="Outer radius of background annulus (fraction of half-box)")
    p.add_argument("--radial-bins", type=int, default=24,
                   help="Number of radial bins for the radial profile")
    p.add_argument("--output-avg", default=None,
                   help="Optional output MRC file for the stack average image")
    return p.parse_args()


def make_masks(H, W, center_frac, bg_inner_frac, bg_outer_frac, device):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(H, W) / 2.0

    center_mask = r <= center_frac * rmax
    bg_mask = (r >= bg_inner_frac * rmax) & (r <= bg_outer_frac * rmax)

    return {
        "center_2d": center_mask,
        "bg_2d": bg_mask,
        "center_flat": center_mask.reshape(-1).float(),
        "bg_flat": bg_mask.reshape(-1).float(),
        "n_center": int(center_mask.sum().item()),
        "n_bg": int(bg_mask.sum().item()),
    }


def make_radial_grid(H, W, n_bins, device):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(H, W) / 2.0

    # bin index for every pixel
    bin_idx = (r / rmax * n_bins).long()
    bin_idx = bin_idx.clamp(0, n_bins - 1).reshape(-1)

    counts_per_image = torch.bincount(
        bin_idx, minlength=n_bins
    ).float().to(device)

    centers = (torch.arange(n_bins, dtype=torch.float32, device=device) + 0.5) / n_bins * rmax

    return {
        "bin_idx": bin_idx,
        "counts_per_image": counts_per_image,
        "centers": centers,
        "rmax": rmax,
    }


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")

    # Memory-map the MRC file so we don't load the whole stack into RAM
    with mrcfile.mmap(args.mrc, mode="r", permissive=True) as mrc:
        data = mrc.data
        header = mrc.header
        voxel_size = mrc.voxel_size

    print(f"MRC shape : {data.shape}")
    print(f"MRC dtype : {data.dtype}")
    print(f"MRC mode  : {header.mode}")

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim != 3:
        raise ValueError("Expected a 2D image or 3D image stack [N,H,W]")

    N, H, W = data.shape
    rmax = min(H, W) / 2.0
    print(f"Images    : {N}")
    print(f"Box size  : {H} x {W}")
    print(f"Center mask radius : {args.center_radius * rmax:.1f} px")
    print(f"Background annulus : {args.bg_inner * rmax:.1f}-{args.bg_outer * rmax:.1f} px\n")

    masks = make_masks(H, W, args.center_radius, args.bg_inner,
                       args.bg_outer, device)
    radial = make_radial_grid(H, W, args.radial_bins, device)

    # Per-image statistics
    center_means = np.empty(N, dtype=np.float64)
    bg_means = np.empty(N, dtype=np.float64)
    center_stds = np.empty(N, dtype=np.float64)
    bg_stds = np.empty(N, dtype=np.float64)
    center_medians = np.empty(N, dtype=np.float64)
    bg_medians = np.empty(N, dtype=np.float64)
    t_stats = np.empty(N, dtype=np.float64)

    # Accumulators for stack-level radial profile and average image
    radial_sums = torch.zeros(args.radial_bins, device=device, dtype=torch.float64)
    radial_counts = torch.zeros(args.radial_bins, device=device, dtype=torch.float64)
    avg_img = torch.zeros(H, W, device=device, dtype=torch.float64)

    center_flat = masks["center_flat"]
    bg_flat = masks["bg_flat"]
    center_2d = masks["center_2d"]
    bg_2d = masks["bg_2d"]
    n_center = masks["n_center"]
    n_bg = masks["n_bg"]
    bin_idx_flat = radial["bin_idx"]

    print("Processing stack...")
    for i in range(0, N, args.batch):
        j = min(i + args.batch, N)

        # Read chunk, cast to float32, send to GPU/CPU
        chunk_np = data[i:j].astype(np.float32, copy=False)
        chunk = torch.from_numpy(chunk_np).to(device)  # B,H,W
        B = chunk.shape[0]
        flat = chunk.view(B, -1)

        # Means
        csum = (flat * center_flat[None, :]).sum(dim=1)
        bsum = (flat * bg_flat[None, :]).sum(dim=1)
        cmean = csum / n_center
        bmean = bsum / n_bg

        # Stds (vectorized, subtract per-image mean)
        cstd = torch.sqrt(
            ((flat - cmean[:, None]) ** 2 * center_flat[None, :]).sum(dim=1) / n_center
        )
        bstd = torch.sqrt(
            ((flat - bmean[:, None]) ** 2 * bg_flat[None, :]).sum(dim=1) / n_bg
        )

        # Medians using boolean indexing (B x n_pixels)
        center_pixels = chunk[:, center_2d]   # B x n_center
        bg_pixels = chunk[:, bg_2d]           # B x n_bg
        cmed = center_pixels.median(dim=1).values
        bmed = bg_pixels.median(dim=1).values

        # Welch-style t-statistic
        se = torch.sqrt(cstd ** 2 / n_center + bstd ** 2 / n_bg)
        t = (cmean - bmean) / se.clamp(min=1e-12)

        # Store per-image stats
        center_means[i:j] = cmean.cpu().numpy()
        bg_means[i:j] = bmean.cpu().numpy()
        center_stds[i:j] = cstd.cpu().numpy()
        bg_stds[i:j] = bstd.cpu().numpy()
        center_medians[i:j] = cmed.cpu().numpy()
        bg_medians[i:j] = bmed.cpu().numpy()
        t_stats[i:j] = t.cpu().numpy()

        # Accumulate radial profile
        bin_idx_b = bin_idx_flat.unsqueeze(0).expand(B, -1)  # B x L
        rad_chunk = torch.zeros(B, args.radial_bins, device=device, dtype=torch.float32)
        rad_chunk.scatter_add_(1, bin_idx_b, flat)
        radial_sums += rad_chunk.sum(dim=0)
        radial_counts += radial["counts_per_image"] * B

        # Accumulate average image
        avg_img += chunk.sum(dim=0).double()

        print(f"  processed {j}/{N} images", end="\r")

    print(f"\nFinished processing {N} images.\n")

    # Finalize stack-level quantities
    avg_img /= N
    radial_profile = radial_sums / radial_counts.clamp(min=1.0)

    diffs = center_means - bg_means
    med_diffs = center_medians - bg_medians

    # ------------------------------------------------------------------
    # CRITERIA
    # ------------------------------------------------------------------

    def verdict_from_diff(diff, name):
        if abs(diff) < 1e-9:
            return f"{name}: NEARLY IDENTICAL (no measurable contrast)"
        elif diff < 0:
            return f"{name}: CENTER IS DARKER than background"
        else:
            return f"{name}: CENTER IS BRIGHTER than background (likely inverted)"

    print("=" * 60)
    print("CONTRAST CHECK RESULTS")
    print("=" * 60)

    # 1. Mean comparison
    print("\n--- Criterion 1: Per-image mean ---")
    print(f"  Avg center mean  : {center_means.mean():.6f}")
    print(f"  Avg background   : {bg_means.mean():.6f}")
    print(f"  Mean difference  : {diffs.mean():.6f}")
    print(f"  Mean ratio       : {center_means.mean() / bg_means.mean():.6f}")
    print(f"  -> {verdict_from_diff(diffs.mean(), 'Mean')}")

    # 2. Median comparison
    print("\n--- Criterion 2: Per-image median (robust) ---")
    print(f"  Avg center median : {center_medians.mean():.6f}")
    print(f"  Avg background    : {bg_medians.mean():.6f}")
    print(f"  Median difference : {med_diffs.mean():.6f}")
    print(f"  -> {verdict_from_diff(med_diffs.mean(), 'Median')}")

    # 3. Standardized contrast (z-score)
    print("\n--- Criterion 3: Standardized contrast (center-bg)/bg_std ---")
    z_scores = diffs / np.maximum(bg_stds, 1e-12)
    print(f"  Mean z-score : {z_scores.mean():.3f}")
    print(f"  Std z-score  : {z_scores.std():.3f}")
    print(f"  -> {verdict_from_diff(z_scores.mean(), 'Z-score')}")

    # 4. Welch t-statistic
    print("\n--- Criterion 4: Welch-style t-statistic ---")
    print(f"  Mean t-statistic : {t_stats.mean():.3f}")
    print(f"  Std t-statistic  : {t_stats.std():.3f}")
    print(f"  -> {verdict_from_diff(t_stats.mean(), 'T-statistic')}")

    # 5. Per-image majority vote
    print("\n--- Criterion 5: Per-image majority vote ---")
    frac_center_darker = (diffs < 0).mean()
    frac_center_brighter = (diffs > 0).mean()
    print(f"  Fraction center < background : {frac_center_darker:.3%}")
    print(f"  Fraction center > background : {frac_center_brighter:.3%}")
    if frac_center_darker > 0.5:
        print("  -> MAJORITY: center is darker (normal dark particle)")
    elif frac_center_brighter > 0.5:
        print("  -> MAJORITY: center is brighter (contrast likely inverted)")
    else:
        print("  -> AMBIGUOUS: no clear trend")

    # 6. Radial profile
    print("\n--- Criterion 6: Radial intensity profile ---")
    inner_bins = max(1, int(args.center_radius * args.radial_bins))
    outer_start = args.radial_bins - max(1, args.radial_bins // 3)

    inner_mean = radial_profile[:inner_bins].mean().item()
    outer_mean = radial_profile[outer_start:].mean().item()
    print(f"  Inner region (r<= {args.center_radius*100:.0f}%) mean : {inner_mean:.6f}")
    print(f"  Outer region mean           : {outer_mean:.6f}")
    print(f"  Inner - outer               : {inner_mean - outer_mean:.6f}")
    print(f"  -> {verdict_from_diff(inner_mean - outer_mean, 'Radial profile')}")

    # 7. Stack-average image
    print("\n--- Criterion 7: Stack average image ---")
    avg_center = avg_img[center_2d].mean().item()
    avg_bg = avg_img[bg_2d].mean().item()
    print(f"  Avg image center mean : {avg_center:.6f}")
    print(f"  Avg image bg mean     : {avg_bg:.6f}")
    print(f"  Difference            : {avg_center - avg_bg:.6f}")
    print(f"  -> {verdict_from_diff(avg_center - avg_bg, 'Average image')}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    signs = [
        np.sign(diffs.mean()),
        np.sign(med_diffs.mean()),
        np.sign(z_scores.mean()),
        np.sign(t_stats.mean()),
        np.sign(frac_center_darker - 0.5),
        np.sign(inner_mean - outer_mean),
        np.sign(avg_center - avg_bg),
    ]
    # votes: negative = darker, positive = brighter
    darker_votes = sum(1 for s in signs if s < 0)
    brighter_votes = sum(1 for s in signs if s > 0)
    print(f"Criteria voting 'center darker'    : {darker_votes}/7")
    print(f"Criteria voting 'center brighter'  : {brighter_votes}/7")

    if darker_votes > brighter_votes:
        print("\n>>> CONCLUSION: Protein appears DARKER than background.")
        print("    This is the expected appearance for normal cryo-EM.")
    elif brighter_votes > darker_votes:
        print("\n>>> CONCLUSION: Protein appears BRIGHTER than background.")
        print("    The stack may have inverted contrast (white particles).")
    else:
        print("\n>>> CONCLUSION: Inconclusive; contrast is weak or ambiguous.")

    # Optional: write the average image
    if args.output_avg:
        avg_np = avg_img.cpu().numpy().astype(np.float32)
        with mrcfile.new(args.output_avg, overwrite=True) as mrc_out:
            mrc_out.set_data(avg_np)
            mrc_out.voxel_size = voxel_size
        print(f"\nWrote stack-average image to {args.output_avg}")


if __name__ == "__main__":
    main()
