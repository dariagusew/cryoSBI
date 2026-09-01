#!/usr/bin/env python3
"""
detect_particle_polarity.py

Decide whether a cryo-EM particle stack (.mrc/.mrcs, shape [N, Y, X]) has
    "white on black"  -> protein at the box centre is BRIGHTER than background
                         (positive contrast, density = high values)
or
    "black on white"  -> protein at the box centre is DARKER than background
                         (negative contrast, as in raw defocus-contrast images)

* The file is memory-mapped (mrcfile.mmap), only one batch at a time is read.
* All image maths is done on the GPU with torch.
* Seven independent criteria are computed per image, each votes with a sign,
  votes are aggregated over the (sub)stack.

Optionally writes a sign-flipped copy so the output is always "white on black".

Requires: torch, numpy, mrcfile
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

try:
    import mrcfile
except ImportError:  # pragma: no cover
    sys.exit("Please `pip install mrcfile`")


# --------------------------------------------------------------------------- #
#  GPU helpers
# --------------------------------------------------------------------------- #
def gaussian_kernel1d(sigma: float, device, truncate: float = 3.0) -> torch.Tensor:
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, k1d: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur. x: (B,1,H,W)."""
    r = (k1d.numel() - 1) // 2
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k1d.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, k1d.view(1, 1, -1, 1))
    return x


def build_masks(h, w, device, r_particle=0.35, r_bg_in=0.50, r_bg_out=1.00):
    """Radii are given in units of the inscribed-circle radius min(h,w)/2."""
    yy = torch.arange(h, device=device, dtype=torch.float32)
    xx = torch.arange(w, device=device, dtype=torch.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = min(h, w) / 2.0
    r = torch.sqrt((yy[:, None] - cy) ** 2 + (xx[None, :] - cx) ** 2) / rad

    masks = {
        "centre": (r <= r_particle),
        "bg": (r >= r_bg_in) & (r <= r_bg_out),
        "inner": (r <= r_bg_out),          # everything inside the inscribed circle
        "r": r,
    }
    if masks["centre"].sum() < 16 or masks["bg"].sum() < 16:
        raise ValueError("Mask radii leave too few pixels; adjust --particle-radius/--bg-inner.")
    return masks


# --------------------------------------------------------------------------- #
#  The seven criteria
# --------------------------------------------------------------------------- #
CRITERIA = [
    "1_centre_minus_bg_mean",
    "2_centre_minus_bg_median",
    "3_skewness",
    "4_extreme_asymmetry",
    "5_radial_slope",
    "6_extremum_position",
    "7_signed_mass_concentration",
]
# relative weights of the criteria in the final vote
WEIGHTS = torch.tensor([1.5, 1.5, 1.0, 1.0, 1.5, 0.75, 1.0])


@torch.inference_mode()
def compute_scores(batch: torch.Tensor, masks: dict, k1d: torch.Tensor) -> torch.Tensor:
    """
    batch : (B, H, W) float32 on GPU
    returns (B, n_criteria) signed scores.  score > 0  =>  centre is BRIGHTER
    """
    B, H, W = batch.shape
    x = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- per-image normalisation (sign preserving) -------------------------
    mu = x.mean(dim=(1, 2), keepdim=True)
    sd = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    x = (x - mu) / sd

    # ---- low-pass filter: cryo-EM SNR per pixel is << 1 --------------------
    xb = gaussian_blur(x.unsqueeze(1), k1d).squeeze(1)

    ctr, bg, inner, r = masks["centre"], masks["bg"], masks["inner"], masks["r"]
    f = xb.reshape(B, -1)
    ctr_f, bg_f, inner_f = ctr.reshape(-1), bg.reshape(-1), inner.reshape(-1)

    v_ctr = f[:, ctr_f]                       # (B, Nc)
    v_bg = f[:, bg_f]                         # (B, Nb)
    v_in = f[:, inner_f]                      # (B, Ni)

    # re-reference everything to the background statistics
    bg_mu = v_bg.mean(dim=1, keepdim=True)
    bg_sd = v_bg.std(dim=1, keepdim=True).clamp_min(1e-8)
    v_ctr = (v_ctr - bg_mu) / bg_sd
    v_bg_n = (v_bg - bg_mu) / bg_sd
    v_in = (v_in - bg_mu) / bg_sd

    scores = []

    # (1) mean of central disc minus mean of background annulus
    scores.append(v_ctr.mean(dim=1))

    # (2) robust version: median centre - median background
    scores.append(v_ctr.median(dim=1).values - v_bg_n.median(dim=1).values)

    # (3) skewness of the (masked) blurred image: the particle creates a tail
    m = v_in - v_in.mean(dim=1, keepdim=True)
    m2 = (m ** 2).mean(dim=1).clamp_min(1e-12)
    m3 = (m ** 3).mean(dim=1)
    scores.append(m3 / m2.pow(1.5))

    # (4) extreme-value asymmetry inside the particle disc:
    #     how far the brightest peak sits above bg vs. the darkest dip below it
    hi = torch.quantile(v_ctr, 0.999, dim=1)
    lo = torch.quantile(v_ctr, 0.001, dim=1)
    scores.append(hi + lo)                    # bg-referenced, so median ~ 0

    # (5) radial slope: correlation between radius and intensity (negated)
    rr = r.reshape(-1)[inner_f]
    rr = rr - rr.mean()
    rr_n = rr / rr.norm().clamp_min(1e-8)
    vv = v_in - v_in.mean(dim=1, keepdim=True)
    vv_n = vv / vv.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scores.append(-(vv_n * rr_n[None, :]).sum(dim=1))

    # (6) which global extremum lies closer to the box centre?
    big = torch.where(inner, xb, torch.full_like(xb, -1e9)).reshape(B, -1)
    small = torch.where(inner, xb, torch.full_like(xb, 1e9)).reshape(B, -1)
    imax = big.argmax(dim=1)
    imin = small.argmin(dim=1)
    rflat = r.reshape(-1)
    scores.append(rflat[imin] - rflat[imax])  # >0 : bright peak is more central

    # (7) signed mass concentration: fraction of positive vs negative
    #     deviation mass that falls inside the central disc
    pos = torch.clamp(f, min=0.0)
    neg = torch.clamp(-f, min=0.0)
    pos_c = (pos * ctr_f).sum(1) / (pos * inner_f).sum(1).clamp_min(1e-8)
    neg_c = (neg * ctr_f).sum(1) / (neg * inner_f).sum(1).clamp_min(1e-8)
    scores.append(pos_c - neg_c)

    return torch.stack(scores, dim=1)         # (B, 7)


# --------------------------------------------------------------------------- #
#  Streaming I/O
# --------------------------------------------------------------------------- #
def iter_batches(mm, indices, batch_size):
    """Yield (idx, float32 C-contiguous ndarray) reading only what is needed."""
    for s in range(0, len(indices), batch_size):
        idx = indices[s: s + batch_size]
        if idx[-1] - idx[0] == len(idx) - 1:          # contiguous -> fast slice
            arr = mm[idx[0]: idx[-1] + 1]
        else:
            arr = mm[idx]                              # fancy index on memmap
        yield idx, np.ascontiguousarray(arr, dtype=np.float32)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input .mrc / .mrcs particle stack")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-images", type=int, default=2000,
                   help="evenly-spaced subset used for the decision (0 = all)")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="low-pass Gaussian sigma in px (0 = auto = boxsize/64)")
    p.add_argument("--particle-radius", type=float, default=0.35,
                   help="central disc radius, fraction of inscribed radius")
    p.add_argument("--bg-inner", type=float, default=0.50)
    p.add_argument("--bg-outer", type=float, default=1.00)
    p.add_argument("--per-image-csv", default=None,
                   help="write per-image scores/votes to this CSV")
    p.add_argument("--write-white-on-black", default=None, metavar="OUT.mrcs",
                   help="stream a copy of the whole stack, sign-flipped if needed")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or "cpu" in args.device else "cpu")
    if device.type == "cuda":
        torch.cuda.init()
        print(f"[i] GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("[!] CUDA not available -> running on CPU")

    # ---------------- memory-mapped open ----------------------------------- #
    mrc = mrcfile.mmap(args.input, mode="r", permissive=True)
    data = mrc.data
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        sys.exit(f"Expected a 3D stack [n,y,x], got shape {data.shape}")
    n_img, H, W = data.shape
    print(f"[i] stack: {n_img} images of {H}x{W}  (dtype {data.dtype}, mmapped)")

    # ---------------- subset selection ------------------------------------- #
    if args.max_images and 0 < args.max_images < n_img:
        indices = np.linspace(0, n_img - 1, args.max_images).astype(np.int64)
        indices = np.unique(indices)
    else:
        indices = np.arange(n_img, dtype=np.int64)
    print(f"[i] analysing {len(indices)} images")

    sigma = args.sigma if args.sigma > 0 else max(1.5, min(H, W) / 64.0)
    k1d = gaussian_kernel1d(sigma, device)
    masks = build_masks(H, W, device,
                        args.particle_radius, args.bg_inner, args.bg_outer)
    w = WEIGHTS.to(device)
    print(f"[i] low-pass sigma = {sigma:.2f} px, "
          f"centre r<{args.particle_radius}, background {args.bg_inner}-{args.bg_outer}")

    # ---------------- streaming loop --------------------------------------- #
    all_scores = []
    for idx, arr in iter_batches(data, indices, args.batch_size):
        t = torch.from_numpy(arr)
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        all_scores.append(compute_scores(t, masks, k1d).float().cpu())
        del t
    scores = torch.cat(all_scores, dim=0)                 # (M, 7)
    del all_scores

    # ---------------- aggregation ------------------------------------------ #
    votes = torch.sign(scores)                            # per image, per criterion
    per_crit = votes.mean(dim=0)                          # in [-1, 1]
    weighted_img = (votes * w.cpu()).sum(dim=1)           # per-image consensus
    img_vote = torch.sign(weighted_img)
    overall = float((per_crit * w.cpu()).sum() / w.cpu().sum())
    frac_images = float((img_vote > 0).float().mean())

    print("\n  criterion                        mean score      vote (+1=white/black)")
    print("  " + "-" * 68)
    for i, name in enumerate(CRITERIA):
        print(f"  {name:<30s} {scores[:, i].mean():>+10.4f}      {per_crit[i]:>+6.3f}")
    print("  " + "-" * 68)
    print(f"  weighted consensus over criteria : {overall:+.3f}")
    print(f"  images voting 'white on black'   : {100*frac_images:.1f} %")

    if abs(overall) < 0.15 or 0.35 < frac_images < 0.65:
        verdict, conf = "AMBIGUOUS", "low"
    elif overall > 0:
        verdict, conf = "WHITE_ON_BLACK", "high" if abs(overall) > 0.6 else "medium"
    else:
        verdict, conf = "BLACK_ON_WHITE", "high" if abs(overall) > 0.6 else "medium"

    print(f"\n  ==> VERDICT: {verdict}  (confidence: {conf})")
    if verdict == "WHITE_ON_BLACK":
        print("      Protein density is BRIGHT (high values) on a dark background.")
    elif verdict == "BLACK_ON_WHITE":
        print("      Protein density is DARK (low values) on a bright background;")
        print("      multiply by -1 to obtain the usual 'white on black' convention.")

    # ---------------- optional per-image CSV -------------------------------- #
    if args.per_image_csv:
        arr = np.concatenate([indices[:, None].astype(np.float32),
                              scores.numpy(),
                              img_vote.numpy()[:, None]], axis=1)
        np.savetxt(args.per_image_csv, arr, delimiter=",", fmt="%.6g",
                   header="index," + ",".join(CRITERIA) + ",image_vote", comments="")
        print(f"[i] per-image scores written to {args.per_image_csv}")

    # ---------------- optional sign-corrected copy -------------------------- #
    if args.write_white_on_black:
        flip = -1.0 if verdict == "BLACK_ON_WHITE" else 1.0
        print(f"[i] writing {args.write_white_on_black} (scale {flip:+.0f}) ...")
        with mrcfile.new_mmap(args.write_white_on_black,
                              shape=(n_img, H, W), mrc_mode=2,
                              overwrite=True) as out:
            for s in range(0, n_img, args.batch_size):
                e = min(s + args.batch_size, n_img)
                chunk = np.asarray(data[s:e], dtype=np.float32)
                out.data[s:e] = flip * chunk
            try:
                out.voxel_size = mrc.voxel_size
            except Exception:
                pass
            out.update_header_from_data()
            out.update_header_stats()
        print("[i] done.")

    mrc.close()


if __name__ == "__main__":
    main()
