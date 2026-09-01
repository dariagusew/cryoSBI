#!/usr/bin/env python3
"""
train_noise_gan.py

Trains a fully-convolutional GAN on cryo-EM ice-noise patches stored as
MRC image stacks (produced by extract_patches_to_mrc.py).

Architecture
------------
Generator     : spatial noise map z (B,1,H,W) → patch (B,1,H,W)
                dilated ResBlocks, no strided ops, translation-equivariant
Discriminator : 3-layer PatchGAN, spectral-normalised on every conv

Generator losses
----------------
  L_adv  :  non-saturating      E[ softplus(-D(G(z))) ]
  L_nps  :  log-scale spectral  L1( log NPS_batch(fake), log NPS_batch(real) )
  L_fm   :  feature matching    mean MSE across D intermediate layers

  L_G = L_adv + λ_nps · L_nps + λ_fm · L_fm

Discriminator loss
------------------
  L_D = softplus(-D(real)) + softplus(D(fake))

Early stopping metric
---------------------
  Validation log-NPS L1 loss — physically meaningful, GAN-agnostic.
"""

import argparse
import contextlib
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import mrcfile
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mrcfile")

try:
    import matplotlib.pyplot as plt
    PLOTTING = True
except ImportError:
    PLOTTING = False

# ---------------------------------------------------------------------------
# Hardcoded hyper-parameters that are never worth tuning for this task
# ---------------------------------------------------------------------------
_ADAM_BETA1  = 0.0      # GAN standard — no momentum
_ADAM_BETA2  = 0.999
_EMA_DECAY   = 0.999
_GROUPS      = 8        # GroupNorm groups for both G and D
_VAL_BATCHES = 50       # mini-batches used per validation pass
_PATIENCE    = 20       # early-stopping patience (epochs)
_MAX_CACHE   = 10_000   # maximum patches kept in each worker's LRU cache


# ===========================================================================
# 1.  DATASET
# ===========================================================================

class MRCStackDataset(Dataset):

    def __init__(self, mrc_path: Path, augment: bool = False):
        self.augment = augment

        with mrcfile.open(str(mrc_path), permissive=True) as mrc:
            raw = mrc.data
            if raw.ndim == 2:
                raw = raw[np.newaxis]           # (1, H, W)

        # Normalise entire stack at once (vectorised, fast)
        data  = torch.from_numpy(np.ascontiguousarray(raw.astype(np.float32)))
        mean  = data.mean(dim=(-2, -1), keepdim=True)
        std   = data.std (dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        data  = (data - mean) / std             # (N, H, W)  normalised

        self._data = data
        self._data.share_memory_()              # zero-copy across all workers

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._data[idx].unsqueeze(0).clone()   # (1, H, W)
        if self.augment:
            if torch.rand(()) > 0.5: img = img.flip(-1)
            if torch.rand(()) > 0.5: img = img.flip(-2)
            k = int(torch.randint(4, ()))
            if k: img = torch.rot90(img, k, (-2, -1))
        return img

# ===========================================================================
# 2.  GENERATOR
# ===========================================================================

class ResBlockG(nn.Module):
    """
    Pre-activation residual block with optional dilation.

    padding = dilation preserves spatial size for any dilation value.
    The second conv is always dilation=1 (standard dilated-ResNet convention).
    """

    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_GROUPS, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_GROUPS, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Generator(nn.Module):
    """
    Fully-convolutional generator: z (B,1,H,W) → patch (B,1,H,W).

    No downsampling / upsampling → translation-equivariant.

    Dilation schedule [1,2,4,8,4,2,1,1] gives a receptive field of ~120 px,
    covering the full 128×128 patch while keeping the architecture local.

    Output is linear — no Tanh / sigmoid because patches are z-scored,
    pixel values are not bounded to [-1, 1].
    """

    _DILATIONS: List[int] = [1, 2, 4, 8, 4, 2, 1, 1]

    def __init__(self, base_channels: int = 64, n_blocks: int = 8):
        super().__init__()
        ch = base_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, ch, 7, padding=3),
            nn.GroupNorm(_GROUPS, ch),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResBlockG(ch, dilation=self._DILATIONS[i % len(self._DILATIONS)])
            for i in range(n_blocks)
        ])
        self.output_conv = nn.Conv2d(ch, 1, 7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(z)
        for rb in self.res_blocks:
            h = rb(h)
        return self.output_conv(h)


# ===========================================================================
# 3.  PATCHGAN DISCRIMINATOR
# ===========================================================================

def _sn(module: nn.Module) -> nn.Module:
    """Apply spectral normalisation to constrain the Lipschitz constant of D."""
    return spectral_norm(module)


class PatchGANDiscriminator(nn.Module):
    """
    3-layer PatchGAN with spectral normalisation on every Conv2d.

    For 128×128 input:
      Spatial output : 14×14 logit grid
      Receptive field: ≈ 70×70 px per output score

    No normalisation on layer 0 (standard PatchGAN practice — normalising
    a single-channel input has trivial and harmful statistics).
    GroupNorm in layers 1–3 (batch-size independent, compatible with SN).

    Returns
    -------
    logits : (B, 1, 14, 14)  raw real/fake scores (no sigmoid)
    feats  : list of 4 intermediate feature maps for feature matching
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        ch = base_channels

        self.layer0 = nn.Sequential(               # 128 → 64
            _sn(nn.Conv2d(1,    ch,   4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer1 = nn.Sequential(               # 64 → 32
            _sn(nn.Conv2d(ch,   ch*2, 4, stride=2, padding=1)),
            nn.GroupNorm(_GROUPS, ch*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(               # 32 → 16
            _sn(nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1)),
            nn.GroupNorm(_GROUPS, ch*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(               # 16 → 15  (stride=1)
            _sn(nn.Conv2d(ch*4, ch*8, 4, stride=1, padding=1)),
            nn.GroupNorm(_GROUPS, ch*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final = _sn(nn.Conv2d(ch*8, 1, 4, stride=1, padding=1))  # 15→14

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        f0  = self.layer0(x)
        f1  = self.layer1(f0)
        f2  = self.layer2(f1)
        f3  = self.layer3(f2)
        return self.final(f3), [f0, f1, f2, f3]


# ===========================================================================
# 4.  PATCH NORMALISATION
# ===========================================================================

def _normalise(x: torch.Tensor) -> torch.Tensor:
    """
    Per-image zero mean, unit std — mirrors MRCStackDataset normalisation.

    Applied to every batch of generated patches immediately after G(z),
    before the tensor is passed to any loss function or to D.  This ensures
    that loss comparisons between real and fake patches are made on a common
    scale and that the NPS loss measures spectral *shape* rather than
    absolute power.

    Parameters
    ----------
    x : (B, 1, H, W)  raw generator output

    Returns
    -------
    (B, 1, H, W)  normalised, gradients preserved through mean/std ops
    """
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std  = x.std (dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return (x - mean) / std


# ===========================================================================
# 5.  LOSSES
# ===========================================================================

def _log_nps(imgs: torch.Tensor) -> torch.Tensor:
    """
    Log of the batch-averaged 2-D power spectrum (NPS).

    DC is removed by zero-meaning each patch before the FFT.
    Because _normalise() is always applied before this function, the
    per-image mean subtraction here is a no-op in practice but is kept
    for correctness in case the function is called standalone.

    Parameters
    ----------
    imgs : (B, 1, H, W)

    Returns
    -------
    log_power : (1, H, W)
    """
    imgs  = imgs - imgs.mean(dim=(-2, -1), keepdim=True)
    power = torch.abs(torch.fft.fft2(imgs)) ** 2          # (B, 1, H, W)
    return torch.log(power.mean(dim=0) + 1e-8)            # (1, H, W)


def nps_spectral_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    L1 loss between log NPS of generated and real patch batches.

    Log scale weights all spatial frequencies equally, including the
    scientifically critical 3.7 Å water ring.  L2 on linear scale would
    be dominated by the DC region alone.

    Both fake and real are assumed to be already normalised (mean=0, std=1).
    """
    return F.l1_loss(_log_nps(fake), _log_nps(real).detach())


def feature_matching_loss(
    fake_feats: List[torch.Tensor],
    real_feats: List[torch.Tensor],
) -> torch.Tensor:
    """
    Mean MSE between D-intermediate features of fake and real patches.

    Real features are detached — gradients flow only through the fake path.
    Prevents mode collapse without any gradient penalty on D.
    """
    loss = sum(F.mse_loss(ff, fr.detach())
               for ff, fr in zip(fake_feats, real_feats))
    return loss / len(fake_feats)


def g_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating generator loss  –E[log σ(D(G(z)))].

    Gradient stays large even when D is confident, unlike the original
    minimax log(1-D(G(z))) which saturates immediately.
    """
    return F.softplus(-fake_logits).mean()


def d_adversarial_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Non-saturating discriminator loss.

    softplus(-real) → push D(real) → +∞
    softplus( fake) → push D(fake) → -∞
    """
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


# ===========================================================================
# 6.  EMA
# ===========================================================================

class EMA:
    """
    Float32 exponential moving average of generator parameters.

    EMA weights are used for validation; raw weights receive gradient updates.
    The `applied` context manager swaps weights in and out cleanly.
    """

    def __init__(self, model: nn.Module):
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().float().clone()
            for k, v in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.named_parameters():
            self.shadow[k].mul_(_EMA_DECAY).add_(
                v.detach().float(), alpha=1.0 - _EMA_DECAY
            )

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

    def state_dict(self) -> Dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict, device: torch.device) -> None:
        self.shadow = {k: v.to(device) for k, v in sd.items()}


# ===========================================================================
# 7.  UTILITIES
# ===========================================================================

def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_loader(
    ds: Dataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
        prefetch_factor    = 4 if num_workers > 0 else None,
        drop_last          = shuffle,
    )


@torch.no_grad()
def _validate(
    G:          nn.Module,
    ema:        EMA,
    val_loader: DataLoader,
    device:     torch.device,
) -> float:
    """
    Log-NPS L1 loss on _VAL_BATCHES mini-batches using EMA weights.

    Generated patches are normalised before comparison, matching the
    treatment in the training loop.
    """
    losses: List[float] = []
    with ema.applied(G):
        G.eval()
        for i, real in enumerate(val_loader):
            if i >= _VAL_BATCHES:
                break
            real = real.to(device, non_blocking=True)
            fake = _normalise(G(torch.randn_like(real)))
            losses.append(nps_spectral_loss(fake, real).item())
    return float(np.mean(losses))


def _save_checkpoint(
    path:     Path,
    epoch:    int,
    G:        nn.Module,
    D:        nn.Module,
    ema:      EMA,
    opt_g:    torch.optim.Optimizer,
    opt_d:    torch.optim.Optimizer,
    best_val: float,
    history:  Dict,
    gen_cfg:  Dict,
    disc_cfg: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":         epoch,
            "generator":     G.state_dict(),
            "discriminator": D.state_dict(),
            "ema":           ema.state_dict(),
            "opt_g":         opt_g.state_dict(),
            "opt_d":         opt_d.state_dict(),
            "best_val_nps":  best_val,
            "history":       history,
            "gen_cfg":       gen_cfg,
            "disc_cfg":      disc_cfg,
        },
        path,
    )


def _plot_history(history: Dict, path: Path) -> None:
    if not PLOTTING:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(history["d_loss"], color="steelblue")
    axes[0].set(title="Discriminator Loss", xlabel="Epoch", ylabel="L_D")
    axes[0].grid(True, ls="--", alpha=0.4)

    for key, lbl, col in [
        ("g_adv",   "L_adv",     "tomato"),
        ("g_nps",   "L_nps",     "seagreen"),
        ("g_fm",    "L_fm",      "goldenrod"),
        ("g_total", "L_G total", "mediumpurple"),
    ]:
        axes[1].plot(history[key], label=lbl, color=col)
    axes[1].set(title="Generator Loss Components", xlabel="Epoch")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.4)

    axes[2].plot(history["val_nps"], color="crimson", label="val log-NPS L1")
    axes[2].set(title="Validation NPS Loss  (early stopping)", xlabel="Epoch")
    axes[2].legend()
    axes[2].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"   Loss curves → {path.resolve()}")


# ===========================================================================
# 8.  MAIN
# ===========================================================================

def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Train a GAN ice-noise generator from MRC image stacks.\n"
            "Losses: non-saturating L_adv + log-NPS spectral + feature matching.\n"
            "Discriminator: PatchGAN + spectral normalisation, 1:1 D/G ratio.\n"
            "Generated patches are normalised (mean=0, std=1) before every loss."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    g = ap.add_argument_group("I/O")
    g.add_argument("--train_mrc", required=True, type=Path,
                   help="Training MRC image stack.")
    g.add_argument("--val_mrc",   required=True, type=Path,
                   help="Validation MRC image stack.")
    g.add_argument("--save_path", default="noise_gan.pt", type=Path,
                   help="Output checkpoint path (default: noise_gan.pt).")

    g = ap.add_argument_group("Generator architecture")
    g.add_argument("--gen_channels", type=int, default=64,
                   help="Base channel width (default: 64).")
    g.add_argument("--gen_blocks",   type=int, default=8,
                   help="Number of dilated ResBlocks (default: 8).")

    g = ap.add_argument_group("Discriminator architecture")
    g.add_argument("--disc_channels", type=int, default=64,
                   help="PatchGAN base channels (default: 64).")

    g = ap.add_argument_group("Loss weights")
    g.add_argument("--lambda_nps", type=float, default=10.0,
                   help="Weight for log-NPS spectral loss (default: 10.0).")
    g.add_argument("--lambda_fm",  type=float, default=1.0,
                   help="Weight for feature matching loss  (default:  1.0).")

    g = ap.add_argument_group("Training")
    g.add_argument("--max_epochs",  type=int,   default=300,
                   help="Maximum training epochs (default: 300).")
    g.add_argument("--steps_per_epoch", type=int, default=None,
                   help="Limit training to this many mini-batches per epoch. "
                        "Processes the full dataset if not set.")
    g.add_argument("--batch_size",  type=int,   default=64,
                   help="Mini-batch size (default: 64).")
    g.add_argument("--lr_g",        type=float, default=1e-4,
                   help="Generator learning rate     (default: 1e-4).")
    g.add_argument("--lr_d",        type=float, default=4e-4,
                   help="Discriminator learning rate (default: 4e-4).")
    g.add_argument("--num_workers", type=int,   default=4,
                   help="DataLoader worker processes (default: 4).")
    g.add_argument("--augment",     action="store_true",
                   help="8-fold dihedral augmentation on training patches.")
    g.add_argument("--seed",        type=int,   default=42)
    g.add_argument("--device",      type=str,   default=None)

    args = ap.parse_args()

    # ── reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    gen_cfg  = dict(base_channels=args.gen_channels, n_blocks=args.gen_blocks)
    disc_cfg = dict(base_channels=args.disc_channels)

    # ── peek at image size without loading pixels ──────────────────────────
    with mrcfile.mmap(str(args.train_mrc), mode="r", permissive=True) as mrc:
        img_h, img_w = int(mrc.header.ny), int(mrc.header.nx)

    # ── banner ─────────────────────────────────────────────────────────────
    SEP = "=" * 70
    patch_mb = img_h * img_w * 4 / 1e6
    print(f"\n{SEP}")
    print("NOISE GAN TRAINING")
    print(SEP)
    print(f"  Train stack    : {args.train_mrc.name}")
    print(f"  Val stack      : {args.val_mrc.name}")
    print(f"  Image size     : {img_h}×{img_w}")
    print(f"  Generator      : base_ch={args.gen_channels}, "
          f"blocks={args.gen_blocks}")
    print(f"  Discriminator  : base_ch={args.disc_channels}  "
          f"[PatchGAN + SpectralNorm]")
    print(f"  Losses         : L_adv + {args.lambda_nps}·L_nps "
          f"+ {args.lambda_fm}·L_fm")
    print(f"  Fake normalise : mean=0 std=1 applied after every G(z) call")
    print(f"  D/G ratio      : 1:1  |  patience : {_PATIENCE}")
    print(f"  Max epochs     : {args.max_epochs}  |  "
          f"batch_size : {args.batch_size}")
    if args.steps_per_epoch:
        print(f"  Steps/epoch    : {args.steps_per_epoch}")
    print(f"  lr_G / lr_D    : {args.lr_g} / {args.lr_d}")
    print(f"  augment        : {args.augment}")
    print(f"  Cache          : {_MAX_CACHE:,} patches/worker  "
          f"({_MAX_CACHE * patch_mb * args.num_workers:.0f} MB total "
          f"across {args.num_workers} workers)")
    print(f"  Save path      : {args.save_path}")
    print(SEP)

    # ── datasets & loaders ─────────────────────────────────────────────────
    print("\nBuilding datasets...")
    train_ds = MRCStackDataset(args.train_mrc, augment=args.augment)
    val_ds   = MRCStackDataset(args.val_mrc,   augment=False)
    print(f"  Training   : {len(train_ds):,} patches")
    print(f"  Validation : {len(val_ds):,} patches")

    train_dl = _make_loader(train_ds, args.batch_size,
                             shuffle=True,  num_workers=args.num_workers)
    val_dl   = _make_loader(val_ds,   args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # ── models ─────────────────────────────────────────────────────────────
    print(f"\nBuilding Generator    : {gen_cfg}")
    G = Generator(**gen_cfg).to(device)
    print(f"  Parameters : {_n_params(G):,}")

    print(f"Building Discriminator: {disc_cfg}")
    D = PatchGANDiscriminator(**disc_cfg).to(device)
    print(f"  Parameters : {_n_params(D):,}")

    # ── EMA / optimisers / schedulers ──────────────────────────────────────
    ema = EMA(G)

    opt_G = torch.optim.Adam(
        G.parameters(), lr=args.lr_g, betas=(_ADAM_BETA1, _ADAM_BETA2)
    )
    opt_D = torch.optim.Adam(
        D.parameters(), lr=args.lr_d, betas=(_ADAM_BETA1, _ADAM_BETA2)
    )
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=args.max_epochs, eta_min=args.lr_g / 20
    )
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=args.max_epochs, eta_min=args.lr_d / 20
    )

    # ── training loop ──────────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        k: [] for k in ["d_loss", "g_adv", "g_nps", "g_fm", "g_total", "val_nps"]
    }
    best_val    = float("inf")
    best_epoch  = 1
    patience_ct = 0
    t_start     = time.time()

    print(f"\nTraining on {device}  |  augment={args.augment}\n")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        G.train(); D.train()

        acc: Dict[str, float] = dict(d=0.0, adv=0.0, nps=0.0, fm=0.0, g=0.0)
        n_steps = 0

        pbar_total = args.steps_per_epoch or len(train_dl)
        pbar = tqdm(train_dl,
                    desc=f"  Ep {epoch:>4}/{args.max_epochs}",
                    total=pbar_total,
                    leave=False, ncols=80)

        for real in pbar:
            if args.steps_per_epoch is not None and n_steps >= args.steps_per_epoch:
                break

            real = real.to(device, non_blocking=True)

            # ── DISCRIMINATOR STEP ──────────────────────────────────────
            # Normalise fake patches so D always compares on equal scale.
            with torch.no_grad():
                fake_d = _normalise(G(torch.randn_like(real)))
            real_logits,   _ = D(real)
            fake_logits_d, _ = D(fake_d)
            loss_D = d_adversarial_loss(real_logits, fake_logits_d)

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # ── GENERATOR STEP ──────────────────────────────────────────
            # Fresh z — normalise immediately; all three losses operate on
            # the same normalised tensor.
            fake_g = _normalise(G(torch.randn_like(real)))
            fake_logits_g, fake_feats = D(fake_g)
            with torch.no_grad():
                _, real_feats = D(real)

            loss_adv = g_adversarial_loss(fake_logits_g)
            loss_nps = nps_spectral_loss(fake_g, real)
            loss_fm  = feature_matching_loss(fake_feats, real_feats)
            loss_G   = (loss_adv
                        + args.lambda_nps * loss_nps
                        + args.lambda_fm  * loss_fm)

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            ema.update(G)

            acc["d"]   += loss_D.item()
            acc["adv"] += loss_adv.item()
            acc["nps"] += loss_nps.item()
            acc["fm"]  += loss_fm.item()
            acc["g"]   += loss_G.item()
            n_steps    += 1

        pbar.close()
        # ── epoch statistics ───────────────────────────────────────────
        sched_G.step(); sched_D.step()
        n  = max(n_steps, 1)
        ep = {k: v / n for k, v in acc.items()}

        val_nps  = _validate(G, ema, val_dl, device)
        improved = val_nps < best_val
        ep_sec   = time.time() - t0
        marker   = "★" if improved else f"patience {patience_ct+1}/{_PATIENCE}"

        print(
            f"  Ep {epoch:>4}/{args.max_epochs}"
            f"  D={ep['d']:.4f}"
            f"  G=[adv={ep['adv']:.3f}"
            f" nps={ep['nps']:.3f}"
            f" fm={ep['fm']:.3f}"
            f" tot={ep['g']:.3f}]"
            f"  val_nps={val_nps:.4f}"
            f"  {ep_sec:.1f}s  {marker}"
        )

        history["d_loss"].append(ep["d"])
        history["g_adv"].append(ep["adv"])
        history["g_nps"].append(ep["nps"])
        history["g_fm"].append(ep["fm"])
        history["g_total"].append(ep["g"])
        history["val_nps"].append(val_nps)

        if improved:
            best_val    = val_nps
            best_epoch  = epoch
            patience_ct = 0
            _save_checkpoint(
                args.save_path, epoch,
                G, D, ema, opt_G, opt_D,
                best_val, history, gen_cfg, disc_cfg,
            )
        else:
            patience_ct += 1
            if patience_ct >= _PATIENCE:
                print(
                    f"\n  Early stopping: val NPS loss did not improve for "
                    f"{_PATIENCE} consecutive epochs."
                )
                break

    # ── final summary ──────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    print(f"\n{SEP}")
    print("✓  Training complete.")
    print(f"   Best val NPS loss : {best_val:.6f}  (epoch {best_epoch})")
    print(f"   Total time        : {elapsed:.1f} min")
    print(f"   Checkpoint        : {args.save_path.resolve()}")
    print(SEP)

    _plot_history(history, args.save_path.with_suffix(".losses.png"))


if __name__ == "__main__":
    main()
