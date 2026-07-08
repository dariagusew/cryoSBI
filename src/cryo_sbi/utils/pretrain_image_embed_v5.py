# "pretrain_image_embed_v5.py"
"""
pretrain_image_embed_v5.py

3-stage VIB pre-training of image encoder on synthetic cryo-EM data.
Stage 1 uses clean (noiseless) synthetic images.

Stages:
    1. VIB pretrain on clean synthetic images
    2. Train a residual U-Net noise model so that clean synthetic images
       look like real images (adversarial + frozen-encoder content loss).
    3. Fine-tune the Stage-1 encoder with the frozen noise model added
       to the simulator.

The encoder is trained as a stochastic sufficient statistic for the full
parameter vector (X, θ). The encoder's public interface returns a
deterministic embedding mu = encoder(d) used by the flow at inference.
During VIB pretraining only, encoder.forward_vib(d) additionally returns
log_var and a reparameterized sample z; the log_var_head is discarded
afterwards.

Usage:
    python pretrain_image_embed_v5.py \
        --image_config config.json \
        --real_data_mrc real_images.mrc \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --epochs 100
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import (
    cryo_em_simulator,
    create_simulation_param,
)

try:
    import mrcfile
except ImportError:
    mrcfile = None


# ============================================================================
# MODEL
# ============================================================================

class FullParamPredictor(nn.Module):
    """
    Predicts all parameters (X, θ) from z.
    Training scaffolding only — discarded after pretraining.
    """
    def __init__(self, embedding_dim: int, n_conformations: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conf_head    = nn.Linear(hidden_dim, n_conformations)
        self.orient_head  = nn.Linear(hidden_dim, 4)
        self.shift_head   = nn.Linear(hidden_dim, 2)
        self.defocus_head = nn.Linear(hidden_dim, 1)
        self.bfactor_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {
            "conf":    self.conf_head(h),
            "orient":  self.orient_head(h),
            "shift":   self.shift_head(h),
            "defocus": self.defocus_head(h).squeeze(-1),
            "bfactor": self.bfactor_head(h).squeeze(-1),
        }


class ImageEmbedPretrainModel(nn.Module):
    def __init__(
        self,
        embedding_name: str,
        embedding_dim: int,
        image_size: int,
        n_conformations: int,
    ):
        super().__init__()
        self.embedding_name = embedding_name
        self.embedding_dim  = embedding_dim
        self.image_size     = image_size

        self.encoder   = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size)
        self.predictor = FullParamPredictor(embedding_dim, n_conformations)

        # Verify the chosen encoder supports the VIB training interface
        if not hasattr(self.encoder, "forward_vib"):
            raise ValueError(
                f"Embedding net '{embedding_name}' must implement forward_vib() "
                f"for VIB pretraining."
            )

        print(f"  Encoder:   {embedding_name}  (D={image_size}, output_dim={embedding_dim})")
        print(f"    mu_head inside; log_var_head inside (training only)")
        print(f"  Predictor: z → (X={n_conformations} classes, orient, shift, defocus, bfactor)")


    def forward(self, x: torch.Tensor):
        mu, log_var, z = self.encoder.forward_vib(x)
        preds = self.predictor(z)
        return mu, log_var, z, preds

# ============================================================================
# NOISE MODEL (Stage 2)
# ============================================================================

class _UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StochasticResidualUNet(torch.nn.Module):
    """
    Stochastic residual U-Net for adding realistic noise/background.
    Samples a latent noise vector internally; optionally accepts an external z
    for reproducible evaluation.

    Input and output are [B, H, W]. Requires image_size divisible by 4.
    """
    def __init__(self, base: int = 32, noise_dim: int = 16):
        super().__init__()
        self.noise_dim = noise_dim

        self.enc1 = _UNetBlock(1, base)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = _UNetBlock(base, base * 2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.bottleneck = _UNetBlock(base * 2, base * 4)

        # Project latent noise into bottleneck feature space
        self.noise_proj = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, base * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(base * 4, base * 4),
        )

        self.up2 = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _UNetBlock(base * 4 + base * 2, base * 2)
        self.up1 = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _UNetBlock(base * 2 + base, base)
        self.outc = torch.nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_x = x
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Sample noise internally if not provided
        if z is None:
            z = torch.randn(
                x.size(0), self.noise_dim,
                device=x.device, dtype=x.dtype
            )

        # Encode
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))

        # Inject noise at bottleneck; broadcasts over spatial dims
        z_feat = self.noise_proj(z).view(x.size(0), -1, 1, 1)
        x3 = x3 + z_feat

        # Decode
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        out = input_x + self.outc(x).squeeze(1)

        # Per-image z-score normalization
        mean = out.mean(dim=(-2, -1), keepdim=True)
        std = out.std(dim=(-2, -1), keepdim=True)
        return (out - mean) / (std + 1e-8)


class PatchDiscriminator(nn.Module):
    """Simple PatchGAN discriminator. Outputs a grid of real/fake logits."""
    def __init__(self, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.net(x)


class SimpleRealImageDataset(Dataset):
    """Minimal real-image MRC loader with per-image z-score normalization."""
    def __init__(self, mrc_path: str, cache_size: int = 10000):
        if mrcfile is None:
            raise ImportError("mrcfile is required. Install with: pip install mrcfile")
        self.mrc_path = mrc_path
        self.cache: dict = {}
        self.cache_order: deque = deque()
        self.cache_size = cache_size

        print(f"  Opening real MRC: {mrc_path}")
        self.mrc_data = mrcfile.open(mrc_path, mode="r").data
        self.n_images = self.mrc_data.shape[0]
        self.image_shape = self.mrc_data.shape[1:]
        print(f"  Real images: {self.n_images:,}, shape: {self.image_shape}")

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.cache:
            return self.cache[idx]
        img = self.mrc_data[idx].astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.popleft()
            del self.cache[oldest]
        self.cache[idx] = torch.from_numpy(img)
        self.cache_order.append(idx)
        return self.cache[idx]


# ============================================================================
# FIXED TARGET NORMALIZER
# ============================================================================

class FixedTargetNormalizer(nn.Module):
    def __init__(self, image_config: dict, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        def _register(key: str, mean, std):
            mean_t = torch.as_tensor(mean, dtype=torch.float32)
            std_t  = torch.as_tensor(std,  dtype=torch.float32)
            self.register_buffer(f"{key}_mean", mean_t)
            self.register_buffer(f"{key}_std",  std_t)

        max_shift = float(image_config["SHIFT"])
        shift_std = max_shift / math.sqrt(3.0)
        _register("shift", [0.0, 0.0], [shift_std, shift_std])

        dmin, dmax = map(float, image_config["DEFOCUS"])
        _register("defocus", (dmin + dmax) / 2.0, (dmax - dmin) / math.sqrt(12.0))

        bmin, bmax = map(float, image_config["B_FACTOR"])
        _register("bfactor", (bmin + bmax) / 2.0, (bmax - bmin) / math.sqrt(12.0))

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, f"{key}_mean").to(x.device)
        std  = getattr(self, f"{key}_std").to(x.device)
        return (x - mean) / (std + self.eps)


# ============================================================================
# VIB LOSS HELPERS
# ============================================================================

def _quaternion_loss(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    q_pred   = F.normalize(q_pred, dim=1)
    q_target = F.normalize(q_target.float(), dim=1)
    d_pos = (q_pred - q_target).pow(2).sum(dim=-1)
    d_neg = (q_pred + q_target).pow(2).sum(dim=-1)
    return torch.minimum(d_pos, d_neg).mean()


def vib_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    beta: float,
    pred_weights: Dict[str, float],
    normalizer: FixedTargetNormalizer,
):
    L_conf = F.cross_entropy(preds["conf"], targets["indices"])
    L_orient = _quaternion_loss(preds["orient"], targets["quaternions"])

    L_shift = F.mse_loss(
        preds["shift"],
        normalizer.normalize("shift", targets["shift"].float()),
    )
    L_defocus = F.mse_loss(
        preds["defocus"],
        normalizer.normalize("defocus", targets["defocus"].float().reshape(-1)),
    )
    L_bfactor = F.mse_loss(
        preds["bfactor"],
        normalizer.normalize("bfactor", targets["b_factor"].float().reshape(-1)),
    )

    total_weight = sum(pred_weights.values())

    L_pred = (
          pred_weights["conf"]    * L_conf
        + pred_weights["orient"]  * L_orient
        + pred_weights["shift"]   * L_shift
        + pred_weights["defocus"] * L_defocus
        + pred_weights["bfactor"] * L_bfactor
    ) / total_weight

    L_kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1).mean()

    return L_pred + beta * L_kl, L_pred, L_kl


# ============================================================================
# UTILITIES
# ============================================================================

def check_embedding_health(embeddings: torch.Tensor, device: str) -> tuple:
    with torch.no_grad():
        emb_std = embeddings.std(dim=0).mean().item()
        if len(embeddings) > 1:
            dists    = torch.cdist(embeddings, embeddings)
            off_diag = dists[~torch.eye(len(embeddings), dtype=bool, device=device)]
            emb_dist = off_diag.mean().item()
        else:
            emb_dist = 0.0
    return emb_std, emb_dist


def count_parameters(model: nn.Module) -> Dict[str, int]:
    encoder_total = sum(p.numel() for p in model.encoder.parameters())
    mu_head_total = sum(p.numel() for p in model.encoder.mu_head.parameters())
    log_var_total = sum(p.numel() for p in model.encoder.log_var_head.parameters())
    trunk_total = encoder_total - mu_head_total - log_var_total

    return {
        "total":        sum(p.numel() for p in model.parameters()),
        "trainable":    sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder":      encoder_total,
        "encoder_trunk": trunk_total,
        "mu_head":      mu_head_total,
        "log_var_head": log_var_total,
        "predictor":    sum(p.numel() for p in model.predictor.parameters()),
    }


def _infinite_dataloader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def _run_vib_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    synthetic_iter,
    synthetic_loader: PriorLoader,
    models: torch.Tensor,
    simulation_param: dict,
    device: str,
    batch_size: int,
    n_batches_per_epoch: int,
    beta: float,
    pred_weights: Dict[str, float],
    normalizer: FixedTargetNormalizer,
    noise_model: Optional[nn.Module] = None,
    use_noiseless_images: bool = False
) -> tuple:
    """Single VIB epoch. If noise_model is provided, applies it to clean images."""
    epoch_loss = 0.0
    epoch_pred_loss = 0.0
    epoch_kl_loss = 0.0
    n_steps = 0
    last_mu: Optional[torch.Tensor] = None

    for _ in range(n_batches_per_epoch):
        try:
            parameters = next(synthetic_iter)
        except StopIteration:
            synthetic_iter = iter(synthetic_loader)
            parameters = next(synthetic_iter)

        indices, quaternions, shift, defocus, b_factor, amp, snr = parameters

        with torch.no_grad():
            # synthetic images out of simulator
            noisy_images, clean_images = cryo_em_simulator(
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

        # clean or noisy
        clean_images = clean_images if use_noiseless_images else noisy_images

        n_full = (len(clean_images) // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            sl = slice(i, i + batch_size)

            with torch.no_grad():
                if noise_model is not None:
                    clean_images[sl] = noise_model(clean_images[sl])

            targets = {
                "indices":     indices[sl].squeeze(-1).round().long().to(device, non_blocking=True),
                "quaternions": quaternions[sl].to(device, non_blocking=True),
                "shift":       shift[sl].to(device,       non_blocking=True),
                "defocus":     defocus[sl].to(device,     non_blocking=True),
                "b_factor":    b_factor[sl].to(device,    non_blocking=True),
            }

            optimizer.zero_grad()
            mu, log_var, z, preds = model(clean_images[sl])
            loss, L_pred, L_kl = vib_loss(
                mu, log_var, preds, targets, beta, pred_weights, normalizer
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss      += loss.item()
            epoch_pred_loss += L_pred.item()
            epoch_kl_loss   += L_kl.item()
            n_steps += 1
            last_mu = mu.detach()

    avg_loss      = epoch_loss      / max(n_steps, 1)
    avg_pred_loss = epoch_pred_loss / max(n_steps, 1)
    avg_kl_loss   = epoch_kl_loss   / max(n_steps, 1)
    return avg_loss, avg_pred_loss, avg_kl_loss, last_mu, synthetic_iter


def _train_noise_model(
    encoder: nn.Module,
    image_prior,
    models: torch.Tensor,
    simulation_param: dict,
    device: str,
    real_data_mrc_path: str,
    noise_epochs: int,
    noise_batch_size: int,
    noise_n_batches_per_epoch: int,
    noise_lr: float,
    lambda_adv: float,
    lambda_content: float,
    check_frequency: int = 5,
    use_noiseless_images: bool = False
) -> nn.Module:
    print("\n" + "=" * 70)
    print("STAGE 2: TRAINING RESIDUAL NOISE MODEL")
    print("=" * 70)

    generator = StochasticResidualUNet(base=32).to(device)
    discriminator = PatchDiscriminator(base=32).to(device)

    print(f"  Generator params:        {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator params:    {sum(p.numel() for p in discriminator.parameters()):,}")

    opt_G = optim.AdamW(generator.parameters(), lr=noise_lr, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_D = optim.AdamW(discriminator.parameters(), lr=noise_lr, betas=(0.5, 0.999), weight_decay=1e-4)

    real_dataset = SimpleRealImageDataset(real_data_mrc_path)
    real_loader = DataLoader(
        real_dataset,
        batch_size=noise_batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    real_iter = _infinite_dataloader(real_loader)

    synthetic_loader = PriorLoader(image_prior, batch_size=noise_batch_size, num_workers=4)
    synthetic_iter = iter(synthetic_loader)

    with tqdm(range(noise_epochs), desc="Stage 2: noise model") as tq:
         for epoch in tq:
            epoch_L_D       = 0.0
            epoch_L_adv     = 0.0
            epoch_L_content = 0.0
            epoch_L_G       = 0.0
            n_steps         = 0
         
            for _ in range(noise_n_batches_per_epoch):
                try:
                    parameters = next(synthetic_iter)
                except StopIteration:
                    synthetic_iter = iter(synthetic_loader)
                    parameters = next(synthetic_iter)
         
                indices, quaternions, shift, defocus, b_factor, amp, snr = parameters
                with torch.no_grad():
                    noisy_images, clean_images = cryo_em_simulator(
                        models,
                        indices.to(device), quaternions.to(device), shift.to(device),
                        defocus.to(device), b_factor.to(device), amp.to(device), snr.to(device),
                        simulation_param,
                        simulation_param["noise"],
                    )

                # clean or noisy
                clean_images = clean_images if use_noiseless_images else noisy_images
         
                real_images = next(real_iter).to(device)
         
                # ---- discriminator step ----
                opt_D.zero_grad()
                with torch.no_grad():
                    fake_images = generator(clean_images)
                d_real = discriminator(real_images)
                d_fake = discriminator(fake_images)
                L_D = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
                L_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                opt_D.step()
         
                # ---- generator step ----
                opt_G.zero_grad()
                fake_images = generator(clean_images)
                d_fake_g = discriminator(fake_images)
                L_adv = F.softplus(-d_fake_g).mean()
         
                with torch.no_grad():
                    z_clean = encoder(clean_images)
                z_fake = encoder(fake_images)
                L_content = F.mse_loss(z_fake, z_clean)
         
                L_G = lambda_adv * L_adv + lambda_content * L_content
                L_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                opt_G.step()
         
                epoch_L_D       += L_D.item()
                epoch_L_adv     += L_adv.item()
                epoch_L_content += L_content.item()
                epoch_L_G       += L_G.item()
                n_steps         += 1
         
            avg_L_D       = epoch_L_D       / max(n_steps, 1)
            avg_L_adv     = epoch_L_adv     / max(n_steps, 1)
            avg_L_content = epoch_L_content / max(n_steps, 1)
            avg_L_G       = epoch_L_G       / max(n_steps, 1)
         
            tq.set_postfix({
                "L_D":       f"{avg_L_D:.4f}",
                "L_G":       f"{avg_L_G:.4f}",
                "L_adv":     f"{avg_L_adv:.4f}",
                "L_content": f"{avg_L_content:.4f}",
            })
         
            if epoch % check_frequency == 0:
                print(f"\n  Stage 2 Epoch {epoch:3d}:")
                print(f"    Discriminator loss:  {avg_L_D:.6f}")
                print(f"    Generator loss:      {avg_L_G:.6f}")
                print(f"    Adversarial loss:    {avg_L_adv:.6f}")
                print(f"    Content loss:        {avg_L_content:.6f}")
                print(f"    D/G balance:         {avg_L_D / max(avg_L_G, 1e-8):.4f}  (healthy ≈ 1.0)")

    return generator


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def pretrain_image_embed(
    image_config_path: str,
    real_data_mrc_path: str,
    resume_from: Optional[str] = None,
    embedding_name: str = "SPATIAL_CRYO",
    device: str = "cuda",
    embedding_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 2e-4,
    simulation_batch_size: int = 1024,
    save_path: str = "pretrained_image_embed.pt",
    check_frequency: int = 5,
    n_batches_per_epoch: int = 100,
    beta: float = 1e-3,
    pred_weights: Optional[Dict[str, float]] = None,
    noise_model_path: str = "noise_model.pt",
    noise_epochs: int = 50,
    noise_batch_size: int = 64,
    noise_lr: float = 2e-4,
    noise_n_batches_per_epoch: int = 100,
    finetune_epochs: int = 50,
    finetune_lr: float = 1e-4,
    lambda_adv: float = 1.0,
    lambda_content: float = 1.0,
    use_noiseless_images: bool = False
):
    print("\n" + "=" * 70)
    print(f"PRETRAINING: {embedding_name}")
    print("Training mode: 3-STAGE VIB ON SYNTHETIC + REAL NOISE MODEL")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 70)

    if use_noiseless_images:
        print("Using NOISELESS synthetic images")
    print("=" * 70)

    if pred_weights is None:
        pred_weights = {
            "conf":    10.0,
            "orient":   2.0,
            "shift":    0.5,
            "defocus":  0.5,
            "bfactor":  0.3,
        }

    print("\nPrediction loss weights:")
    for key, val in pred_weights.items():
        print(f"  {key:10s}: {val:.2f}")

    # ------------------------------------------------------------------
    # Config and conformational models
    # ------------------------------------------------------------------
    with open(image_config_path) as f:
        image_config = json.load(f)
    image_size = image_config["N_PIXELS"]

    print("\nLoading conformational models...")
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(device).float()

    n_conformations = len(models)
    print(f"  Number of conformations: {n_conformations}")
    print(f"  Image size: {image_size}x{image_size}")

    if image_size % 4 != 0:
        print(f"  ⚠️  WARNING: image_size={image_size} is not divisible by 4; StochasticResidualUNet may fail.")

    image_prior      = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    synthetic_loader = PriorLoader(image_prior, batch_size=simulation_batch_size, num_workers=4)
    synthetic_iter   = iter(synthetic_loader)
    simulation_param = create_simulation_param(image_config, models, device=device)

    # ------------------------------------------------------------------
    # Fixed target normalization
    # ------------------------------------------------------------------
    print("Building fixed target normalizer from prior ranges...")
    normalizer = FixedTargetNormalizer(image_config).to(device)

    for key in ("shift", "defocus", "bfactor"):
        mean = getattr(normalizer, f"{key}_mean")
        std  = getattr(normalizer, f"{key}_std")
        print(f"  {key:8s}: mean={mean.tolist()}, std={std.tolist()}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\nBuilding model with {embedding_name}...")
    try:
        model = ImageEmbedPretrainModel(
            embedding_name, embedding_dim, image_size, n_conformations
        ).to(device)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return None, 0.0

    if resume_from:
        print(f"\nLoading checkpoint from: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print("✅ Checkpoint loaded successfully")

    # ------------------------------------------------------------------
    # Stage 1: VIB pretraining on clean synthetic images
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: VIB PRETRAINING ON CLEAN SYNTHETIC IMAGES")
    print("=" * 70)

    model.train()
    print("\nConfiguring training...")
    print("  Setting BatchNorm momentum = 0.01")
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.momentum = 0.01

    params = count_parameters(model)
    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Encoder (total):      {params['encoder']:,}")
    print(f"    Trunk:              {params['encoder_trunk']:,}")
    print(f"    mu_head:            {params['mu_head']:,}")
    print(f"    log_var_head:       {params['log_var_head']:,}")
    print(f"  Predictor parameters: {params['predictor']:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("\nTraining configuration:")
    print(f"  Embedding:         {embedding_name}")
    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Beta (KL weight):  {beta}")
    print(f"  Epochs:            {epochs}")
    print(f"  Mini-batch size:   {batch_size}")
    print(f"  Simulation batch:  {simulation_batch_size}")
    print(f"  Learning rate:     {lr}")
    print(f"  Batches/epoch:     {n_batches_per_epoch}")
    print(f"  Samples/epoch:     {n_batches_per_epoch * simulation_batch_size:,}")
    print("=" * 70)

    history: Dict = {
        "loss": [], "pred_loss": [], "kl_loss": [],
        "emb_std": [], "emb_dist": [],
    }
    last_mu: Optional[torch.Tensor] = None

    with tqdm(range(epochs), desc="Stage 1: VIB pretraining") as tq:
        for epoch in tq:
            avg_loss, avg_pred_loss, avg_kl_loss, last_mu, synthetic_iter = _run_vib_epoch(
                model=model,
                optimizer=optimizer,
                synthetic_iter=synthetic_iter,
                synthetic_loader=synthetic_loader,
                models=models,
                simulation_param=simulation_param,
                device=device,
                batch_size=batch_size,
                n_batches_per_epoch=n_batches_per_epoch,
                beta=beta,
                pred_weights=pred_weights,
                normalizer=normalizer,
                noise_model=None,
                use_noiseless_images=use_noiseless_images
            )

            history["loss"].append(avg_loss)
            history["pred_loss"].append(avg_pred_loss)
            history["kl_loss"].append(avg_kl_loss)

            tq.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "pred": f"{avg_pred_loss:.4f}",
                "kl":   f"{avg_kl_loss:.4f}",
            })

            if epoch % check_frequency == 0 and last_mu is not None:
                emb_std, emb_dist = check_embedding_health(last_mu, device)
                history["emb_std"].append(emb_std)
                history["emb_dist"].append(emb_dist)

                print(f"\n  Stage 1 Epoch {epoch:3d}:")
                print(f"    Total loss:     {avg_loss:.6f}")
                print(f"    Pred loss:      {avg_pred_loss:.6f}")
                print(f"    KL loss:        {avg_kl_loss:.6f}")
                print(f"    Embedding std:  {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")

    print("\nStage 1 complete.")

    # Save paths
    save_path_obj = Path(save_path)
    stem   = save_path_obj.stem
    suffix = save_path_obj.suffix

    # Save encoder after Stage 1
    stage1_encoder_path = save_path_obj.with_name(f"{stem}_stage1{suffix}")
    torch.save(model.encoder.state_dict(), stage1_encoder_path)
    print(f"✅ Stage 1 encoder weights:    {stage1_encoder_path}")

    # ------------------------------------------------------------------
    # Stage 2: train residual noise model on real images
    # ------------------------------------------------------------------

    # Freeze full model for Stage 2
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    noise_model = _train_noise_model(
        encoder=model.encoder,
        image_prior=image_prior,
        models=models,
        simulation_param=simulation_param,
        device=device,
        real_data_mrc_path=real_data_mrc_path,
        noise_epochs=noise_epochs,
        noise_batch_size=noise_batch_size,
        noise_n_batches_per_epoch=noise_n_batches_per_epoch,
        noise_lr=noise_lr,
        lambda_adv=lambda_adv,
        lambda_content=lambda_content,
        check_frequency=check_frequency,
        use_noiseless_images=use_noiseless_images
    )

    # Save noise model after Stage 2
    torch.save(noise_model.state_dict(), noise_model_path)
    print(f"✅ Noise model:                {noise_model_path}")

    # ------------------------------------------------------------------
    # Stage 3: fine-tune encoder with frozen noise model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3: FINE-TUNING ENCODER WITH FROZEN NOISE MODEL")
    print("=" * 70)

    # Re-enable full model for Stage 3
    model.train()
    for p in model.parameters():
        p.requires_grad = True
        
    # Freeze noise model for Stage 3
    noise_model.eval()
    for p in noise_model.parameters():
        p.requires_grad = False

    # Set optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.01)

    synthetic_iter = iter(synthetic_loader)

    with tqdm(range(finetune_epochs), desc="Stage 3: fine-tuning") as tq:
        for epoch in tq:
            avg_loss, avg_pred_loss, avg_kl_loss, last_mu, synthetic_iter = _run_vib_epoch(
                model=model,
                optimizer=optimizer,
                synthetic_iter=synthetic_iter,
                synthetic_loader=synthetic_loader,
                models=models,
                simulation_param=simulation_param,
                device=device,
                batch_size=batch_size,
                n_batches_per_epoch=n_batches_per_epoch,
                beta=beta,
                pred_weights=pred_weights,
                normalizer=normalizer,
                noise_model=noise_model,
                use_noiseless_images=use_noiseless_images
            )

            history["loss"].append(avg_loss)
            history["pred_loss"].append(avg_pred_loss)
            history["kl_loss"].append(avg_kl_loss)

            tq.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "pred": f"{avg_pred_loss:.4f}",
                "kl":   f"{avg_kl_loss:.4f}",
            })

            if epoch % check_frequency == 0 and last_mu is not None:
                emb_std, emb_dist = check_embedding_health(last_mu, device)
                history["emb_std"].append(emb_std)
                history["emb_dist"].append(emb_dist)

                print(f"\n  Stage 3 Epoch {epoch:3d}:")
                print(f"    Total loss:     {avg_loss:.6f}")
                print(f"    Pred loss:      {avg_pred_loss:.6f}")
                print(f"    KL loss:        {avg_kl_loss:.6f}")
                print(f"    Embedding std:  {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")

    # ------------------------------------------------------------------
    # Final embedding health check
    # ------------------------------------------------------------------
    print("\nComputing final embedding statistics...")
    with torch.no_grad():
        final_emb_std, final_emb_dist = check_embedding_health(last_mu, device)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE")
    print("=" * 70)

    final_loss      = history["loss"][-1]
    final_pred_loss = history["pred_loss"][-1]
    final_kl_loss   = history["kl_loss"][-1]
    final_std       = history["emb_std"][-1] if history["emb_std"] else final_emb_std
    final_dist      = history["emb_dist"][-1] if history["emb_dist"] else final_emb_dist

    print(f"\nFinal metrics:")
    print(f"  Embedding:      {embedding_name}")
    print(f"  Total loss:     {final_loss:.6f}")
    print(f"  Pred loss:      {final_pred_loss:.6f}")
    print(f"  KL loss:        {final_kl_loss:.6f}")
    print(f"  Embedding std:  {final_std:.6f}")
    print(f"  Embedding dist: {final_dist:.6f}")

    print("\nQuality assessment:")

    if final_std < 0.01:
        print("  ❌ WARNING: Low embedding diversity (possible collapse)")
    elif final_std < 0.1:
        print("  ⚠️  Embedding diversity is moderate")
    else:
        print("  ✅ Good embedding diversity")

    if final_dist > 20:
        print("  ⚠️  Embeddings very spread out")
    elif final_dist > 15:
        print("  🟡 Embeddings moderately spread")
    elif final_dist > 10:
        print("  ✅ Embeddings reasonably compact")
    else:
        print("  ✅ Embeddings very compact (excellent for flow training)")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING WEIGHTS")
    print("=" * 70)

    torch.save(model.encoder.state_dict(), save_path)
    print(f"✅ Encoder weights:            {save_path}")

    full_model_path = save_path_obj.with_name(f"{stem}_full_model{suffix}")
    torch.save(model.state_dict(), full_model_path)
    print(f"✅ Full model checkpoint:      {full_model_path}")

    history_path = save_path_obj.with_name(f"{stem}_history{suffix}")
    history.update({
        "embedding_name": embedding_name,
        "embedding_dim":  embedding_dim,
        "image_size":     image_size,
        "encoder_params": params["encoder"],
        "mu_head_params": params["mu_head"],
        "log_var_head_params": params["log_var_head"],
        "beta":           beta,
        "pred_weights":   pred_weights,
        "resumed_from":   resume_from,
        "noise_model_path": noise_model_path,
    })
    torch.save(history, history_path)
    print(f"✅ Training history:           {history_path}")

    print("=" * 70 + "\n")
    return model, final_loss


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-stage VIB pretraining of cryo-EM image encoder"
    )

    parser.add_argument("--image_config",          required=True,                       help="Path to image config JSON")
    parser.add_argument("--real_data_mrc",         required=True,                       help="Path to real MRC stack for noise model training")
    parser.add_argument("--embedding",             default="SPATIAL_CRYO",              help="Embedding architecture name")
    parser.add_argument("--embedding_dim",         type=int,   default=16,              help="Encoder output dimension")
    parser.add_argument("--epochs",                type=int,   default=50,             help="Stage 1 epochs")
    parser.add_argument("--batch_size",            type=int,   default=256,             help="Mini-batch size")
    parser.add_argument("--lr",                    type=float, default=2e-4,            help="Stage 1 learning rate")
    parser.add_argument("--simulation_batch_size", type=int,   default=1024,            help="Images generated per simulator call")
    parser.add_argument("--n_batches_per_epoch",   type=int,   default=100,             help="Simulation calls per epoch")
    parser.add_argument("--save_path",             default="pretrained_image_embed.pt", help="Output path for encoder weights")
    parser.add_argument("--check_frequency",       type=int,   default=5,               help="Epoch interval for detailed stats")
    parser.add_argument("--resume_from",           default=None,                        help="Checkpoint path to resume from")
    parser.add_argument("--device",                default="cuda",                      help="Compute device (cuda / cpu)")
    parser.add_argument("--beta",                  type=float, default=1e-3,            help="KL weight")

    parser.add_argument("--weight_conf",    type=float, default=10.0, help="Conformation prediction loss weight")
    parser.add_argument("--weight_orient",  type=float, default=2.0,  help="Orientation prediction loss weight")
    parser.add_argument("--weight_shift",   type=float, default=0.5,  help="Shift prediction loss weight")
    parser.add_argument("--weight_defocus", type=float, default=0.5,  help="Defocus prediction loss weight")
    parser.add_argument("--weight_bfactor", type=float, default=0.3,  help="B-factor prediction loss weight")

    parser.add_argument("--noise_model_path",      default="noise_model.pt", help="Path to save/load noise model")
    parser.add_argument("--noise_epochs",          type=int,   default=50,    help="Stage 2 epochs")
    parser.add_argument("--noise_batch_size",      type=int,   default=64,    help="Stage 2 mini-batch size")
    parser.add_argument("--noise_lr",              type=float, default=2e-4,  help="Stage 2 learning rate")
    parser.add_argument("--noise_n_batches_per_epoch", type=int, default=100, help="Stage 2 batches per epoch")
    parser.add_argument("--finetune_epochs",       type=int,   default=50,    help="Stage 3 epochs")
    parser.add_argument("--finetune_lr",           type=float, default=1e-4,  help="Stage 3 learning rate")
    parser.add_argument("--lambda_adv",            type=float, default=1.0,   help="Stage 2 adversarial loss weight")
    parser.add_argument("--lambda_content",        type=float, default=1.0,   help="Stage 2 content preservation loss weight")
    parser.add_argument("--use_noiseless_images",  action="store_true",       help="Use noiseless images")

    args = parser.parse_args()

    pretrain_image_embed(
        image_config_path     = args.image_config,
        real_data_mrc_path    = args.real_data_mrc,
        resume_from           = args.resume_from,
        embedding_name        = args.embedding,
        device                = args.device,
        embedding_dim         = args.embedding_dim,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        lr                    = args.lr,
        simulation_batch_size = args.simulation_batch_size,
        save_path             = args.save_path,
        check_frequency       = args.check_frequency,
        n_batches_per_epoch   = args.n_batches_per_epoch,
        beta                  = args.beta,
        pred_weights          = {
            "conf":    args.weight_conf,
            "orient":  args.weight_orient,
            "shift":   args.weight_shift,
            "defocus": args.weight_defocus,
            "bfactor": args.weight_bfactor,
        },
        noise_model_path      = args.noise_model_path,
        noise_epochs          = args.noise_epochs,
        noise_batch_size      = args.noise_batch_size,
        noise_lr              = args.noise_lr,
        noise_n_batches_per_epoch = args.noise_n_batches_per_epoch,
        finetune_epochs       = args.finetune_epochs,
        finetune_lr           = args.finetune_lr,
        lambda_adv            = args.lambda_adv,
        lambda_content        = args.lambda_content,
        use_noiseless_images  = args.use_noiseless_images,
    )
