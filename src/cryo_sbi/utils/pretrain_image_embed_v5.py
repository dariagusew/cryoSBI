# "pretrain_image_embed_v5.py"
"""
pretrain_image_embed_v5.py

VIB pre-training of image encoder on synthetic cryo-EM data.

The encoder is trained as a stochastic sufficient statistic for the full
parameter vector (X, θ). The encoder's public interface returns a
deterministic embedding mu = encoder(d) used by the flow at inference.
During VIB pretraining only, encoder.forward_vib(d) additionally returns
log_var and a reparameterized sample z; the log_var_head is discarded
afterwards.

Usage:
    python pretrain_image_embed_v5.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO \
        --embedding_dim 16 \
        --epochs 100
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import (
    cryo_em_simulator,
    create_simulation_param,
)


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
        self.snr_head     = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {
            "conf":    self.conf_head(h),
            "orient":  self.orient_head(h),
            "shift":   self.shift_head(h),
            "defocus": self.defocus_head(h).squeeze(-1),
            "bfactor": self.bfactor_head(h).squeeze(-1),
            "snr":     self.snr_head(h).squeeze(-1),
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
        print(f"  Predictor: z → (X={n_conformations} classes, orient, shift, defocus, bfactor, snr)")


    def forward(self, x: torch.Tensor):
        mu, log_var, z = self.encoder.forward_vib(x)
        preds = self.predictor(z)
        return mu, log_var, z, preds


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

        snr_min, snr_max = map(float, image_config["SNR"])
        _register("snr", (snr_min + snr_max) / 2.0, (snr_max - snr_min) / math.sqrt(12.0))

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
    # conformation loss: normalized 0 - 1
    n_classes = preds["conf"].size(-1)
    L_conf = F.cross_entropy(preds["conf"], targets["indices"]) / math.log(n_classes)

    # orientation loss: normalized to 0 - 1
    L_orient = _quaternion_loss(preds["orient"], targets["quaternions"]) / 2.0

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
    L_snr = F.mse_loss(
        preds["snr"],
        normalizer.normalize("snr", targets["snr"].float().reshape(-1)),
    )

    total_weight = sum(pred_weights.values())

    L_pred = (
          pred_weights["conf"]    * L_conf
        + pred_weights["orient"]  * L_orient
        + pred_weights["shift"]   * L_shift
        + pred_weights["defocus"] * L_defocus
        + pred_weights["bfactor"] * L_bfactor
        + pred_weights["snr"]     * L_snr
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
) -> tuple:
    """Single VIB epoch on synthetic images."""
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
            # Synthetic images out of simulator; use noisy output.
            noisy_images, _ = cryo_em_simulator(
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

        n_full = (len(noisy_images) // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            sl = slice(i, i + batch_size)

            targets = {
                "indices":     indices[sl].squeeze(-1).round().long().to(device, non_blocking=True),
                "quaternions": quaternions[sl].to(device, non_blocking=True),
                "shift":       shift[sl].to(device,       non_blocking=True),
                "defocus":     defocus[sl].to(device,     non_blocking=True),
                "b_factor":    b_factor[sl].to(device,    non_blocking=True),
                "snr":         snr[sl].to(device,         non_blocking=True),
            }

            optimizer.zero_grad()
            mu, log_var, z, preds = model(noisy_images[sl])
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


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def pretrain_image_embed(
    image_config_path: str,
    resume_from: Optional[str] = None,
    embedding_name: str = "SPATIAL_CRYO",
    device: str = "cuda",
    embedding_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-4,
    simulation_batch_size: int = 1024,
    save_path: str = "pretrained_image_embed.pt",
    check_frequency: int = 5,
    n_batches_per_epoch: int = 100,
    beta: float = 1e-3,
    pred_weights: Optional[Dict[str, float]] = None,
):
    print("\n" + "=" * 70)
    print(f"PRETRAINING: {embedding_name}")
    print("Training mode: Stage 1 only on synthetic images")

    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 70)

    if pred_weights is None:
        pred_weights = {
            "conf":    1.0,
            "orient":  0.0,
            "shift":   0.0,
            "defocus": 0.0,
            "bfactor": 0.0,
            "snr":     0.0,
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

    image_prior      = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    synthetic_loader = PriorLoader(image_prior, batch_size=simulation_batch_size, num_workers=4)
    synthetic_iter   = iter(synthetic_loader)
    simulation_param = create_simulation_param(image_config, models, device=device)

    # ------------------------------------------------------------------
    # Fixed target normalization
    # ------------------------------------------------------------------
    print("Building fixed target normalizer from prior ranges...")
    normalizer = FixedTargetNormalizer(image_config).to(device)

    for key in ("shift", "defocus", "bfactor", "snr"):
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
    # Stage 1: VIB pretraining on synthetic images
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: VIB PRETRAINING ON SYNTHETIC IMAGES")
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

    # Simple cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
            )

            scheduler.step()

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

    save_path_obj = Path(save_path)
    stem   = save_path_obj.stem
    suffix = save_path_obj.suffix

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
        description="VIB pretraining of cryo-EM image encoder"
    )

    parser.add_argument("--image_config",          required=True,                       help="Path to image config JSON")
    parser.add_argument("--embedding",             default="SPATIAL_CRYO",              help="Embedding architecture name")
    parser.add_argument("--embedding_dim",         type=int,   default=16,              help="Encoder output dimension")
    parser.add_argument("--epochs",                type=int,   default=100,             help="Stage 1 epochs")
    parser.add_argument("--batch_size",            type=int,   default=256,             help="Mini-batch size")
    parser.add_argument("--lr",                    type=float, default=2e-4,            help="Stage 1 learning rate")
    parser.add_argument("--simulation_batch_size", type=int,   default=1024,            help="Images generated per simulator call")
    parser.add_argument("--n_batches_per_epoch",   type=int,   default=100,             help="Simulation calls per epoch")
    parser.add_argument("--save_path",             default="pretrained_image_embed.pt", help="Output path for encoder weights")
    parser.add_argument("--check_frequency",       type=int,   default=5,               help="Epoch interval for detailed stats")
    parser.add_argument("--resume_from",           default=None,                        help="Checkpoint path to resume from")
    parser.add_argument("--device",                default="cuda",                      help="Compute device (cuda / cpu)")
    parser.add_argument("--beta",                  type=float, default=1e-3,            help="KL weight")

    parser.add_argument("--weight_conf",    type=float, default=1.0,  help="Conformation prediction loss weight")
    parser.add_argument("--weight_orient",  type=float, default=0.0,  help="Orientation prediction loss weight")
    parser.add_argument("--weight_shift",   type=float, default=0.0,  help="Shift prediction loss weight")
    parser.add_argument("--weight_defocus", type=float, default=0.0,  help="Defocus prediction loss weight")
    parser.add_argument("--weight_bfactor", type=float, default=0.0,  help="B-factor prediction loss weight")
    parser.add_argument("--weight_snr",     type=float, default=0.0,  help="SNR prediction loss weight")

    args = parser.parse_args()

    pretrain_image_embed(
        image_config_path     = args.image_config,
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
            "snr":     args.weight_snr,
        },
    )
