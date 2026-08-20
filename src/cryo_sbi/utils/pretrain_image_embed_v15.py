# "pretrain_image_embed_v11.py"
"""
pretrain_image_embed_v11.py

VIB pre-training of image encoder on synthetic cryo-EM data.

The encoder is trained as a stochastic sufficient statistic for the full
parameter vector (X, theta). The encoder's public interface returns a
deterministic embedding mu = encoder(d) used by the flow at inference.
During VIB pretraining only, encoder.forward_vib(d) additionally returns
log_var and a reparameterized sample z; the log_var_head is discarded
afterwards.

This version keeps the original predictor head and adds an auxiliary
Neural Ratio Estimation (NRE) head that operates on the deterministic
embedding mu.  Class labels are fed to the NRE head as raw integer
indices (no one-hot, no learned embedding).

Usage:
    python pretrain_image_embed_v11.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO \
        --embedding_dim 16 \
        --epochs 100 \
        --real_data_mrc real_images.mrc
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
from geomloss import SamplesLoss
import mrcfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
    Predicts all parameters (X, theta) from z.
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

class NREHead(nn.Module):
    """
    Neural Ratio Estimation head operating on the deterministic
    embedding mu with Spectral Normalization for Lipschitz smoothness.
    """
    def __init__(
        self,
        x_dim: int,
        n_conformations: int,
        hidden_features: Tuple[int, ...] = (256, 128, 64),
        activation: nn.Module = nn.LeakyReLU,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        dims = [n_conformations + x_dim] + list(hidden_features) + [1]
        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            
            # Apply Spectral Normalization to all hidden layers
            if i < len(dims) - 2:
                layers.append(nn.utils.spectral_norm(linear))
                layers.append(activation())
                layers.append(nn.Dropout(p=dropout_p))
            else:
                # Output layer: plain linear (or spectral_norm if you want strict output bounding)
                layers.append(linear)
                
        self.net = nn.Sequential(*layers)

    def forward(self, theta_one_hot: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # theta_one_hot: (B, K) one-hot float tensor
        # x:             (B, D)
        h = torch.cat([theta_one_hot, x], dim=-1)
        return self.net(h).squeeze(-1)


class NRELossCrossClass(nn.Module):
    """
    NRE loss using only cross-class off-diagonal pairs as negatives.
    For K classes this estimates log p(d|X) - log p(d|X' != X).
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_r: torch.Tensor, theta_idx: torch.Tensor) -> torch.Tensor:
        """
        log_r:     (N, N) matrix where log_r[i, j] = log r(theta_i, x_j)
        theta_idx: (N,) long tensor of class indices
        """
        N = log_r.size(0)

        # Positive term: diagonal (joint samples)
        log_r_pos = torch.diag(log_r)

        # Negative mask: off-diagonal AND different class
        diff_class = theta_idx.unsqueeze(0) != theta_idx.unsqueeze(1)   # (N, N)
        mask = torch.eye(N, dtype=torch.bool, device=log_r.device)
        valid_neg = diff_class & ~mask

        if valid_neg.sum() == 0:
            # Fallback to plain off-diagonal if all samples are same class
            log_r_neg = log_r[~mask]
        else:
            log_r_neg = log_r[valid_neg]

        loss = -F.logsigmoid(log_r_pos).mean()
        loss = loss + (-F.logsigmoid(-log_r_neg).mean())
        return loss

class NRELossHybrid(nn.Module):
    """
    NRE loss using simulated images for positive (joint) samples and
    real images for negative (marginal) samples.
    Estimates log p(d_sim|theta) - log p(d_real).
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        nre_head: nn.Module,
        theta_one_hot_sim: torch.Tensor,
        mu_sim: torch.Tensor,
        mu_real: torch.Tensor
    ) -> torch.Tensor:
        """
        nre_head:          The NRE head module (model.nre).
        theta_one_hot_sim: (N_sim, K) one-hot tensor of labels for simulated data.
        mu_sim:            (N_sim, D) embeddings from the simulated batch.
        mu_real:           (N_real, D) embeddings from the real image batch.
        """
        # 1. Positive term: log r(theta_sim, mu_sim) from joint samples
        log_r_pos = nre_head(theta_one_hot_sim, mu_sim)
        loss_pos = -F.logsigmoid(log_r_pos).mean()

        # 2. Negative term: log r(theta_sim, mu_real) from marginal samples
        N_sim = theta_one_hot_sim.size(0)
        N_real = mu_real.size(0)

        # Expand tensors to form all (theta_sim, mu_real) pairs
        theta_expanded = theta_one_hot_sim.unsqueeze(1).expand(N_sim, N_real, -1)
        mu_expanded = mu_real.unsqueeze(0).expand(N_sim, N_real, -1)

        # Reshape for batch processing by the NRE head
        theta_flat = theta_expanded.reshape(N_sim * N_real, -1)
        mu_flat = mu_expanded.reshape(N_sim * N_real, -1)

        log_r_neg = nre_head(theta_flat, mu_flat)
        loss_neg = -F.logsigmoid(-log_r_neg).mean()

        return loss_pos + loss_neg

class MrcDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for an MRC file.
    Uses mrcfile.mmap to read images from disk on-the-fly.
    """
    def __init__(self, mrc_path: str):
        super().__init__()
        self.mrc_path = mrc_path
        # Use a persistent memory-map object; will be opened in worker processes
        self.mrc = mrcfile.mmap(mrc_path, mode='r')
        self.n_images = self.mrc.data.shape[0]

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Slicing the mmap object reads from disk
        image_np = self.mrc.data[idx].copy().astype(np.float32)
        image = torch.from_numpy(image_np)

        # Add channel dimension if it's missing
        if image.ndim == 2:
            image = image.unsqueeze(0)
        
        # Per-image normalization
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + 1e-8)
        
        return image

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (N, D) embeddings, from two views concatenated [view1, view2]
        labels:   (N,) class labels, repeated [labels, labels]
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        pos_mask = mask * logits_mask # Mask for positive pairs (excluding self)

        exp_logits = torch.exp(logits) * logits_mask
        
        # Denominator for DCL includes ONLY the negative pairs.
        # We get the negative mask by inverting the full positive mask (including self-identity).
        neg_mask = 1. - mask
        log_prob = logits - torch.log((exp_logits * neg_mask).sum(1, keepdim=True))

        # Compute loss over positive pairs
        # Note: handle cases with no positive pairs to avoid division by zero
        num_pos = pos_mask.sum(1)
        num_pos[num_pos == 0] = 1
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / num_pos
        
        loss = -mean_log_prob_pos.mean()
        return loss

class ImageEmbedPretrainModel(nn.Module):
    def __init__(
        self,
        embedding_name: str,
        embedding_dim: int,
        image_size: int,
        n_conformations: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_name = embedding_name
        self.embedding_dim  = embedding_dim
        self.image_size     = image_size

        self.encoder   = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size, dropout=dropout)
        self.predictor = FullParamPredictor(embedding_dim, n_conformations)
        self.nre       = NREHead(x_dim=embedding_dim, n_conformations=n_conformations)

        # Verify the chosen encoder supports the VIB training interface
        if not hasattr(self.encoder, "forward_vib"):
            raise ValueError(
                f"Embedding net '{embedding_name}' must implement forward_vib() "
                f"for VIB pretraining."
            )

        print(f"  Encoder:   {embedding_name}  (D={image_size}, output_dim={embedding_dim})")
        print(f"    mu_head inside; log_var_head inside (training only)")
        print(f"  Predictor: z -> (X={n_conformations} classes, orient, shift, defocus, bfactor, snr)")
        print(f"  NRE head:  theta (one-hot) + mu (dim={embedding_dim}) -> log r")


    def forward(self, x: torch.Tensor, theta: torch.Tensor, noise_factor: float = 0.05):
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

def centered_cosine_consistency_loss(mu_A: torch.Tensor, mu_B: torch.Tensor) -> torch.Tensor:
    """
    Mean-centers embeddings per batch before computing cosine similarity.
    Prevents the encoder from shifting embeddings away from origin (0,0).
    Bounded in [0, 1].
    """
    mu_A_centered = mu_A - mu_A.mean(dim=0, keepdim=True)
    mu_B_centered = mu_B - mu_B.mean(dim=0, keepdim=True)
    cosine_sim = F.cosine_similarity(mu_A_centered, mu_B_centered, dim=-1)
    return 0.5 * (1.0 - cosine_sim).mean()


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
    L_conf = F.cross_entropy(preds["conf"], targets["indices"], label_smoothing=0.0) / math.log(n_classes)

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

    ind_losses = {
        "conf":    L_conf,
        "orient":  L_orient,
        "shift":   L_shift,
        "defocus": L_defocus,
        "bfactor": L_bfactor,
        "snr":     L_snr,
    }

    return L_pred + beta * L_kl, L_pred, L_kl, ind_losses


# ============================================================================
# VALIDATION METRICS
# ============================================================================

def compute_manifold_overlap(
    z_synth: torch.Tensor, 
    z_real: torch.Tensor, 
    k: int = 3
) -> Tuple[float, float, float]:
    dists_real_synth = torch.cdist(z_real, z_synth)
    
    min_dists, _ = dists_real_synth.min(dim=1)
    median_dist = min_dists.median().item()
    p90_dist = torch.quantile(min_dists, 0.90).item()
    
    dists_synth_synth = torch.cdist(z_synth, z_synth)
    
    sorted_dists, _ = dists_synth_synth.sort(dim=1)
    radii = sorted_dists[:, k]
    
    is_covered = (dists_real_synth <= radii.unsqueeze(0)).any(dim=1)
    coverage_pct = (is_covered.float().mean().item()) * 100.0
    
    return coverage_pct, median_dist, p90_dist

def get_embeddings_in_batches(encoder, images: torch.Tensor, batch_size: int) -> torch.Tensor:
    z_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            end_idx = min(i + batch_size, len(images))
            out = encoder.forward_inference(images[i:end_idx])
            z = out[0] if isinstance(out, tuple) else out
            z_list.append(z)
    return torch.cat(z_list, dim=0)

def get_synth_embeddings_vib(encoder, images: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    mu_list, z_list, sigma_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            end_idx = min(i + batch_size, len(images))
            batch = images[i:end_idx]
            mu, log_var, z = encoder.forward_vib(batch)
            sigma = (0.5 * log_var).exp()
            mu_list.append(mu)
            z_list.append(z)
            sigma_list.append(sigma)
    mu_synth = torch.cat(mu_list, dim=0)
    z_synth  = torch.cat(z_list, dim=0)
    sigma_synth = torch.cat(sigma_list, dim=0)

    emb_std = mu_synth.std(dim=0).mean().item()
    noise_std = sigma_synth.mean().item()
    nsr = (noise_std / (emb_std + 1e-8)) * 100.0
    return mu_synth, z_synth, nsr


# ============================================================================
# UTILITIES
# ============================================================================

def compute_intra_class_jitter(mu: torch.Tensor, labels: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
    """
    Computes jitter proportional to the WITHIN-CLASS standard deviation (D_intra).
    Guarantees noise perturbation never spills into neighboring conformation clusters.
    
    mu:     (N, D) latent embeddings
    labels: (N,) class indices
    eta:    fraction of intra-class std to use (e.g. 0.15 = 15%)
    """
    with torch.no_grad():
        N, D = mu.shape
        n_classes = int(labels.max().item()) + 1
        
        # 1. Compute mean embedding per conformation class in this batch
        class_means = torch.zeros(n_classes, D, device=mu.device)
        class_counts = torch.zeros(n_classes, 1, device=mu.device)
        
        class_means.index_add_(0, labels, mu)
        class_counts.index_add_(0, labels, torch.ones((N, 1), device=mu.device))
        
        class_counts = torch.clamp(class_counts, min=1.0)
        class_means = class_means / class_counts  # (K, D)
        
        # 2. Compute residual vectors (distance from each sample to its class mean)
        mu_centered = mu - class_means[labels]  # (N, D)
        
        # 3. Average intra-class standard deviation per latent dimension
        std_intra = torch.sqrt((mu_centered ** 2).mean(dim=0) + 1e-8)  # (D,)
        
        # 4. Generate jitter
        jitter = torch.randn_like(mu) * (eta * std_intra)
        
    return mu + jitter

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
        "nre_head":     sum(p.numel() for p in model.nre.parameters()),
    }


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def _run_vib_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    synthetic_iter,
    synthetic_loader: PriorLoader,
    real_data_loader: Optional[DataLoader],
    real_data_iter: Optional[iter],
    use_hybrid_nre: bool,
    models: torch.Tensor,
    simulation_param: dict,
    device: str,
    batch_size: int,
    n_batches_per_epoch: int,
    beta: float,
    pred_weights: Dict[str, float],
    normalizer: FixedTargetNormalizer,
    epoch: int,
    weight_cons: float,
    beta_NRE: float,
    beta_OT: float,
    snr_range: Optional[Tuple[float, float]] = None,
    nre_warmup_epochs: int = 15,
    jittering_factor: float = 0.0,
    supcon_temperature: float = 0.1,
) -> tuple:
    """Single VIB epoch on synthetic images."""
    if use_hybrid_nre:
        nre_loss_fn = NRELossHybrid()
        assert real_data_loader is not None, "Real data loader is required for hybrid NRE loss."
    else:
        nre_loss_fn = NRELossCrossClass()
    if beta_OT>0.0:
       sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    epoch_loss = 0.0
    epoch_pred_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_cons_loss = 0.0
    epoch_nre_loss = 0.0
    epoch_ot_loss = 0.0
    epoch_ind_losses = {"conf": 0.0, "orient": 0.0, "shift": 0.0, "defocus": 0.0, "bfactor": 0.0, "snr": 0.0}
    n_steps = 0
    last_mu: Optional[torch.Tensor] = None

    for _ in range(n_batches_per_epoch):
        # 1. First set of images for all losses except SupCon
        try:
            parameters = next(synthetic_iter)
        except StopIteration:
            synthetic_iter = iter(synthetic_loader)
            parameters = next(synthetic_iter)

        indices, quaternions, shift, defocus, b_factor, amp, snr = parameters

        # Generate Set A
        with torch.no_grad():
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

        if weight_cons > 0.0:    
           # 2. Second set of images for SupCon only 
           try:
               parameters_B = next(synthetic_iter)
           except StopIteration:
               synthetic_iter = iter(synthetic_loader)
               parameters_B = next(synthetic_iter)
           
           indices_B, quaternions_B, shift_B, defocus_B, b_factor_B, amp_B, snr_B = parameters_B
           
           # === k-GROUPED BATCH
           num_groups_per_minibatch = 8
           group_size = batch_size // num_groups_per_minibatch
           total_groups = quaternions_B.size(0) // group_size

           # Pick total_groups distinct poses out of the pool
           selected_quats  = quaternions_B[::group_size][:total_groups]
           selected_shifts = shift_B[::group_size][:total_groups]

           # Repeat each pose
           quaternions_B = selected_quats.repeat_interleave(group_size, dim=0)
           shift_B       = selected_shifts.repeat_interleave(group_size, dim=0)

           # Generate Set B 
           with torch.no_grad():
                noisy_images_B, _ = cryo_em_simulator(
                    models,
                    indices_B.to(device,     non_blocking=True),
                    quaternions_B.to(device, non_blocking=True),
                    shift_B.to(device,       non_blocking=True),
                    defocus_B.to(device,   non_blocking=True),
                    b_factor_B.to(device,  non_blocking=True),
                    amp_B.to(device,         non_blocking=True),
                    snr_B.to(device,       non_blocking=True),
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

            # load batch of real images
            if use_hybrid_nre or beta_OT>0.0:
                try:
                    real_images = next(real_data_iter)
                except StopIteration:
                    real_data_iter = iter(real_data_loader)
                    real_images = next(real_data_iter)

                real_images = real_images.to(device, non_blocking=True)
                # Get deterministic embedding mu for real images
                mu_real = model.encoder(real_images)

            # VIB loss
            optimizer.zero_grad()
            mu, log_var, z, preds = model(noisy_images[sl], targets["indices"])
            loss, L_pred, L_kl, ind_losses = vib_loss(
                mu, log_var, preds, targets, beta, pred_weights, normalizer
            )

            # auxiliary NRE loss
            theta_indices = targets["indices"]
            n_conformations = model.predictor.conf_head.out_features
            theta_one_hot = F.one_hot(theta_indices, num_classes=n_conformations).float()
            
            if use_hybrid_nre:
                L_nre = nre_loss_fn(model.nre, theta_one_hot, mu, mu_real.detach())
            else:
                # Original logic: use off-diagonal synthetic samples as negatives
                if model.training and jittering_factor > 0:
                    mu_nre = compute_intra_class_jitter(mu, targets["indices"], eta=jittering_factor)
                else:
                    mu_nre = mu

                N = theta_indices.size(0)
                theta_one_hot_i = theta_one_hot.unsqueeze(1).expand(N, N, -1).reshape(N * N, -1)
                mu_j = mu_nre.unsqueeze(0).expand(N, N, -1).reshape(N * N, -1)
            
                log_r_matrix = model.nre(theta_one_hot_i, mu_j).view(N, N)
                L_nre = nre_loss_fn(log_r_matrix, targets["indices"])

            # current weight for NRE loss
            current_beta_NRE = beta_NRE * min(1.0, epoch / max(1.0, float(nre_warmup_epochs))) 
            loss = loss + current_beta_NRE * L_nre

            # optimal transport loss
            if beta_OT>0.0:
                L_ot = sinkhorn(mu, mu_real)
                epoch_ot_loss += L_ot.item()
                loss = loss + beta_OT * L_ot

            # consistency loss
            if weight_cons > 0.0:
                mu_B = model.encoder(noisy_images_B[sl])

                supcon_loss_fn = SupervisedContrastiveLoss(temperature=supcon_temperature)
                features = F.normalize(mu_B, dim=1)
                labels = indices_B[sl].squeeze(-1).round().long().to(device, non_blocking=True) 
                L_cons = supcon_loss_fn(features, labels)

                epoch_cons_loss += L_cons.item()
                loss = loss + weight_cons * L_cons

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss      += loss.item()
            epoch_pred_loss += L_pred.item()
            epoch_kl_loss   += L_kl.item()
            epoch_nre_loss  += L_nre.item()
            for k, v in ind_losses.items():
                epoch_ind_losses[k] += v.item()
            n_steps += 1
            last_mu = mu.detach()

    avg_loss      = epoch_loss      / max(n_steps, 1)
    avg_pred_loss = epoch_pred_loss / max(n_steps, 1)
    avg_kl_loss   = epoch_kl_loss   / max(n_steps, 1)
    avg_cons_loss = epoch_cons_loss / max(n_steps, 1)
    avg_nre_loss  = epoch_nre_loss  / max(n_steps, 1)
    avg_ot_loss   = epoch_ot_loss   / max(n_steps, 1)
    avg_ind_losses = {k: v / max(n_steps, 1) for k, v in epoch_ind_losses.items()}
    return avg_loss, avg_pred_loss, avg_kl_loss, avg_cons_loss, avg_nre_loss, avg_ot_loss, avg_ind_losses, last_mu, synthetic_iter, real_data_iter


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
    beta: float = 1e-5,
    pred_weights: Optional[Dict[str, float]] = None,
    real_data_mrc: Optional[str] = None,
    val_size: int = 3000,
    val_k: int = 3,
    weight_cons: float = 0.0,
    beta_NRE: float = 0.1,
    beta_OT: float = 0.0,
    nre_warmup_epochs: int = 15,
    jittering_factor: float = 0.0,
    supcon_temperature: float = 0.1,
    use_real_for_nre_negatives: bool = False,
    dropout: float = 0.0,
):
    print("\n" + "=" * 70)
    print(f"TRAINING: {embedding_name}")

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
    nre_negative_source = "real images" if use_real_for_nre_negatives else "synthetic (off-diagonal)"
    print(f"  {'beta_NRE':10s}: {beta_NRE:.2f} (NRE negatives from: {nre_negative_source})")
    print(f"  {'cons':10s}: {weight_cons:.2f}")

    # ------------------------------------------------------------------
    # Config and conformational models
    # ------------------------------------------------------------------
    with open(image_config_path) as f:
        image_config = json.load(f)
    image_size = image_config["N_PIXELS"]
    snr_range = tuple(map(float, image_config["SNR"]))

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
    # Fixed Validation Sets Construction & Real Data Loader
    # ------------------------------------------------------------------
    val_real_tensor = None
    val_synth_tensor = None
    real_data_loader = None

    if real_data_mrc is not None:
        print(f"\nBuilding static validation sets for Manifold Coverage (Size={val_size})...")
        
        print(f"  -> Reading {val_size} random images from {real_data_mrc} via memmap...")
        try:
            with mrcfile.mmap(real_data_mrc, mode='r') as mrc:
                total_imgs = mrc.data.shape[0]
                actual_val_size = min(val_size, total_imgs)
                idx = np.random.choice(total_imgs, size=actual_val_size, replace=False)
                idx.sort()
                real_imgs_np = mrc.data[idx].copy()
            
            val_real_tensor = torch.from_numpy(real_imgs_np).float().to(device)
            if val_real_tensor.ndim == 3:
                val_real_tensor = val_real_tensor.unsqueeze(1)
                
            mu_r = val_real_tensor.mean(dim=(-1, -2), keepdim=True)
            std_r = val_real_tensor.std(dim=(-1, -2), keepdim=True)
            val_real_tensor = (val_real_tensor - mu_r) / (std_r + 1e-8)

            if use_real_for_nre_negatives or beta_OT>0.0:
                print(f"  -> Building memory-efficient DataLoader for Hybrid NRE training...")
                real_dataset = MrcDataset(real_data_mrc)
                real_data_loader = DataLoader(
                    real_dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=0, 
                    pin_memory=True, 
                    drop_last=True
                )
                print(f"  ✅ Created Hybrid NRE DataLoader with {len(real_dataset)} images.")
            
        except Exception as e:
            print(f"  ❌ Failed to load real images: {e}")
            if use_real_for_nre_negatives or beta_OT>0.0:
                raise ValueError("Could not load real images, which are required for --use_real_for_nre_negatives.")
            val_real_tensor = None
            
        if val_real_tensor is not None:
            print(f"  -> Generating {actual_val_size} synthetic images...")
            synth_imgs_list = []
            with torch.no_grad():
                while sum(len(x) for x in synth_imgs_list) < actual_val_size:
                    try:
                        v_params = next(synthetic_iter)
                    except StopIteration:
                        synthetic_iter = iter(synthetic_loader)
                        v_params = next(synthetic_iter)

                    v_idx, v_quat, v_shift, v_def, v_bf, v_amp, v_snr = v_params
                    v_imgs, _ = cryo_em_simulator(
                        models,
                        v_idx.to(device, non_blocking=True),
                        v_quat.to(device, non_blocking=True),
                        v_shift.to(device, non_blocking=True),
                        v_def.to(device, non_blocking=True),
                        v_bf.to(device, non_blocking=True),
                        v_amp.to(device, non_blocking=True),
                        v_snr.to(device, non_blocking=True),
                        simulation_param,
                        simulation_param["noise"],
                    )
                    synth_imgs_list.append(v_imgs)
                    
            val_synth_tensor = torch.cat(synth_imgs_list, dim=0)[:actual_val_size]
            print("  ✅ Validation sets successfully fixed on device.")
    elif use_real_for_nre_negatives or beta_OT>0.0:
         raise ValueError("--real_data_mrc must be provided when --use_real_for_nre_negatives is set.")
    else:
        print("\n⚠️ No --real_data_mrc provided. Skipping Sim2Real manifold validation.")

    # ------------------------------------------------------------------
    # Fixed target normalization
    # ------------------------------------------------------------------
    print("\nBuilding fixed target normalizer from prior ranges...")
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
            embedding_name, embedding_dim, image_size, n_conformations, dropout
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
    print("VIB and NRE TRAINING ON SYNTHETIC IMAGES")
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
    print(f"  NRE head parameters:  {params['nre_head']:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    warmup_epochs = 5
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    print("\nTraining configuration:")
    print(f"  Embedding:         {embedding_name}")
    print(f"  Embedding dim:     {embedding_dim}")
    print(f"  Beta (KL weight):  {beta}")
    print(f"  Beta NRE:          {beta_NRE}")
    print(f"  Beta OT:           {beta_OT}")
    print(f"  Epochs:            {epochs}")
    print(f"  Mini-batch size:   {batch_size}")
    print(f"  Simulation batch:  {simulation_batch_size}")
    print(f"  Learning rate:     {lr}")
    print(f"  Batches/epoch:     {n_batches_per_epoch}")
    print(f"  Samples/epoch:     {n_batches_per_epoch * simulation_batch_size:,}")
    print("=" * 70)

    history: Dict = {
        "loss": [], "pred_loss": [], "kl_loss": [], "cons_loss": [], "nre_loss": [],
        "ot_loss": [], "emb_std": [], "emb_dist": [],
        "val_coverage_pct": [], "val_med_dist": [], "val_p90_dist": [],
        "val_mu_coverage_pct": [], "val_mu_med_dist": [], "val_mu_p90_dist": [],
        "val_nsr": [],
    }
    last_mu: Optional[torch.Tensor] = None
    save_path_obj = Path(save_path)

    real_data_iter = iter(real_data_loader) if real_data_loader else None

    with tqdm(range(epochs), desc="Stage 1: VIB pretraining") as tq:
        for epoch in tq:
            avg_loss, avg_pred_loss, avg_kl_loss, avg_cons_loss, avg_nre_loss, avg_ot_loss, avg_ind_losses, last_mu, synthetic_iter, real_data_iter = _run_vib_epoch(
                model=model,
                optimizer=optimizer,
                synthetic_iter=synthetic_iter,
                synthetic_loader=synthetic_loader,
                real_data_loader=real_data_loader,
                real_data_iter=real_data_iter,
                use_hybrid_nre=use_real_for_nre_negatives,
                models=models,
                simulation_param=simulation_param,
                device=device,
                batch_size=batch_size,
                n_batches_per_epoch=n_batches_per_epoch,
                beta=beta,
                pred_weights=pred_weights,
                normalizer=normalizer,
                epoch=epoch,
                weight_cons=weight_cons,
                beta_NRE=beta_NRE,
                beta_OT=beta_OT,
                snr_range=snr_range,
                nre_warmup_epochs=nre_warmup_epochs,
                jittering_factor=jittering_factor,
                supcon_temperature=supcon_temperature,
            )

            scheduler.step()

            history["loss"].append(avg_loss)
            history["pred_loss"].append(avg_pred_loss)
            history["kl_loss"].append(avg_kl_loss)
            history["cons_loss"].append(avg_cons_loss)
            history["nre_loss"].append(avg_nre_loss)
            history["ot_loss"].append(avg_ot_loss)

            postfix_dict = {
                "loss": f"{avg_loss:.4f}",
                "pred": f"{avg_pred_loss:.4f}",
                "kl":   f"{avg_kl_loss:.4f}",
                "nre":  f"{avg_nre_loss:.4f}",
            }
            if weight_cons > 0.0:
                postfix_dict["cons"] = f"{avg_cons_loss:.4f}"
            if beta_OT > 0.0:
                postfix_dict["ot"] = f"{avg_ot_loss:.4f}"
            tq.set_postfix(postfix_dict)

            if epoch % check_frequency == 0 and last_mu is not None:
                emb_std, emb_dist = check_embedding_health(last_mu, device)
                history["emb_std"].append(emb_std)
                history["emb_dist"].append(emb_dist)

                print(f"\n  Stage 1 Epoch {epoch:3d}:")
                print(f"    Total loss:     {avg_loss:.6f}")
                print(f"    Pred loss:      {avg_pred_loss:.6f}")
                print(f"    KL loss:        {avg_kl_loss:.6f}")
                print(f"    NRE loss:       {avg_nre_loss:.6f}")
                if weight_cons > 0.0:
                    print(f"    Cons loss:      {avg_cons_loss:.6f}")
                if beta_OT > 0.0:
                    print(f"    Ot loss:      {avg_ot_loss:.6f}")
                print(f"    Unscaled Predictor Losses:")
                print(f"      conf:    {avg_ind_losses['conf']:.6f}")
                print(f"      orient:  {avg_ind_losses['orient']:.6f}")
                print(f"      shift:   {avg_ind_losses['shift']:.6f}")
                print(f"      defocus: {avg_ind_losses['defocus']:.6f}")
                print(f"      bfactor: {avg_ind_losses['bfactor']:.6f}")
                print(f"      snr:     {avg_ind_losses['snr']:.6f}")
                print(f"    Embedding std:  {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")

                if val_real_tensor is not None and val_synth_tensor is not None:
                    model.eval()
                    mu_synth, z_synth, nsr = get_synth_embeddings_vib(model.encoder, val_synth_tensor, batch_size)
                    mu_real = get_embeddings_in_batches(model.encoder, val_real_tensor, batch_size)
                    model.train()
                    
                    cov_pct_z, med_dist_z, p90_dist_z = compute_manifold_overlap(z_synth, mu_real, k=val_k)
                    cov_pct_mu, med_dist_mu, p90_dist_mu = compute_manifold_overlap(mu_synth, mu_real, k=val_k)

                    history["val_coverage_pct"].append(cov_pct_z)
                    history["val_med_dist"].append(med_dist_z)
                    history["val_p90_dist"].append(p90_dist_z)
                    history["val_mu_coverage_pct"].append(cov_pct_mu)
                    history["val_mu_med_dist"].append(med_dist_mu)
                    history["val_mu_p90_dist"].append(p90_dist_mu)
                    history["val_nsr"].append(nsr)
                    
                    print(f"    Validation (Sim2Real Overlap):")
                    print(f"      Noise-to-Signal Ratio:    {nsr:.2f}%")
                    print(f"      Using z_synth (stochastic):")
                    print(f"        Coverage (% in manifold): {cov_pct_z:.2f}%")
                    print(f"        Distance (Med / p90):     {med_dist_z:.4f} / {p90_dist_z:.4f}")
                    print(f"      Using mu_synth (deterministic):")
                    print(f"        Coverage (% in manifold): {cov_pct_mu:.2f}%")
                    print(f"        Distance (Med / p90):     {med_dist_mu:.4f} / {p90_dist_mu:.4f}")
                
                ep_stem = save_path_obj.stem
                ep_suffix = save_path_obj.suffix
                
                ep_enc_path = save_path_obj.with_name(f"{ep_stem}_epoch{epoch:03d}{ep_suffix}")
                torch.save(model.state_dict(), ep_enc_path)
                print(f"    Saved Checkpoints -> {ep_enc_path.name}")

    # ------------------------------------------------------------------
    # Final embedding health check
    # ------------------------------------------------------------------
    print("\nComputing final embedding statistics...")
    if last_mu is not None:
        with torch.no_grad():
            final_emb_std, final_emb_dist = check_embedding_health(last_mu, device)
    else:
        final_emb_std, final_emb_dist = 0.0, 0.0


    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE")
    print("=" * 70)

    final_loss      = history["loss"][-1]
    final_pred_loss = history["pred_loss"][-1]
    final_kl_loss   = history["kl_loss"][-1]
    final_nre_loss  = history["nre_loss"][-1]
    final_ot_loss   = history["ot_loss"][-1]
    final_std       = history["emb_std"][-1] if history["emb_std"] else final_emb_std
    final_dist      = history["emb_dist"][-1] if history["emb_dist"] else final_emb_dist

    print(f"\nFinal metrics:")
    print(f"  Embedding:      {embedding_name}")
    print(f"  Total loss:     {final_loss:.6f}")
    print(f"  Pred loss:      {final_pred_loss:.6f}")
    print(f"  KL loss:        {final_kl_loss:.6f}")
    print(f"  NRE loss:       {final_nre_loss:.6f}")
    if weight_cons > 0.0:
        print(f"  Cons loss:      {history['cons_loss'][-1]:.6f}")
    if beta_OT > 0.0:
        print(f"  Ot loss:      {history['ot_loss'][-1]:.6f}")
    print(f"  Embedding std:  {final_std:.6f}")
    print(f"  Embedding dist: {final_dist:.6f}")

    if val_real_tensor is not None and val_synth_tensor is not None:
        model.eval()
        f_mu_synth, f_z_synth, f_nsr = get_synth_embeddings_vib(model.encoder, val_synth_tensor, batch_size)
        f_mu_real = get_embeddings_in_batches(model.encoder, val_real_tensor, batch_size)
        model.train()

        f_cov_z, f_med_z, f_p90_z = compute_manifold_overlap(f_z_synth, f_mu_real, k=val_k)
        f_cov_mu, f_med_mu, f_p90_mu = compute_manifold_overlap(f_mu_synth, f_mu_real, k=val_k)

        print(f"\n  Final Validation (Sim2Real Overlap):")
        print(f"    Noise-to-Signal Ratio:    {f_nsr:.2f}%")
        print(f"    Using z_synth (stochastic):")
        print(f"      Coverage (% in manifold): {f_cov_z:.2f}%")
        print(f"      Distance (Med / p90):     {f_med_z:.4f} / {f_p90_z:.4f}")
        print(f"    Using mu_synth (deterministic):")
        print(f"      Coverage (% in manifold): {f_cov_mu:.2f}%")
        print(f"      Distance (Med / p90):     {f_med_mu:.4f} / {f_p90_mu:.4f}")

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

    torch.save(model.state_dict(), save_path)
    print(f"✅ Full model checkpoint:            {save_path}")

    stem = save_path_obj.stem
    suffix = save_path_obj.suffix
    history_path = save_path_obj.with_name(f"{stem}_history{suffix}")
    history.update({
        "embedding_name": embedding_name,
        "embedding_dim":  embedding_dim,
        "image_size":     image_size,
        "encoder_params": params["encoder"],
        "mu_head_params": params["mu_head"],
        "log_var_head_params": params["log_var_head"],
        "predictor_params": params["predictor"],
        "nre_head_params": params["nre_head"],
        "beta":           beta,
        "beta_NRE":       beta_NRE,
        "beta_OT":        beta_OT,
        "pred_weights":   pred_weights,
        "weight_cons":    weight_cons,
        "nre_warmup_epochs": nre_warmup_epochs,
        "use_real_for_nre_negatives": use_real_for_nre_negatives,
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
        description="VIB pretraining of cryo-EM image encoder with auxiliary NRE head"
    )

    parser.add_argument("--image_config",                  required=True,                       help="Path to image config JSON")
    parser.add_argument("--embedding",                     default="SPATIAL_CRYO",              help="Embedding architecture name")
    parser.add_argument("--embedding_dim",                 type=int,   default=16,              help="Encoder output dimension")
    parser.add_argument("--epochs",                        type=int,   default=100,             help="Stage 1 epochs")
    parser.add_argument("--batch_size",                    type=int,   default=256,             help="Mini-batch size")
    parser.add_argument("--lr",                            type=float, default=2e-4,            help="Stage 1 learning rate")
    parser.add_argument("--simulation_batch_size",         type=int,   default=1024,            help="Images generated per simulator call")
    parser.add_argument("--n_batches_per_epoch",           type=int,   default=100,             help="Simulation calls per epoch")
    parser.add_argument("--save_path",                     default="pretrained_image_embed.pt", help="Output path for encoder weights")
    parser.add_argument("--check_frequency",               type=int,   default=5,               help="Epoch interval for detailed stats")
    parser.add_argument("--resume_from",                   default=None,                        help="Checkpoint path to resume from")
    parser.add_argument("--device",                        default="cuda",                      help="Compute device (cuda / cpu)")
    parser.add_argument("--beta",                          type=float, default=1e-5,            help="KL weight")
    parser.add_argument("--beta_NRE",                      type=float, default=0.1,             help="Auxiliary NRE loss weight (gamma fixed to 1.0)")
    parser.add_argument("--beta_OT",                       type=float, default=0.0,             help="Optimal transport loss weight")
    parser.add_argument("--beta_cons",                     type=float, default=0.0,             help="Noise consistency loss weight")
    parser.add_argument("--nre_warmup_epochs",             type=int,   default=15,              help="Epochs over which to warm up NRE loss weight")
    parser.add_argument("--jittering_factor",              type=float, default=0.0,             help="NRE jittering factor")
    parser.add_argument("--supcon_temperature",            type=float, default=0.1,             help="SupCon temperature")
    parser.add_argument("--dropout",                       type=float, default=0.0,             help="Embedding dropout")

    parser.add_argument("--weight_conf",    type=float, default=1.0,  help="Conformation prediction loss weight")
    parser.add_argument("--weight_orient",  type=float, default=0.0,  help="Orientation prediction loss weight")
    parser.add_argument("--weight_shift",   type=float, default=0.0,  help="Shift prediction loss weight")
    parser.add_argument("--weight_defocus", type=float, default=0.0,  help="Defocus prediction loss weight")
    parser.add_argument("--weight_bfactor", type=float, default=0.0,  help="B-factor prediction loss weight")
    parser.add_argument("--weight_snr",     type=float, default=0.0,  help="SNR prediction loss weight")
    
    parser.add_argument("--real_data_mrc",  default=None, help="Path to real .mrc images for Sim2Real validation")
    parser.add_argument("--val_size",       type=int, default=3000, help="Number of images in fixed validation set")
    parser.add_argument("--val_k",          type=int, default=3, help="k value for Synthetic Manifold Radius check")

    parser.add_argument("--use_real_for_nre_negatives", action="store_true", default=False,
                        help="Use real images for NRE negative samples instead of synthetic ones.")

    args = parser.parse_args()

    pretrain_image_embed(
        image_config_path             = args.image_config,
        resume_from                   = args.resume_from,
        embedding_name                = args.embedding,
        device                        = args.device,
        embedding_dim                 = args.embedding_dim,
        epochs                        = args.epochs,
        batch_size                    = args.batch_size,
        lr                            = args.lr,
        simulation_batch_size         = args.simulation_batch_size,
        save_path                     = args.save_path,
        check_frequency               = args.check_frequency,
        n_batches_per_epoch           = args.n_batches_per_epoch,
        beta                          = args.beta,
        pred_weights                  = {
            "conf":    args.weight_conf,
            "orient":  args.weight_orient,
            "shift":   args.weight_shift,
            "defocus": args.weight_defocus,
            "bfactor": args.weight_bfactor,
            "snr":     args.weight_snr,
        },
        real_data_mrc                 = args.real_data_mrc,
        val_size                      = args.val_size,
        val_k                         = args.val_k,
        weight_cons                   = args.beta_cons, 
        beta_NRE                      = args.beta_NRE,
        beta_OT                       = args.beta_OT,
        nre_warmup_epochs             = args.nre_warmup_epochs,
        jittering_factor              = args.jittering_factor,
        supcon_temperature            = args.supcon_temperature,
        use_real_for_nre_negatives    = args.use_real_for_nre_negatives,
        dropout                       = args.dropout,
        )
