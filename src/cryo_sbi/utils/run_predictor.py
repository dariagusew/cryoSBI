"""
run_predictor.py.

Evaluates a stack of real cryo-EM images (.mrc / .mrcs) using a full pretrained VIB model 
(Encoder + Predictor) to estimate the global conformational distribution.

Estimates probabilities in two distinct ways:
  1. Hard Assignment (Argmax): Each image is assigned to its highest-logit model class.
     Reported as the percentage of particles assigned to each conformation.
  2. Soft Assignment: 
     A. Average Softmax: Computes softmax per image, then averages probabilities across all images.
     B. Softmax of Average Logits: Averages raw logits across all images, then applies softmax.

Usage:
    python run_predictor.py \
        --real_mrc real_particles.mrcs \
        --full_model_ckpt pretrained_image_embed_full_model.pt \
        --image_config config.json \
        --embedding SPATIAL_CRYO \
        --embedding_dim 16 \
        --batch_size 512 \
        --output_csv conformation_estimates.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import mrcfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


# ============================================================================
# MODEL DEFINITIONS (Exact match to pretrain_image_embed_v6.py)
# ============================================================================

class FullParamPredictor(nn.Module):
    """Predicts all parameters (X, θ) from z."""
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
    """Full Model Scaffolding (Encoder + Predictor)."""
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

    def forward(self, x: torch.Tensor):
        # At inference on real data, use deterministic mu embedding
        out = self.encoder.forward_inference(x)
        mu = out[0] if isinstance(out, tuple) else out
        preds = self.predictor(mu)
        return mu, preds


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def estimate_conformations(args):
    device = args.device

    # 1. Load Image Config & Conformational Models Count
    with open(args.image_config) as f:
        image_config = json.load(f)
    image_size = image_config["N_PIXELS"]

    if image_config["MODEL_FILE"].endswith("npy"):
        models = np.load(image_config["MODEL_FILE"])
    else:
        models = torch.load(image_config["MODEL_FILE"])
    n_conformations = len(models)

    print("\n" + "=" * 75)
    print("CONFORMATIONAL PROBABILITY ESTIMATION")
    print("=" * 75)
    print(f"  Image Size:         {image_size}x{image_size}")
    print(f"  N Conformations:    {n_conformations}")
    print(f"  Embedding Net:      {args.embedding} (dim={args.embedding_dim})")
    print(f"  Full Model Checkpoint: {args.full_model_ckpt}")
    print(f"  Real MRC Stack:     {args.real_mrc}")
    print("=" * 75 + "\n")

    # 2. Build Model and Load Weights
    model = ImageEmbedPretrainModel(
        args.embedding, args.embedding_dim, image_size, n_conformations
    ).to(device)

    print(f"Loading checkpoint weights into full model...")
    state_dict = torch.load(args.full_model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model successfully loaded in eval mode.\n")

    # 3. Stream Real MRC Stack and Predict Conformation Logits
    hard_counts = np.zeros(n_conformations, dtype=np.int64)
    sum_softmax_probs = np.zeros(n_conformations, dtype=np.float64)
    sum_logits = np.zeros(n_conformations, dtype=np.float64)
    total_particles = 0

    with mrcfile.mmap(args.real_mrc, permissive=True) as mrc:
        num_images = mrc.data.shape[0]
        print(f"Processing {num_images:,} particles in batches of {args.batch_size}...")

        with torch.no_grad():
            for i in tqdm(range(0, num_images, args.batch_size), desc="Inferring Conformations"):
                # Memory-safe batch slicing
                batch_np = mrc.data[i:i + args.batch_size].copy().astype(np.float32)
                batch_tensor = torch.from_numpy(batch_np).to(device)

                # Add channel dimension if needed: (B, H, W) -> (B, 1, H, W)
                if batch_tensor.ndim == 3:
                    batch_tensor = batch_tensor.unsqueeze(1)

                # Standard per-image normalization (matching V6 pretraining)
                mu_img = batch_tensor.mean(dim=(-1, -2), keepdim=True)
                std_img = batch_tensor.std(dim=(-1, -2), keepdim=True)
                batch_tensor = (batch_tensor - mu_img) / (std_img + 1e-8)

                # Forward pass: Real image -> Deterministic mu -> Predictor
                _, preds = model(batch_tensor)
                conf_logits = preds["conf"]  # Shape: (batch_size, n_conformations)

                # Softmax per image
                softmax_probs = F.softmax(conf_logits, dim=-1)

                # Argmax per image (Highest logit model assignment)
                argmax_indices = torch.argmax(conf_logits, dim=-1).cpu().numpy()

                # Accumulate statistics on CPU
                for idx in argmax_indices:
                    hard_counts[idx] += 1

                sum_softmax_probs += softmax_probs.sum(dim=0).cpu().numpy()
                sum_logits += conf_logits.sum(dim=0).cpu().numpy()
                total_particles += len(batch_np)

    # 4. Calculate Global Estimates
    # Method 1: Hard Assignment Percentage
    hard_probs = (hard_counts / total_particles) * 100.0

    # Method 2A: Average Softmax Probability Percentage
    soft_probs_avg = (sum_softmax_probs / total_particles) * 100.0

    # Method 2B: Softmax of Averaged Logits Percentage
    avg_logits = sum_logits / total_particles
    avg_logits_tensor = torch.from_numpy(avg_logits).float()
    soft_logit_probs = (F.softmax(avg_logits_tensor, dim=-1).numpy()) * 100.0

    # 5. Display Formatted Results Table
    header = f"{'Model ID':<10} | {'Hard Assign (Argmax) %':<24} | {'Avg Softmax Prob %':<22} | {'Softmax(Avg Logits) %':<22}"
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"CONFORMATIONAL DISTRIBUTION RESULTS (Total Particles = {total_particles:,})")
    print("=" * len(header))
    print(header)
    print(separator)

    for c in range(n_conformations):
        print(f"Model {c:<4d} | {hard_probs[c]:<24.2f} | {soft_probs_avg[c]:<22.2f} | {soft_logit_probs[c]:<22.2f}")
    print("=" * len(header))

    # 6. Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        with open(output_path, "w") as f:
            f.write("model_id,hard_argmax_count,hard_argmax_pct,avg_softmax_pct,softmax_of_avg_logits_pct\n")
            for c in range(n_conformations):
                f.write(f"{c},{hard_counts[c]},{hard_probs[c]:.4f},{soft_probs_avg[c]:.4f},{soft_logit_probs[c]:.4f}\n")
        print(f"\n✅ Results successfully saved to CSV: {output_path.resolve()}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate conformational probabilities from real cryo-EM images using full VIB predictor model."
    )
    parser.add_argument("--real_mrc",        required=True, help="Path to real MRC/MRCS image stack")
    parser.add_argument("--full_model_ckpt", required=True, help="Path to full model checkpoint (*_full_model.pt)")
    parser.add_argument("--image_config",    required=True, help="Path to image config JSON")
    parser.add_argument("--embedding",       default="SPATIAL_CRYO", help="Embedding architecture name")
    parser.add_argument("--embedding_dim",   type=int, default=16, help="Encoder embedding dimension")
    parser.add_argument("--batch_size",      type=int, default=512, help="Mini-batch size for evaluation")
    parser.add_argument("--device",          default="cuda", help="Compute device (cuda / cpu)")
    parser.add_argument("--output_csv",      default="conformation_estimates.csv", help="Output path for CSV summary")

    args = parser.parse_args()
    estimate_conformations(args)
