# infer_synthetic_nre.py
"""
Synthetic validation of a trained Neural Ratio Estimator.

For each known conformation/state:
    1. Generate synthetic cryo-EM images with that state forced.
    2. Run the trained encoder to obtain deterministic mu.
    3. Evaluate the trained NRE for every possible state.
    4. Predict the state using argmax(log likelihood ratio).
    5. Compute accuracy and a K x K confusion matrix.

This uses:
    - the SAME image/simulation config used by CryoEmSimulator
    - the SAME encoder architecture as infer_populations_nre.py
    - the SAME NRE architecture
    - the SAME full-model checkpoint saved by:
          torch.save(model.state_dict(), save_path)

No experimental images are used.
No population-weight optimization is performed.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi import CryoEmSimulator


# ============================================================================
# NRE HEAD
# ============================================================================

class NREHead(nn.Module):
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

            if i < len(dims) - 2:
                layers.append(nn.utils.spectral_norm(linear))
                layers.append(activation())
                layers.append(nn.Dropout(p=dropout_p))
            else:
                layers.append(linear)

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        theta_one_hot: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:

        h = torch.cat([theta_one_hot, x], dim=-1)
        return self.net(h).squeeze(-1)


# ============================================================================
# ENCODER + NRE
# ============================================================================

class NREInferenceModel(nn.Module):
    def __init__(self, encoder, nre):
        super().__init__()
        self.encoder = encoder
        self.nre = nre


# ============================================================================
# CROSS-CLASS NRE CORRECTION
# ============================================================================

def log_ratio_from_log_rtilde(
    log_r_tilde: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """
    Convert

        log r_tilde =
            log p(d|X) - log p(d|X' != X)

    into

        log r =
            log p(d|X) - log p(d)

    using

        r = K * r_tilde / (r_tilde + K - 1)
    """

    logK = math.log(float(K))
    logK_minus_1 = math.log(float(K - 1))

    log_den = torch.logaddexp(
        torch.tensor(
            logK_minus_1,
            dtype=log_r_tilde.dtype,
            device=log_r_tilde.device,
        ),
        log_r_tilde,
    )

    return logK + log_r_tilde - log_den


# ============================================================================
# NRE EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_log_ratios(
    model,
    images: torch.Tensor,
    K: int,
    device: torch.device,
    batch_size: int = 256,
    pair_batch_size: int = 4096,
    normalize_images: bool = False,
    flip_contrast: bool = False,
    skip_rtilde_correction: bool = False,
):
    """
    Evaluate all K NRE ratios for a tensor of synthetic images.

    Returns:
        log_ratio: [N_images, K]
    """

    model.eval()

    results = []

    for start in range(0, len(images), batch_size):

        end = min(start + batch_size, len(images))

        imgs = images[start:end].clone().float()

        if flip_contrast:
            imgs = -imgs

        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(1)

        if normalize_images:
            mean = imgs.mean(dim=(-1, -2), keepdim=True)
            std = imgs.std(dim=(-1, -2), keepdim=True)

            imgs = (imgs - mean) / (std + 1e-8)

        imgs = imgs.to(device)

        # ------------------------------------------------------------
        # Deterministic encoder output
        # ------------------------------------------------------------

        out = model.encoder.forward_inference(imgs)

        mu = out[0] if isinstance(out, tuple) else out

        B, D = mu.shape

        # ------------------------------------------------------------
        # Evaluate all (image, state) combinations
        # ------------------------------------------------------------

        mu_rep = (
            mu.unsqueeze(1)
            .expand(-1, K, -1)
            .reshape(B * K, D)
        )

        theta_indices = (
            torch.arange(K, device=device)
            .unsqueeze(0)
            .expand(B, -1)
            .reshape(-1)
        )

        log_r_raw = torch.empty(
            B * K,
            dtype=torch.float32,
            device="cpu",
        )

        for p_start in range(0, B * K, pair_batch_size):

            p_end = min(
                p_start + pair_batch_size,
                B * K,
            )

            theta = F.one_hot(
                theta_indices[p_start:p_end],
                num_classes=K,
            ).float()

            mu_batch = mu_rep[p_start:p_end]

            scores = model.nre(
                theta,
                mu_batch,
            )

            log_r_raw[p_start:p_end] = scores.cpu()

        log_r_raw = log_r_raw.reshape(B, K)

        # ------------------------------------------------------------
        # Cross-class correction
        # ------------------------------------------------------------

        if skip_rtilde_correction:
            log_ratio = log_r_raw
        else:
            log_ratio = log_ratio_from_log_rtilde(
                log_r_raw,
                K,
            )

        results.append(log_ratio)

    return torch.cat(results, dim=0)


# ============================================================================
# MAIN
# ============================================================================

def main(args):

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    if "cuda" in args.device and not torch.cuda.is_available():
        print(
            f"CUDA unavailable. Switching from {args.device} to CPU."
        )
        args.device = "cpu"

    device = torch.device(args.device)

    print("=" * 70)
    print("SYNTHETIC NRE VALIDATION")
    print("=" * 70)
    print(f"Device: {device}")

    # ------------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------------

    with open(args.config) as f:
        config = json.load(f)

    image_size = config["N_PIXELS"]

    print(f"Image size: {image_size}")
    print(f"Model file: {config['MODEL_FILE']}")

    # ------------------------------------------------------------------------
    # Create simulator
    #
    # This automatically:
    #   - loads the config
    #   - loads MODEL_FILE
    #   - initializes priors
    #   - initializes simulation parameters
    # ------------------------------------------------------------------------

    print("\nInitializing CryoEmSimulator...")

    simulator = CryoEmSimulator(
        args.config,
        device=str(device),
    )

    # Number of conformational states
    K = simulator._models.shape[0]

    print(f"Number of states: K = {K}")

    # ------------------------------------------------------------------------
    # Build encoder + NRE
    # ------------------------------------------------------------------------

    print("\nBuilding encoder...")
    print(f"Embedding: {args.embedding}")
    print(f"Embedding dimension: {args.embedding_dim}")

    encoder = EMBEDDING_NETS[args.embedding](
        args.embedding_dim,
        D=image_size,
    )

    nre = NREHead(
        x_dim=args.embedding_dim,
        n_conformations=K,
    )

    model = NREInferenceModel(
        encoder,
        nre,
    )

    # ------------------------------------------------------------------------
    # Load your pretrained_image_embed.pt
    # ------------------------------------------------------------------------

    print(f"\nLoading checkpoint:")
    print(args.full_model)

    checkpoint = torch.load(
        args.full_model,
        map_location="cpu",
    )

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False,
    )

    print("\nCheckpoint loaded.")

    if missing:
        print("\nMissing keys:")
        for key in missing:
            print("  ", key)

    if unexpected:
        print("\nUnexpected keys:")
        for key in unexpected:
            print("  ", key)

    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------------

    confusion = np.zeros(
        (K, K),
        dtype=np.int64,
    )

    accuracies = []

    # ------------------------------------------------------------------------
    # Run each pure state
    # ------------------------------------------------------------------------

    for true_state in range(K):

        print("\n" + "=" * 70)
        print(
            f"TRUE STATE {true_state + 1}/{K}"
        )
        print("=" * 70)

        # ------------------------------------------------------------
        # Force simulator to use this conformation.
        #
        # CryoEmSimulator expects:
        #     indices.shape == (batch_size, 1)
        #     indices.dtype == torch.float32
        #
        # State numbering internally is 0 ... K-1.
        # ------------------------------------------------------------

        indices = torch.full(
            (args.num_sim, 1),
            float(true_state),
            dtype=torch.float32,
            device=device,
        )

        print(
            f"Generating {args.num_sim} synthetic images "
            f"from state S{true_state + 1}..."
        )

        images = simulator.simulate(
            num_sim=args.num_sim,
            indices=indices,
            return_parameters=False,
            batch_size=args.sim_batch_size,
        )

        print(
            f"Generated images: {tuple(images.shape)}"
        )

        # ------------------------------------------------------------
        # NRE
        # ------------------------------------------------------------

        print("Evaluating NRE...")

        log_ratio = evaluate_log_ratios(
            model=model,
            images=images,
            K=K,
            device=device,
            batch_size=args.batch_size,
            pair_batch_size=args.pair_batch_size,
            normalize_images=args.normalize_images,
            flip_contrast=args.flip_contrast,
            skip_rtilde_correction=args.skip_rtilde_correction,
        )

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        predicted = torch.argmax(
            log_ratio,
            dim=1,
        ).numpy()

        accuracy = np.mean(
            predicted == true_state
        )

        accuracies.append(accuracy)

        print(
            f"\nAccuracy for true S{true_state + 1}: "
            f"{100 * accuracy:.2f}%"
        )

        # ------------------------------------------------------------
        # Confusion matrix row
        # ------------------------------------------------------------

        counts = np.bincount(
            predicted,
            minlength=K,
        )

        confusion[true_state, :] = counts

        # ------------------------------------------------------------
        # Log-ratio diagnostics
        # ------------------------------------------------------------

        mean_log_ratio = log_ratio.mean(dim=0).numpy()
        median_log_ratio = log_ratio.median(dim=0).values.numpy()

        print("\nMean log ratios:")
        for state in range(K):
            print(
                f"  S{state + 1}: "
                f"{mean_log_ratio[state]:+.5f}"
            )

        print("\nMedian log ratios:")
        for state in range(K):
            print(
                f"  S{state + 1}: "
                f"{median_log_ratio[state]:+.5f}"
            )

        print("\nPredicted state distribution:")
        for state in range(K):
            pct = 100 * counts[state] / args.num_sim
            print(
                f"  S{state + 1}: "
                f"{counts[state]:6d} "
                f"({pct:6.2f}%)"
            )

        # ------------------------------------------------------------
        # Save log ratios
        # ------------------------------------------------------------

        save_file = (
            output_dir /
            f"log_ratio_true_S{true_state + 1}.pt"
        )

        torch.save(
            log_ratio,
            save_file,
        )

        print(
            f"\nSaved: {save_file}"
        )

        # Free synthetic images before next state
        del images
        del log_ratio

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("FINAL SYNTHETIC VALIDATION RESULTS")
    print("=" * 70)

    print("\nPer-state accuracy:")

    for state, accuracy in enumerate(accuracies):
        print(
            f"  S{state + 1}: "
            f"{100 * accuracy:.2f}%"
        )

    overall_accuracy = np.trace(confusion) / np.sum(confusion)

    print(
        f"\nOverall accuracy: "
        f"{100 * overall_accuracy:.2f}%"
    )

    # ------------------------------------------------------------------------
    # Confusion matrix
    #
    # Rows = TRUE state
    # Columns = PREDICTED state
    # ------------------------------------------------------------------------

    print("\nConfusion matrix (counts):")
    print(confusion)

    confusion_percent = (
        confusion /
        confusion.sum(axis=1, keepdims=True)
        * 100.0
    )

    print("\nConfusion matrix (%):")
    print(
        np.array2string(
            confusion_percent,
            formatter={
                "float_kind": lambda x: f"{x:7.2f}"
            },
        )
    )

    # ------------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------------

    np.save(
        output_dir / "confusion_matrix.npy",
        confusion,
    )

    np.save(
        output_dir / "confusion_matrix_percent.npy",
        confusion_percent,
    )

    # CSV
    np.savetxt(
        output_dir / "confusion_matrix_percent.csv",
        confusion_percent,
        delimiter=",",
        fmt="%.4f",
    )

    summary = {
        "checkpoint": str(args.full_model),
        "config": str(args.config),
        "embedding": args.embedding,
        "embedding_dim": args.embedding_dim,
        "K": K,
        "num_sim_per_state": args.num_sim,
        "overall_accuracy": float(overall_accuracy),
        "per_state_accuracy": [
            float(x) for x in accuracies
        ],
        "normalize_images": args.normalize_images,
        "flip_contrast": args.flip_contrast,
        "skip_rtilde_correction": args.skip_rtilde_correction,
    }

    with open(
        output_dir / "synthetic_validation_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to:\n"
        f"{output_dir}"
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Validate a trained NRE using synthetic cryo-EM "
            "images with known conformational states."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Same simulation/image config used by CryoEmSimulator.",
    )

    parser.add_argument(
        "--embedding",
        default="SPATIAL_CRYO",
        help="Embedding architecture.",
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        required=True,
        help="Embedding dimension used during training.",
    )

    parser.add_argument(
        "--full_model",
        required=True,
        help="Path to pretrained_image_embed.pt.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for synthetic validation results.",
    )

    parser.add_argument(
        "--num_sim",
        type=int,
        default=2000,
        help="Number of synthetic images per true state.",
    )

    parser.add_argument(
        "--sim_batch_size",
        type=int,
        default=1024,
        help="Simulation batch size.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of images per NRE encoder batch.",
    )

    parser.add_argument(
        "--pair_batch_size",
        type=int,
        default=4096,
        help="Number of (image,state) pairs per NRE forward pass.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu.",
    )

    parser.add_argument(
        "--normalize_images",
        action="store_true",
        help="Apply the same per-image normalization as experimental inference.",
    )

    parser.add_argument(
        "--flip_contrast",
        action="store_true",
        help="Flip image contrast before encoding.",
    )

    parser.add_argument(
        "--skip_rtilde_correction",
        action="store_true",
        help=(
            "Skip cross-class NRE correction. "
            "Only use if trained with real/hybrid negatives."
        ),
    )

    args = parser.parse_args()

    main(args)