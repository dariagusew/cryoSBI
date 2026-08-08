# "infer_populations_nre.py"
# infer_populations_nre.py
"""
Standalone population-weight inference using an NRE estimator trained with
cross-class negatives.

The full-model checkpoint from pretrain_image_embed_v6.py is expected to
contain:
    encoder.*   : the deterministic image encoder
    nre.*       : the auxiliary NRE head

The NRE head outputs r_tilde(d,X) = p(d|X) / p(d|X'!=X).  This script
converts it to the true marginal likelihood ratio:
    r(d,X) = p(d|X) / p(d) = K * r_tilde / (1 + (K-1) * r_tilde)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List

import mrcfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


# ============================================================================
# NRE HEAD (same architecture as in pretrain_image_embed_v6.py)
# ============================================================================

class NREHead(nn.Module):
    """
    Auxiliary Neural Ratio Estimation head operating on the deterministic
    embedding mu. Class labels are passed as raw integer indices.
    """
    def __init__(
        self,
        x_dim: int,
        hidden_features: Tuple[int, ...] = (256, 128, 64),
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        dims = [1 + x_dim] + list(hidden_features) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # theta: (B,) long tensor of class indices
        # x:     (B, D)
        theta_f = theta.float().unsqueeze(-1)  # (B, 1), no embedding
        h = torch.cat([theta_f, x], dim=-1)
        return self.net(h).squeeze(-1)


class NREInferenceModel(nn.Module):
    def __init__(self, encoder: nn.Module, nre: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.nre = nre

    def forward(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # images: (B, 1, H, W) or (B, H, W)
        # theta:  (B,) long
        if images.ndim == 3:
            images = images.unsqueeze(1)
        out = self.encoder.forward_inference(images)
        mu = out[0] if isinstance(out, tuple) else out
        return self.nre(theta, mu)


# ============================================================================
# CONVERSION  r_tilde  ->  true likelihood ratio
# ============================================================================

def log_ratio_from_log_rtilde(log_r_tilde: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert log r_tilde = log p(d|X) - log p(d|X' != X) to
            log r     = log p(d|X) - log p(d)
    using:
        r = K * r_tilde / (r_tilde + K - 1)
    """
    K = float(K)
    logK = math.log(K)

    # Clamp to a safe range for float32 exp()
    x = torch.clamp(log_r_tilde, -20.0, 40.0)

    log_num = logK + x
    log_den = torch.log((K - 1.0) + torch.exp(x))
    log_ratio = log_num - log_den

    # Asymptotic corrections outside the clamped range
    log_ratio = torch.where(
        log_r_tilde > 40.0,
        torch.tensor(logK, dtype=log_ratio.dtype, device=log_ratio.device),
        log_ratio,
    )
    log_ratio = torch.where(
        log_r_tilde < -20.0,
        torch.tensor(logK - math.log(K - 1.0), dtype=log_ratio.dtype, device=log_ratio.device)
        + log_r_tilde,
        log_ratio,
    )
    return log_ratio

# ============================================================================
# LIKELIHOOD MATRIX EVALUATION (memory-mapped images)
# ============================================================================

@torch.no_grad()
def evaluate_log_ratios(
    model: NREInferenceModel,
    mrc_path: str,
    K: int,
    device: str,
    image_batch_size: int = 256,
    pair_batch_size: int = 4096,
    normalize_images: bool = False,
    flip_contrast: bool = False,
) -> torch.Tensor:
    """
    Returns log-ratio matrix of shape [N_images, K] using memory-mapped .mrc
    images.  The full image stack is never loaded into host memory.
    """
    model.eval()
    model.to(device)

    log_ratio_rows = []

    with mrcfile.mmap(mrc_path, mode="r") as mrc:
        N_images = mrc.data.shape[0]
        print(f"Memory-mapped stack: {N_images} images, shape {mrc.data.shape[1:]}")

        theta_all = torch.arange(K, dtype=torch.long, device=device)

        for start in tqdm(range(0, N_images, image_batch_size), desc="Image chunks"):
            end = min(start + image_batch_size, N_images)

            # Load only this chunk from disk
            imgs_np = mrc.data[start:end].copy()
            if flip_contrast:
                imgs_np = -imgs_np
            imgs = torch.from_numpy(imgs_np).float()

            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(1)  # (B, 1, H, W)

            if normalize_images:
                mu = imgs.mean(dim=(-1, -2), keepdim=True)
                std = imgs.std(dim=(-1, -2), keepdim=True)
                imgs = (imgs - mu) / (std + 1e-8)

            imgs = imgs.to(device)

            # Deterministic embeddings for the chunk
            out = model.encoder.forward_inference(imgs)
            mu_chunk = out[0] if isinstance(out, tuple) else out  # (B, D)
            B, D = mu_chunk.shape

            # Build all (image, class) pairs
            mu_rep = mu_chunk.unsqueeze(1).expand(-1, K, -1).reshape(B * K, D)
            theta_rep = theta_all.unsqueeze(0).expand(B, -1).reshape(-1)

            # Evaluate NRE head in pair-batches to keep GPU memory bounded
            log_r_tilde = torch.empty(B * K, dtype=torch.float32, device="cpu")
            for p_start in range(0, B * K, pair_batch_size):
                p_end = min(p_start + pair_batch_size, B * K)
                batch_theta = theta_rep[p_start:p_end]
                batch_mu = mu_rep[p_start:p_end].to(device)
                log_r_tilde[p_start:p_end] = model.nre(batch_theta, batch_mu).cpu().float()

            log_r_tilde = log_r_tilde.view(B, K)
            log_ratio_chunk = log_ratio_from_log_rtilde(log_r_tilde, K)
            log_ratio_rows.append(log_ratio_chunk.cpu())

    return torch.cat(log_ratio_rows, dim=0)  # [N_images, K]


# ============================================================================
# WEIGHT OPTIMIZERS (copied from infer_populations.py)
# ============================================================================

class WeightOptimizer:
    def __init__(
        self,
        log_p: np.ndarray,
        w0: Optional[np.ndarray] = None,
        theta: float = 0.0,
        device: str = "cpu",
    ):
        self.device = device
        self.log_p = torch.tensor(log_p, dtype=torch.float64, device=device)
        self.log_p = self.log_p - self.log_p.max()  # global max for stability
        self.n_j, self.n_i = self.log_p.shape

        if w0 is None:
            self.w0 = torch.ones(self.n_j, dtype=torch.float64, device=device) / self.n_j
            if theta > 0:
                print(f"Prior weights w0 not specified, using uniform: w0 = 1/{self.n_j}")
        else:
            self.w0 = torch.tensor(w0, dtype=torch.float64, device=device)

        self.theta = torch.tensor(theta, dtype=torch.float64, device=device)

    def compute_loss(self, w: torch.Tensor) -> torch.Tensor:
        eps = 1e-15
        log_w = torch.log(w + eps)
        log_terms = log_w.unsqueeze(1) + self.log_p
        term1 = -torch.logsumexp(log_terms, dim=0).sum() / self.n_i

        if self.theta > 0:
            log_w0 = torch.log(self.w0 + eps)
            term2 = self.theta * torch.sum(w * (log_w - log_w0)) / self.n_j
        else:
            term2 = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        return term1 + term2

    def optimize(
        self,
        lr: float = 0.1,
        max_iter: int = 10000,
        tol: float = 1e-9,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[float]]:
        z_init = torch.randn(self.n_j, dtype=torch.float64, device=self.device)
        z = z_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        losses = []
        for iteration in range(max_iter):
            optimizer.zero_grad()
            w = torch.softmax(z, dim=0)
            loss = self.compute_loss(w)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Loss = {loss.item():.8f}")

            if iteration > 10 and abs(losses[-1] - losses[-2]) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

        with torch.no_grad():
            w_opt = torch.softmax(z, dim=0)

        return w_opt.cpu().numpy(), losses


class WeightOptimizerLBFGS(WeightOptimizer):
    def optimize(
        self,
        max_iter: int = 100,
        tol: float = 1e-9,
        verbose: bool = False,
        history_size: int = 100,
    ) -> Tuple[np.ndarray, List[float]]:
        z_init = torch.randn(self.n_j, dtype=torch.float64, device=self.device)
        z = z_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [z],
            history_size=history_size,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        losses = []
        for iteration in range(max_iter):
            def closure():
                optimizer.zero_grad()
                w = torch.softmax(z, dim=0)
                loss = self.compute_loss(w)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            losses.append(loss.item())

            if verbose:
                print(f"Iter {iteration}: Loss = {loss.item():.8f}")

            if iteration > 0 and abs(losses[-1] - losses[-2]) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

        with torch.no_grad():
            w_opt = torch.softmax(z, dim=0)

        return w_opt.cpu().numpy(), losses


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    if not torch.cuda.is_available() and "cuda" in args.device:
        print(f"CUDA not available. Switching device from '{args.device}' to 'cpu'.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load configs
    with open(args.image_config) as f:
        image_config = json.load(f)

    embedding_name = args.embedding
    embedding_dim = args.embedding_dim
    image_size = image_config["N_PIXELS"]

    # Load 3D models (only needed for K)
    print(f"Loading 3D models from {args.models_file}")
    models = torch.load(args.models_file).to(device)
    K = models.shape[0]
    print(f"Loaded {K} models")

    # Build encoder and NRE head
    print(f"Building encoder: {embedding_name}, dim={embedding_dim}")
    encoder = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size)
    nre = NREHead(x_dim=embedding_dim)
    inf_model = NREInferenceModel(encoder, nre).to(device)

    # Load full-model checkpoint
    print(f"Loading full-model checkpoint from {args.full_model}")
    ckpt = torch.load(args.full_model, map_location=device)
    inf_model.load_state_dict(ckpt, strict=False)
    inf_model.eval()

    # Evaluate log p(d|X) - log p(d) for all images and classes
    print("Evaluating NRE log-ratio matrix...")
    log_ratio_matrix = evaluate_log_ratios(
        model=inf_model,
        mrc_path=args.image_stack,
        K=K,
        device=device,
        image_batch_size=args.batch_size_images,
        pair_batch_size=args.batch_size_pairs,
        normalize_images=args.normalize_images,
        flip_contrast=args.flip_contrast,
    )
    print(f"Log-ratio matrix shape: {log_ratio_matrix.shape}")

    # Transpose to [K, N_images] as expected by WeightOptimizer
    log_ratio_matrix = log_ratio_matrix.T.numpy()  # [K, N_images]

    # Optional: save likelihood matrix
    if args.log_likelihood_file is not None:
        print(f"Saving log-ratio matrix to {args.log_likelihood_file}")
        torch.save(torch.from_numpy(log_ratio_matrix), args.log_likelihood_file)

    # Optimize weights
    print("Optimizing population weights...")
    adam_opt = WeightOptimizer(log_ratio_matrix, theta=args.theta, device=device)
    lbfgs_opt = WeightOptimizerLBFGS(log_ratio_matrix, theta=args.theta, device=device)

    print("\nOptimizing with Adam")
    w_adam, losses_adam = adam_opt.optimize(
        lr=args.lr, max_iter=args.max_iter, tol=args.tol, verbose=True
    )
    print(f"Adam converged in {len(losses_adam)} iterations")
    print(f"Final Adam weights: {w_adam}")
    print(f"Final Adam loss:    {losses_adam[-1]:.8f}")

    print("\nOptimizing with L-BFGS")
    w_lbfgs, losses_lbfgs = lbfgs_opt.optimize(
        max_iter=args.max_iter_lbfgs, tol=args.tol, verbose=True
    )
    print(f"L-BFGS converged in {len(losses_lbfgs)} iterations")
    print(f"Final L-BFGS weights: {w_lbfgs}")
    print(f"Final L-BFGS loss:    {losses_lbfgs[-1]:.8f}")

    w_opt = w_adam if losses_adam[-1] < losses_lbfgs[-1] else w_lbfgs
    print(f"\nSelected optimal weights:\n{w_opt}")

    print(f"Saving weights to {args.output_file}")
    torch.save(torch.from_numpy(w_opt).float(), args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer population weights from real cryo-EM images using a pretrained NRE estimator."
    )

    parser.add_argument("--image_config",        required=True,             help="Path to image config JSON")
    parser.add_argument("--embedding",           default="SPATIAL_CRYO",    help="Embedding architecture name")
    parser.add_argument("--embedding_dim",       type=int,   default=16,    help="Encoder output dimension")
    parser.add_argument("--full_model",          required=True,             help="Path to full_model checkpoint from pretraining")
    parser.add_argument("--models_file",         required=True,             help="Path to 3D models .pt file")
    parser.add_argument("--image_stack",         required=True,             help="Path to experimental .mrc image stack")
    parser.add_argument("--output_file",         required=True,             help="Path to save optimized weights")
    parser.add_argument("--log_likelihood_file", default=None,              help="Optional path to save log-ratio matrix")
    parser.add_argument("--device",              default="cuda",            help="Compute device")
    parser.add_argument("--batch_size_images",   type=int, default=256,     help="Images processed per chunk")
    parser.add_argument("--batch_size_pairs",    type=int, default=4096,    help="(image, class) pairs per forward pass")
    parser.add_argument("--theta",               type=float, default=0.0,   help="Regularization strength")
    parser.add_argument("--lr",                  type=float, default=0.1,   help="Adam learning rate")
    parser.add_argument("--max_iter",            type=int, default=1000,    help="Adam max iterations")
    parser.add_argument("--max_iter_lbfgs",      type=int, default=100,     help="L-BFGS max iterations")
    parser.add_argument("--tol",                 type=float, default=1e-10, help="Convergence tolerance")
    parser.add_argument("--normalize_images",    action="store_true",       help="Per-image mean/std normalize (as in pretrain validation)")
    parser.add_argument("--flip_contrast",       action="store_true",       help="Flip contrast of mrc images")

    args = parser.parse_args()
    main(args)
