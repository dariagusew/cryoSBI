"""
noise_generator_ice.py

Inference-only wrapper around the trained GAN-ICE generator.

"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# These must match the constants in train_noise_gan.py exactly.
# ---------------------------------------------------------------------------
_GROUPS = 8


# ===========================================================================
# Generator
# ===========================================================================

class _ResBlockG(nn.Module):
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


class _Generator(nn.Module):
    _DILATIONS = [1, 2, 4, 8, 4, 2, 1, 1]

    def __init__(self, base_channels: int = 64, n_blocks: int = 8):
        super().__init__()
        ch = base_channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, ch, 7, padding=3),
            nn.GroupNorm(_GROUPS, ch),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList([
            _ResBlockG(ch, dilation=self._DILATIONS[i % len(self._DILATIONS)])
            for i in range(n_blocks)
        ])
        self.output_conv = nn.Conv2d(ch, 1, 7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(z)
        for rb in self.res_blocks:
            h = rb(h)
        return self.output_conv(h)


# ===========================================================================
# Public class
# ===========================================================================

class NoiseGeneratorICE:
    """
    Inference wrapper for the trained GAN-ICE ice-noise generator.

    Loads EMA weights from a checkpoint produced by train_noise_gan.py.
    All operations run on *device*; the caller never touches torch directly.

    Parameters
    ----------
    pt_file : path to noise_gan.pt checkpoint
    device  : 'cuda', 'cpu', or None (auto-detect)

    Usage
    -----
    gen     = NoiseGeneratorICE(pt_file="noise_gan.pt", device="cuda")
    patches = gen.sample(n=16, box_size=128)   # (16, 128, 128) numpy float32
    """

    def __init__(
        self,
        pt_file: str | Path,
        device:  Optional[str] = None,
    ):
        self.pt_file = Path(pt_file)
        if not self.pt_file.exists():
            raise FileNotFoundError(f"GAN-ICE checkpoint not found: {self.pt_file}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model = self._load(self.pt_file, self.device)
        self._model.eval()

        print(
            f"  [NoiseGeneratorICE] Loaded '{self.pt_file.name}' "
            f"on {self.device}  "
            f"| params: {sum(p.numel() for p in self._model.parameters()):,}"
        )

    # ── private ──────────────────────────────────────────────────────────

    @staticmethod
    def _load(pt_file: Path, device: torch.device) -> nn.Module:
        """
        Reconstruct the generator from the checkpoint and apply EMA weights.

        The checkpoint stores:
          ckpt["gen_cfg"]   – dict  forwarded to _Generator(**gen_cfg)
          ckpt["ema"]       – dict  name → float32 shadow tensor
        """
        ckpt = torch.load(str(pt_file), map_location=device, weights_only=True)

        gen_cfg: Dict = ckpt.get("gen_cfg", {})
        model   = _Generator(**gen_cfg).to(device)

        # EMA shadow tensors may be float32 on CPU regardless of training device
        ema_sd: Dict[str, torch.Tensor] = ckpt["ema"]
        model_sd = {
            k: v.to(device=device, dtype=next(model.parameters()).dtype)
            for k, v in ema_sd.items()
        }
        model.load_state_dict(model_sd, strict=True)
        return model

    @staticmethod
    def _normalise(x: torch.Tensor) -> torch.Tensor:
        """Per-image zero mean, unit std — identical to training post-processing."""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std  = x.std (dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return (x - mean) / std

    # ── public interface ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n:        int,
        box_size: int,
    ) -> torch.Tensor:
        """
        Generate *n* independent noise patches.

        Parameters
        ----------
        n        : batch size — number of patches to generate
        box_size : side length of each square patch (pixels)

        Returns
        -------
        torch.Tensor, shape (n, box_size, box_size), dtype float32
            Each patch has mean≈0, std≈1.
            Tensor lives on self.device.
        """
        z = torch.randn(n, 1, box_size, box_size, device=self.device)
        return self._normalise(self._model(z)).squeeze(1)   # (n, H, W)

    # ── repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cfg = {
            k: getattr(self._model, k, "?")
            for k in ("base_channels", "n_blocks")
        }
        return (
            f"NoiseGeneratorICE("
            f"checkpoint='{self.pt_file.name}', "
            f"device={self.device}, "
            f"gen_cfg={cfg})"
        )
