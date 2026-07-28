"""
noise_generator_hpf.py

Inference-only wrapper around the trained GAN-HPF generator.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ===========================================================================
# Generator Architecture
# ===========================================================================

class SpatialNoiseGenerator(nn.Module):
    """
    Lightweight U-Net that filters spatial Gaussian white noise into 
    structured cryo-EM background noise (ice, detector MTF, carbon edges).
    """
    def __init__(self, base_ch=32):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, base_ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1), nn.BatchNorm2d(base_ch*2), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1), nn.BatchNorm2d(base_ch*4), nn.LeakyReLU(0.2, inplace=True))
        
        # Decoder
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1), nn.BatchNorm2d(base_ch*2), nn.LeakyReLU(0.2, inplace=True))
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*4, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.LeakyReLU(0.2, inplace=True))
        
        # Final Output Layer (base_ch*2 because of the skip connection from e1)
        self.final = nn.Conv2d(base_ch*2, 1, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        d1 = self.dec1(e3)
        d1 = torch.cat([d1, e2], dim=1) # Skip connection
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e1], dim=1) # Skip connection
        
        return self.final(d2)


# ===========================================================================
# Public wrapper class
# ===========================================================================

class NoiseGeneratorHPF:
    """
    Inference wrapper for the trained GAN-HPF ice-noise generator.

    Loads weights from a checkpoint produced by train_noise_gan_with_val.py.
    All operations run on *device*.

    Parameters
    ----------
    pt_file : path to pretrained_noise_generator.pt checkpoint
    device  : 'cuda', 'cpu', or None (auto-detect)
    base_ch : Base channels of the U-Net (must match training, default 32)
    """

    def __init__(
        self,
        pt_file: str | Path,
        device:  Optional[str] = None,
        base_ch: int = 32
    ):
        self.pt_file = Path(pt_file)
        if not self.pt_file.exists():
            raise FileNotFoundError(f"GAN-HPF checkpoint not found: {self.pt_file}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.base_ch = base_ch
        self._model = self._load(self.pt_file, self.device, self.base_ch)
        self._model.eval()

        print(
            f"  [NoiseGeneratorHPF] Loaded '{self.pt_file.name}' "
            f"on {self.device}  "
            f"| params: {sum(p.numel() for p in self._model.parameters()):,}"
        )

    # ── private ──────────────────────────────────────────────────────────

    @staticmethod
    def _load(pt_file: Path, device: torch.device, base_ch: int) -> nn.Module:
        """
        Reconstruct the generator and load the state dictionary.
        """
        model = SpatialNoiseGenerator(base_ch=base_ch).to(device)
        state_dict = torch.load(str(pt_file), map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    # ── public interface ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        """
        Process spatial Gaussian white noise through the U-Net to generate 
        the structural ML noise residual.

        Parameters
        ----------
        z : torch.Tensor, shape (B, 1, H, W)
            Spatial Gaussian white noise tensor on self.device.

        Returns
        -------
        torch.Tensor, shape (B, 1, H, W)
            The generated noise residual.
        """
        return self._model(z)

    # ── repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NoiseGeneratorReal("
            f"checkpoint='{self.pt_file.name}', "
            f"device={self.device}, "
            f"base_ch={self.base_ch})"
        )
