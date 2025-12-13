"""
pretrain_spatial_cryo_v4.py

Unsupervised pre-training of SPATIAL_CRYO encoder.
- Mode 1 (no real images): Reconstruction only (like v3)
- Mode 2 (with real images): Reconstruction + Domain adaptation

Usage (reconstruction only):
    python pretrain_spatial_cryo.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 256 \
        --epochs 100 \
        --batch_size 512 \
        --simulation_batch_size 1024

Usage (with domain adaptation):
    python pretrain_spatial_cryo.py \
        --image_config config.json \
        --real_images real_data.mrc \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 256 \
        --epochs 100 \
        --batch_size 512 \
        --simulation_batch_size 1024 \
        --lambda_domain 0.1
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from itertools import islice
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not installed. Real image loading disabled.")
    print("Install with: pip install mrcfile")

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator

# ============================================================================
# MRC FILE HANDLING
# ============================================================================

def check_mrc_file_size(filepath):
    """Check MRC file size in bytes and GB."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    file_size_gb = file_size / (1024**3)
    return file_size, file_size_gb


def validate_mrc_data(data):
    """Validate MRC data after reading."""
    if data is None:
        return False, "Data is None"
    if data.size == 0:
        return False, "Data is empty"
    if data.ndim not in [2, 3]:
        return False, f"Invalid dimensions: {data.ndim}D"
    try:
        if np.all(data == 0):
            return False, "All data is zero"
        if np.any(np.isnan(data)):
            return False, "Data contains NaN"
        if np.any(np.isinf(data)):
            return False, "Data contains inf"
        if np.std(data) == 0:
            return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"


def read_mrc_header_raw(filepath):
    """Read MRC header manually."""
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(1024)
            if len(header_bytes) < 1024:
                return None
            import struct
            nx, ny, nz = struct.unpack('iii', header_bytes[0:12])
            mode = struct.unpack('i', header_bytes[12:16])[0]
            return {'nx': nx, 'ny': ny, 'nz': nz, 'mode': mode, 'header_size': 1024}
    except:
        return None


def get_dtype_from_mode(mode):
    """Convert MRC mode to numpy dtype."""
    dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    return dtype_map.get(mode, np.float32)


def validate_mrc_dimensions(nx, ny, nz):
    """Check if dimensions are reasonable."""
    if nx <= 0 or ny <= 0 or nz <= 0:
        return False, f"Non-positive: {nz}×{ny}×{nx}"
    if nx > 8192 or ny > 8192:
        return False, f"Too large: {ny}×{nx}"
    if nz > 50000000:
        return False, f"Stack too large: {nz}"
    return True, "Valid"


def open_mrc_robust(filepath, max_size_gb=None):
    """Robustly open MRC file with fallback methods."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None, False, "File not found"
    
    file_size, file_size_gb = check_mrc_file_size(filepath)
    if max_size_gb is not None and file_size_gb > max_size_gb:
        return None, False, f"Too large: {file_size_gb:.2f} GB"
    
    # Method 1: Standard
    try:
        with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
            if mrc.data is not None and mrc.data.size > 0:
                is_valid, msg = validate_mrc_data(mrc.data)
                if is_valid:
                    data = np.array(mrc.data) if file_size_gb < 1.0 else mrc.data
                    return data, True, "Standard"
    except:
        pass
    
    # Method 2: Force-read
    try:
        header_info = read_mrc_header_raw(filepath)
        if header_info is not None:
            nx, ny, nz, mode = header_info['nx'], header_info['ny'], header_info['nz'], header_info['mode']
            is_valid, msg = validate_mrc_dimensions(nx, ny, nz)
            if not is_valid:
                return None, False, msg
            
            dtype = get_dtype_from_mode(mode)
            data = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            
            is_valid, msg = validate_mrc_data(data[0])
            if is_valid:
                return data, True, f"Force-read memmap"
    except Exception as e:
        return None, False, f"Failed: {str(e)[:100]}"
    
    return None, False, "All methods failed"


# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    Forward: identity
    Backward: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for domain adversarial training"""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """Dynamically adjust reversal strength during training"""
        self.lambda_ = lambda_


# ============================================================================
# DOMAIN DISCRIMINATOR
# ============================================================================

class DomainDiscriminator(nn.Module):
    """
    Discriminates between synthetic and real embeddings
    Binary classifier: 0 = synthetic, 1 = real
    """
    def __init__(self, embedding_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1)  # Binary output
        )
    
    def forward(self, embedding, reverse_gradient=True):
        """
        Args:
            embedding: [B, embedding_dim]
            reverse_gradient: whether to apply gradient reversal
        Returns:
            logits: [B, 1] (0=synthetic, 1=real)
        """
        if reverse_gradient:
            embedding = self.grl(embedding)
        return self.network(embedding)
    
    def set_lambda(self, lambda_):
        """Adjust gradient reversal strength"""
        self.grl.set_lambda(lambda_)


# ============================================================================
# DECODER
# ============================================================================

class SpatialCryoDecoder(nn.Module):
    """
    Lightweight decoder that mirrors SPATIAL_CRYO encoder
    All-convolutional design, symmetric architecture
    """
    def __init__(self, embedding_dim, image_size):
        super().__init__()
        
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        
        # Calculate upsampling stages (mirror encoder)
        import math
        n_stages = int(math.log2(image_size)) - 2
        
        # Start from 4x4 (where encoder ends)
        start_size = 4
        
        # Initial channel count (mirror encoder's final channels)
        start_channels = 16 * (2 ** (n_stages - 1))
        
        # Lightweight FC: only project to channel dimension
        self.fc = nn.Linear(embedding_dim, start_channels)
        
        # Expand 1x1 to 4x4
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(start_channels, start_channels,
                             kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(start_channels),
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling (mirror encoder stages in reverse)
        layers = []
        in_channels = start_channels
        
        for i in range(n_stages):
            out_channels = in_channels // 2
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels,
                                 kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Final conv to get single channel
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: [B, embedding_dim]
        Returns:
            reconstruction: [B, 1, H, W]
        """
        B = z.shape[0]
        
        # Project to channel dimension
        x = self.fc(z)
        
        # Reshape to [B, start_channels, 1, 1]
        x = x.view(B, -1, 1, 1)
        
        # Expand to 4x4
        x = self.initial_conv(x)
        
        # Upsample to full size
        x = self.decoder(x)
        
        return x


# ============================================================================
# MODEL WRAPPER
# ============================================================================

class AdaptivePretrainModel(nn.Module):
    """
    SPATIAL_CRYO encoder + decoder + optional domain discriminator
    """
    def __init__(self, embedding_name, embedding_dim, image_size, use_domain_adaptation=False):
        super().__init__()
        
        self.embedding_name = embedding_name
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.use_domain_adaptation = use_domain_adaptation
        
        # Create encoder
        if embedding_name not in EMBEDDING_NETS:
            raise ValueError(
                f"{embedding_name} not found in EMBEDDING_NETS!\n"
                f"Available: {list(EMBEDDING_NETS.keys())}"
            )
        
        self.encoder = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size)
        self.decoder = SpatialCryoDecoder(embedding_dim, image_size)
        
        print(f"  Encoder: {embedding_name} (D={image_size})")
        print(f"  Decoder: SpatialCryoDecoder (symmetric)")
        
        if use_domain_adaptation:
            self.discriminator = DomainDiscriminator(embedding_dim, hidden_dim=256, dropout=0.2)
            print(f"  Discriminator: DomainDiscriminator (domain adaptation)")
        else:
            self.discriminator = None
            print(f"  Discriminator: None (reconstruction only)")
    
    def forward(self, x, return_domain_pred=False):
        """
        Args:
            x: [B, H, W]
            return_domain_pred: whether to return domain prediction
        Returns:
            embeddings: [B, embedding_dim]
            reconstruction: [B, 1, H, W]
            domain_pred: [B, 1] (if return_domain_pred=True and discriminator exists)
        """
        embeddings = self.encoder(x)
        reconstruction = self.decoder(embeddings)
        
        if return_domain_pred and self.discriminator is not None:
            domain_pred = self.discriminator(embeddings, reverse_gradient=True)
            return embeddings, reconstruction, domain_pred
        
        return embeddings, reconstruction
    
    def set_grl_lambda(self, lambda_):
        """Adjust gradient reversal strength"""
        if self.discriminator is not None:
            self.discriminator.set_lambda(lambda_)


# ============================================================================
# REAL IMAGE DATASET (MRC LOADER)
# ============================================================================

class RealImageMRCDataset(Dataset):
    """
    Dataset for loading real images from MRC stack
    Efficient streaming without loading all into memory
    """
    def __init__(self, mrc_path, cache_size=10000):
        """
        Args:
            mrc_path: Path to MRC file
            cache_size: Number of images to keep in memory cache
        """
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile not installed. Install with: pip install mrcfile")
        
        self.mrc_path = mrc_path
        self.cache_size = cache_size
        
        # Open MRC file ONCE and keep persistent reference
        print(f"  Opening MRC file: {mrc_path}")
        self.mrc_data, success, method = open_mrc_robust(mrc_path)
        
        if not success:
            raise RuntimeError(f"Failed to open MRC file: {method}")
        
        self.n_images = self.mrc_data.shape[0]
        self.image_shape = self.mrc_data.shape[1:]
        
        print(f"  Loaded MRC: {self.n_images} images of shape {self.image_shape}")
        print(f"  Loading method: {method}")
        
        # Cache for recently accessed images
        self.cache = {}
        self.cache_order = []
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        """Load a single image"""
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load from persistent MRC reference (no file opening!)
        img = self.mrc_data[idx].astype(np.float32)
        
        # Normalize (zero mean, unit std)
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # Add to cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[idx] = torch.FloatTensor(img)
        self.cache_order.append(idx)
        
        return self.cache[idx]
    
    def __del__(self):
        """Clean up MRC file reference"""
        # If mrc_data is a memmap or file handle, ensure it's properly closed
        if hasattr(self, 'mrc_data'):
            if isinstance(self.mrc_data, np.memmap):
                del self.mrc_data

def create_real_image_loader(mrc_path, batch_size=256, num_workers=4):
    """
    Create dataloader for real images
    
    Args:
        mrc_path: Path to MRC stack
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        DataLoader
    """
    dataset = RealImageMRCDataset(mrc_path, cache_size=10000)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Random sampling
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    return dataloader


# ============================================================================
# UTILITIES
# ============================================================================

def check_embedding_health(embeddings, device):
    """Check if embeddings are diverse"""
    with torch.no_grad():
        emb_std = embeddings.std(dim=0).mean().item()
        
        if len(embeddings) > 1:
            dists = torch.cdist(embeddings, embeddings)
            off_diag = dists[~torch.eye(len(embeddings), dtype=bool, device=device)]
            emb_dist = off_diag.mean().item()
        else:
            emb_dist = 0.0
    
    return emb_std, emb_dist


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())
    
    result = {
        'total': total,
        'trainable': trainable,
        'encoder': encoder,
        'decoder': decoder,
    }
    
    if model.discriminator is not None:
        result['discriminator'] = sum(p.numel() for p in model.discriminator.parameters())
    
    return result


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def pretrain_spatial_cryo(
    image_config_path: str,
    real_images_path: str = None,
    embedding_name: str = 'SPATIAL_CRYO_FFT_FILTER',
    device: str = 'cuda',
    embedding_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-4,
    simulation_batch_size: int = 1024,
    save_path: str = 'pretrained_encoder.pt',
    check_frequency: int = 5,
    n_batches_per_epoch: int = 100,
    l2_weight: float = 0.0,
    lambda_domain: float = 0.1,
):
    """
    Unsupervised pre-training with optional domain adaptation
    
    Args:
        image_config_path: Path to image config JSON
        real_images_path: Path to real images MRC file (optional, None = no domain adaptation)
        embedding_name: Name of embedding architecture
        device: 'cuda', 'cuda:0', 'cuda:1', or 'cpu'
        embedding_dim: Output dimension of embedding
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        simulation_batch_size: Batch size for image simulation
        save_path: Where to save pretrained weights
        check_frequency: How often to print detailed stats
        n_batches_per_epoch: Number of simulation batches per epoch
        l2_weight: Weight for L2 regularization on embeddings
        lambda_domain: Weight for domain adversarial loss (only used if real_images_path provided)
    
    Returns:
        model: Trained model
        final_loss: Final total loss
    """
    
    use_domain_adaptation = real_images_path is not None
    
    print("\n" + "="*70)
    if use_domain_adaptation:
        print(f"DOMAIN ADAPTIVE PRETRAINING: {embedding_name}")
    else:
        print(f"RECONSTRUCTION PRETRAINING: {embedding_name}")
    print("="*70)
    
    # Load image config
    image_config = json.load(open(image_config_path))
    image_size = image_config["N_PIXELS"]
    
    # Load conformational models
    print("\nLoading conformational models...")
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(device).float()
    
    n_conformations = len(models)
    print(f"  Number of conformations: {n_conformations}")
    print(f"  Image size: {image_size}x{image_size}")
    
    # Load real images if provided
    real_loader = None
    real_loader_iter = None
    if use_domain_adaptation:
        print("\nLoading real images...")
        try:
            real_loader = create_real_image_loader(
                real_images_path, 
                batch_size=simulation_batch_size,
                num_workers=4
            )
            real_loader_iter = iter(real_loader)
        except Exception as e:
            print(f"❌ Error loading real images: {e}")
            print("Falling back to reconstruction-only mode")
            use_domain_adaptation = False
    else:
        print("\nNo real images provided - reconstruction only mode")
    
    # Build model
    print(f"\nBuilding model with {embedding_name}...")
    try:
        model = AdaptivePretrainModel(
            embedding_name, 
            embedding_dim, 
            image_size, 
            use_domain_adaptation=use_domain_adaptation
        ).to(device)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return None, 0.0
    
    model.train()
    
    # Configure BatchNorm for stability
    print("\nConfiguring training...")
    print("  Setting BatchNorm momentum = 0.01")
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.momentum = 0.01
    
    # Count parameters
    params = count_parameters(model)
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Encoder parameters: {params['encoder']:,}")
    print(f"  Decoder parameters: {params['decoder']:,}")
    if 'discriminator' in params:
        print(f"  Discriminator parameters: {params['discriminator']:,}")
    
    # Setup optimizers
    if use_domain_adaptation:
        optimizer_main = optim.AdamW(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=lr, weight_decay=0.01
        )
        optimizer_discriminator = optim.AdamW(
            model.discriminator.parameters(),
            lr=lr * 2,  # Discriminator can learn faster
            weight_decay=0.01
        )
    else:
        optimizer_main = optim.AdamW(
            model.parameters(),
            lr=lr, weight_decay=0.01
        )
        optimizer_discriminator = None
    
    # Setup data generation
    print("  Setting up data generation...")
    image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    prior_loader = PriorLoader(
        image_prior, 
        batch_size=simulation_batch_size, 
        num_workers=4
    )
    
    num_pixels = torch.tensor(image_config["N_PIXELS"], dtype=torch.float32, device=device)
    pixel_size = torch.tensor(image_config["PIXEL_SIZE"], dtype=torch.float32, device=device)
    voltage = image_config.get("VOLTAGE", 300.0)
    cs = image_config.get("SPHERICAL_ABERRATION", 0.0)
 
    print("\nTraining configuration:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Embedding dimension: {embedding_dim}")
    if use_domain_adaptation:
        print(f"  Task: Reconstruction + Domain Adaptation")
        print(f"  Domain adversarial weight: {lambda_domain}")
    else:
        print(f"  Task: Reconstruction Only")
    print(f"  L2 regularization weight: {l2_weight}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Simulation batch size: {simulation_batch_size}")
    print(f"  Batches per epoch: {n_batches_per_epoch}")
    print(f"  Samples per epoch: {n_batches_per_epoch * simulation_batch_size:,}")
    print("="*70)
    
    # Training history
    history = {
        'loss': [],
        'recon_loss': [],
        'l2_loss': [],
        'emb_std': [],
        'emb_dist': []
    }
    
    if use_domain_adaptation:
        history['domain_loss'] = []
        history['disc_loss'] = []
        history['disc_accuracy'] = []
    
    # Training loop
    print("\nStarting training...\n")
    
    with tqdm(range(epochs), desc="Pretraining") as tq:
        for epoch in tq:
            
            model.train()
            
            # Adjust gradient reversal lambda (progressive schedule) if using domain adaptation
            if use_domain_adaptation:
                p = float(epoch) / epochs
                lambda_grl = 2. / (1. + np.exp(-10 * p)) - 1
                model.set_grl_lambda(lambda_grl)
            
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_l2_loss = 0
            epoch_domain_loss = 0
            epoch_disc_loss = 0
            epoch_disc_acc = 0
            n_batches = 0
            
            # Train on multiple simulation batches per epoch
            for parameters in islice(prior_loader, n_batches_per_epoch):
                (indices, quaternions, res, shift, defocus, b_factor, amp, snr) = parameters
                
                # Simulate images (inherently noisy from SNR)
                images = cryo_em_simulator(
                    models,
                    indices.to(device, non_blocking=True),
                    quaternions.to(device, non_blocking=True),
                    res.to(device, non_blocking=True),
                    shift.to(device, non_blocking=True),
                    defocus.to(device, non_blocking=True),
                    b_factor.to(device, non_blocking=True),
                    amp.to(device, non_blocking=True),
                    snr.to(device, non_blocking=True),
                    num_pixels,
                    pixel_size,
                    voltage,
                    cs
                )

                # Sample real images ONCE per simulation batch (if using domain adaptation)
                if use_domain_adaptation:
                    try:
                        real_images_full = next(real_loader_iter)
                    except StopIteration:
                        real_loader_iter = iter(real_loader)
                        real_images_full = next(real_loader_iter)
                    
                    real_images_full = real_images_full.to(device, non_blocking=True)
                
                # Split into mini-batches (with aligned synthetic/real pairs)
                synthetic_batches = images.split(batch_size)
                
                if use_domain_adaptation:
                    real_batches = real_images_full.split(batch_size)
                else:
                    real_batches = [None] * len(synthetic_batches)
                
                # Train on mini-batches
                for batch_images, real_images in zip(synthetic_batches, real_batches):
                    
                    if use_domain_adaptation:
                        # ==========================================
                        # Phase 1: Update discriminator
                        # ==========================================
                        optimizer_discriminator.zero_grad()
                        
                        # Encode both domains (stop gradient to encoder)
                        with torch.no_grad():
                            emb_syn = model.encoder(batch_images)
                            emb_real = model.encoder(real_images)
                        
                        # Discriminator predictions (no gradient reversal)
                        pred_syn = model.discriminator(emb_syn, reverse_gradient=False)
                        pred_real = model.discriminator(emb_real, reverse_gradient=False)
                        
                        # Labels: 0 = synthetic, 1 = real
                        labels_syn = torch.zeros_like(pred_syn)
                        labels_real = torch.ones_like(pred_real)
                        
                        # Discriminator loss
                        disc_loss = (
                            F.binary_cross_entropy_with_logits(pred_syn, labels_syn) +
                            F.binary_cross_entropy_with_logits(pred_real, labels_real)
                        ) / 2
                        
                        disc_loss.backward()
                        optimizer_discriminator.step()
                        
                        # Discriminator accuracy
                        disc_acc = (
                            ((pred_syn < 0).float().mean() +  # Correct if < 0 (synthetic)
                             (pred_real > 0).float().mean())  # Correct if > 0 (real)
                        ) / 2
                    
                    # ==========================================
                    # Phase 2: Update encoder + decoder
                    # ==========================================
                    optimizer_main.zero_grad()
                    
                    # Forward pass on synthetic (with reconstruction)
                    embeddings_syn, reconstruction = model(batch_images, return_domain_pred=False)
                    
                    # Reconstruction loss (simple spatial MSE)
                    recon_loss = F.mse_loss(reconstruction.squeeze(1), batch_images)
                    
                    # L2 regularization on embeddings
                    l2_loss = (embeddings_syn ** 2).mean()
                    
                    # Total loss starts with reconstruction + L2
                    loss = recon_loss + l2_weight * l2_loss
                    
                    # Add domain adversarial loss if using domain adaptation
                    if use_domain_adaptation:
                        # Forward pass for domain adaptation
                        embeddings_syn_da = model.encoder(batch_images)
                        embeddings_real = model.encoder(real_images)
                        
                        # Domain adversarial loss (with gradient reversal)
                        domain_pred_syn = model.discriminator(embeddings_syn_da, reverse_gradient=True)
                        domain_pred_real = model.discriminator(embeddings_real, reverse_gradient=True)
                        
                        # Encoder wants: synthetic predicted as real, real as synthetic
                        domain_loss = (
                            F.binary_cross_entropy_with_logits(
                                domain_pred_syn, torch.ones_like(domain_pred_syn)
                            ) +
                            F.binary_cross_entropy_with_logits(
                                domain_pred_real, torch.zeros_like(domain_pred_real)
                            )
                        ) / 2
                        
                        loss = loss + lambda_domain * domain_loss
                        epoch_domain_loss += domain_loss.item()
                        epoch_disc_loss += disc_loss.item()
                        epoch_disc_acc += disc_acc.item()
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_main.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_l2_loss += l2_loss.item()
                    n_batches += 1
            
            # Epoch statistics
            avg_loss = epoch_loss / n_batches
            avg_recon_loss = epoch_recon_loss / n_batches
            avg_l2_loss = epoch_l2_loss / n_batches
            
            history['loss'].append(avg_loss)
            history['recon_loss'].append(avg_recon_loss)
            history['l2_loss'].append(avg_l2_loss)
            
            # Update progress bar
            if use_domain_adaptation:
                avg_domain_loss = epoch_domain_loss / n_batches
                avg_disc_loss = epoch_disc_loss / n_batches
                avg_disc_acc = epoch_disc_acc / n_batches
                
                history['domain_loss'].append(avg_domain_loss)
                history['disc_loss'].append(avg_disc_loss)
                history['disc_accuracy'].append(avg_disc_acc)
                
                tq.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    recon=f"{avg_recon_loss:.6f}",
                    domain=f"{avg_domain_loss:.6f}",
                    disc_acc=f"{avg_disc_acc:.3f}"
                )
            else:
                tq.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    recon=f"{avg_recon_loss:.6f}",
                    l2=f"{avg_l2_loss:.4f}"
                )
            
            # Detailed check every N epochs
            if epoch % check_frequency == 0:
                model.eval()
                with torch.no_grad():
                    # Check on last batch
                    test_imgs = batch_images[:20]
                    test_embs, test_recon = model(test_imgs)
                    
                    emb_std, emb_dist = check_embedding_health(test_embs, device)
                    recon_error = F.mse_loss(test_recon.squeeze(1), test_imgs).item()
                    test_l2 = (test_embs ** 2).mean().item()
                
                history['emb_std'].append(emb_std)
                history['emb_dist'].append(emb_dist)
                
                print(f"\n  Epoch {epoch:3d}:")
                print(f"    Total loss: {avg_loss:.6f}")
                print(f"    Reconstruction loss: {avg_recon_loss:.6f}")
                print(f"    L2 loss: {avg_l2_loss:.4f}")
                
                if use_domain_adaptation:
                    print(f"    Domain loss: {avg_domain_loss:.6f}")
                    print(f"    Disc loss: {avg_disc_loss:.6f}")
                    print(f"    Disc accuracy: {avg_disc_acc:.3f}")
                    print(f"    GRL lambda: {lambda_grl:.3f}")
                
                print(f"    Reconstruction error (test): {recon_error:.6f}")
                print(f"    Embedding std: {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")
                
                model.train()
    
    # Final embedding health check (ensure we have final epoch values)
    print("\nComputing final embedding statistics...")
    model.eval()
    with torch.no_grad():
        # Use synthetic images for final evaluation
        test_imgs = batch_images[:20]
        final_embs, final_recon = model(test_imgs)
        final_emb_std, final_emb_dist = check_embedding_health(final_embs, device)
    
    # Update history with final values (if last epoch wasn't checked)
    if not history['emb_std'] or (epochs - 1) % check_frequency != 0:
        history['emb_std'].append(final_emb_std)
        history['emb_dist'].append(final_emb_dist)
    
    # Final evaluation
    print("\n" + "="*70)
    print("PRETRAINING COMPLETE")
    print("="*70)
    
    final_loss = history['loss'][-1]
    final_recon = history['recon_loss'][-1]
    final_std = history['emb_std'][-1] if history['emb_std'] else final_emb_std
    final_dist = history['emb_dist'][-1] if history['emb_dist'] else final_emb_dist
    
    print(f"\nFinal metrics:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Total loss: {final_loss:.6f}")
    print(f"  Reconstruction loss: {final_recon:.6f}")
    print(f"  Embedding std: {final_std:.6f}")
    print(f"  Embedding dist: {final_dist:.6f}")
    
    if use_domain_adaptation:
        final_domain = history['domain_loss'][-1]
        final_disc_acc = history['disc_accuracy'][-1]
        print(f"  Domain loss: {final_domain:.6f}")
        print(f"  Discriminator accuracy: {final_disc_acc:.3f}")
    
    # Quality assessment
    print("\nQuality assessment:")
    
    # Domain adaptation (only if used)
    if use_domain_adaptation:
        if final_disc_acc > 0.8:
            print("  ❌ WARNING: Discriminator too strong (encoder not fooling it)")
        elif final_disc_acc < 0.45:
            print("  ❌ WARNING: Discriminator collapsed (too weak)")
        elif 0.5 <= final_disc_acc <= 0.7:
            print("  ✅ Excellent domain adaptation (discriminator confused)")
        else:
            print("  🟡 Moderate domain adaptation")
    
    # Reconstruction
    if final_recon > 0.1:
        print("  ⚠️  High reconstruction error - may need more training")
    elif final_recon > 0.01:
        print("  ✅ Moderate reconstruction quality")
    else:
        print("  ✅ Excellent reconstruction quality")
    
    # Diversity
    if final_std < 0.01:
        print("  ❌ WARNING: Low embedding diversity (possible collapse)")
    elif final_std < 0.1:
        print("  ⚠️  Embedding diversity is moderate")
    else:
        print("  ✅ Good embedding diversity")
    
    # Compactness (for flow training)
    if final_dist > 20:
        print("  ⚠️  Embeddings very spread out - consider higher L2 weight")
    elif final_dist > 15:
        print("  🟡 Embeddings moderately spread - may impact flow training")
    elif final_dist > 10:
        print("  ✅ Embeddings reasonably compact")
    else:
        print("  ✅ Embeddings very compact (excellent for flow training)")
    
    # Save weights
    print("\n" + "="*70)
    print("SAVING WEIGHTS")
    print("="*70)
    
    # Save encoder weights only (for downstream use)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"✅ Encoder weights: {save_path}")
    
    # Save full model (including decoder and discriminator)
    full_model_path = save_path.replace('.pt', '_full_model.pt')
    torch.save(model.state_dict(), full_model_path)
    if use_domain_adaptation:
        print(f"✅ Full model (encoder+decoder+discriminator): {full_model_path}")
    else:
        print(f"✅ Full model (encoder+decoder): {full_model_path}")
    
    # Save training history
    history_path = save_path.replace('.pt', '_history.pt')
    history['embedding_name'] = embedding_name
    history['embedding_dim'] = embedding_dim
    history['image_size'] = image_size
    history['encoder_params'] = params['encoder']
    history['decoder_params'] = params['decoder']
    if 'discriminator' in params:
        history['discriminator_params'] = params['discriminator']
    history['l2_weight'] = l2_weight
    history['use_domain_adaptation'] = use_domain_adaptation
    if use_domain_adaptation:
        history['lambda_domain'] = lambda_domain
    torch.save(history, history_path)
    print(f"✅ Training history: {history_path}")
    
    print("="*70 + "\n")
    
    return model, final_loss


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive pre-training for SPATIAL_CRYO encoder (with optional domain adaptation)'
    )
    
    # Required arguments
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to image config JSON')
    
    # Optional real images for domain adaptation
    parser.add_argument('--real_images', type=str, default=None,
                       help='Path to real images MRC file (optional, enables domain adaptation)')
    
    # Embedding architecture
    parser.add_argument('--embedding', type=str, default='SPATIAL_CRYO_FFT_FILTER',
                       choices=['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER'],
                       help='Embedding architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 0.0002)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension (default: 256)')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                       help='L2 regularization weight on embeddings (default: 0.0)')
    parser.add_argument('--lambda_domain', type=float, default=0.1,
                       help='Domain adversarial loss weight (default: 0.1, only used with --real_images)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='pretrained_spatial_cryo.pt',
                       help='Output path for pretrained encoder weights')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: "cpu", "cuda", "cuda:0", "cuda:1", etc.')
    
    # Other
    parser.add_argument('--simulation_batch_size', type=int, default=1024,
                       help='Simulation batch size (default: 1024)')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                       help='Number of simulation batches per epoch (default: 100)')
    parser.add_argument('--check_frequency', type=int, default=5,
                       help='Print detailed stats every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"❌ CUDA not available! Falling back to CPU")
            args.device = 'cpu'
        else:
            if ':' in args.device:
                gpu_id = int(args.device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    print(f"❌ GPU {gpu_id} not available!")
                    print(f"   Available GPUs: 0-{torch.cuda.device_count()-1}")
                    print(f"   Falling back to cuda:0")
                    args.device = 'cuda:0'
            
            print(f"✅ Using device: {args.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(args.device)}")
    
    # Validate real_images if provided
    if args.real_images is not None:
        if not MRCFILE_AVAILABLE:
            print(f"❌ ERROR: --real_images provided but mrcfile not installed!")
            print(f"   Install with: pip install mrcfile")
            print(f"   Falling back to reconstruction-only mode")
            args.real_images = None
    
    # Run pretraining
    model, final_loss = pretrain_spatial_cryo(
        image_config_path=args.image_config,
        real_images_path=args.real_images,
        embedding_name=args.embedding,
        device=args.device,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        simulation_batch_size=args.simulation_batch_size,
        save_path=args.output,
        n_batches_per_epoch=args.batches_per_epoch,
        check_frequency=args.check_frequency,
        l2_weight=args.l2_weight,
        lambda_domain=args.lambda_domain,
    )
    
    if model is None:
        return 1
    
    if args.real_images is not None:
        print(f"\n✅ Domain-adaptive pre-training complete!")
        print(f"   Architecture: {args.embedding}")
        print(f"   Mode: Reconstruction + Domain Adaptation")
        print(f"   Domain adversarial weight: {args.lambda_domain}")
    else:
        print(f"\n✅ Reconstruction pre-training complete!")
        print(f"   Architecture: {args.embedding}")
        print(f"   Mode: Reconstruction Only")
    
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Encoder weights saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
