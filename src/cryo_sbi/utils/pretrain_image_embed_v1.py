"""
pretrain_image_embed_v1.py

Simplified unsupervised pre-training of image encoder.
Supports 3 training modes:
- 'synthetic': Train on simulated images only
- 'real': Train on real images only  
- 'mixed': Train on both (50/50 mix)

Can resume from checkpoint for fine-tuning with different data mix.

Usage (train on synthetic):
    python pretrain_image_embed.py \
        --image_config config.json \
        --training_mode synthetic \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 256 \
        --epochs 100 \
        --batch_size 512

Usage (fine-tune on real):
    python pretrain_image_embed.py \
        --image_config config.json \
        --training_mode real \
        --real_images real_data.mrc \
        --resume_from pretrained_image_embed_full_model.pt \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 256 \
        --epochs 20 \
        --batch_size 512 \
        --lr 1e-4

Usage (train on mixed):
    python pretrain_image_embed.py \
        --image_config config.json \
        --training_mode mixed \
        --real_images real_data.mrc \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 256 \
        --epochs 100 \
        --batch_size 512
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator
try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not installed. Real image loading disabled.")
    print("Install with: pip install mrcfile")


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
# DECODERS
# ============================================================================

class SimpleDecoder(nn.Module):
    """Lightweight decoder for reconstruction"""
    def __init__(self, embedding_dim, image_size):
        super().__init__()
        
        # Calculate how many channels we need at the smallest spatial size
        # For image_size=128, we want to start from 8x8 or 16x16
        start_size = image_size // 16  # e.g., 128 -> 8
        
        self.fc = nn.Linear(embedding_dim, 512 * start_size * start_size)
        self.start_size = start_size
        
        # Upsample from start_size to image_size
        layers = []
        in_channels = 512
        
        # Each conv_transpose2d with stride=2 doubles spatial dimensions
        num_upsamples = int(np.log2(image_size // start_size))
        
        for i in range(num_upsamples):
            out_channels = in_channels // 2 if i < num_upsamples - 1 else 64
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Final layer to get single channel output
        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: [B, embedding_dim]
        Returns:
            reconstruction: [B, 1, H, W]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.start_size, self.start_size)
        x = self.decoder(x)
        return x


class SpatialCryoDecoder(nn.Module):
    """
    Lightweight decoder that perfectly mirrors SPATIAL_CRYO encoder
    All-convolutional design, truly symmetric architecture
    """
    def __init__(self, embedding_dim, image_size):
        super().__init__()
        
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        
        # Calculate upsampling stages (must match encoder)
        import math
        n_stages = int(math.log2(image_size)) - 2
        
        # Initial channel count (mirror encoder's final channels before last conv)
        start_channels = 16 * (2 ** (n_stages - 1))
        
        # Project directly to 4x4 spatial size (mirror encoder's final conv in reverse)
        # Encoder: [B, start_channels, 4, 4] → [B, output_dim, 1, 1]
        # Decoder: [B, output_dim] → [B, start_channels, 4, 4]
        self.fc = nn.Linear(embedding_dim, start_channels * 4 * 4)
        self.start_channels = start_channels
        
        # Progressive upsampling (mirror encoder stages in reverse)
        layers = []
        in_channels = start_channels
        
        # Perform n_stages upsampling operations
        for i in range(n_stages):
            # Last stage: force to 16 channels (mirror encoder's first conv output)
            if i == n_stages - 1:
                out_channels = 16
            else:
                out_channels = in_channels // 2
            
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels,
                                 kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Final conv to get single channel (mirror encoder's input)
        layers.append(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: [B, embedding_dim] - Encoder embeddings (LayerNorm'd)
        
        Returns:
            reconstruction: [B, 1, H, W] - Reconstructed images (in normalized space)
        """
        B = z.shape[0]
        
        # Project to [B, start_channels * 4 * 4]
        x = self.fc(z)
        
        # Reshape to [B, start_channels, 4, 4]
        x = x.view(B, self.start_channels, 4, 4)
        
        # Upsample to full size
        x = self.decoder(x)
        
        return x

# ============================================================================
# MODEL WRAPPER
# ============================================================================

class ImageEmbedPretrainModel(nn.Module):
    """
    Image encoder + decoder for pretraining
    """
    def __init__(self, embedding_name, embedding_dim, image_size):
        super().__init__()
        
        self.embedding_name = embedding_name
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        
        # Create encoder based on embedding_name
        self.encoder = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size)
        print(f"  Encoder: {embedding_name} (D={image_size})")

        # Create decoder based on embedding_name
        if embedding_name in ['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER', 'SPATIAL_CRYO_GAUSS_FFT_FILTER']:
           self.decoder = SpatialCryoDecoder(embedding_dim, image_size)
           print(f"  Decoder: SpatialCryoDecoder")
        else:
           self.decoder = SimpleDecoder(embedding_dim, image_size)
           print(f"  Decoder: SimpleDecoder")
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W]
        Returns:
            embeddings: [B, embedding_dim]
            reconstruction: [B, 1, H, W]
        """
        embeddings = self.encoder(x)
        reconstruction = self.decoder(embeddings)
        return embeddings, reconstruction


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
        if hasattr(self, 'mrc_data'):
            if isinstance(self.mrc_data, np.memmap):
                del self.mrc_data


def create_real_image_loader(mrc_path, batch_size=1024, num_workers=4):
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
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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
    
    return {
        'total': total,
        'trainable': trainable,
        'encoder': encoder,
        'decoder': decoder
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def pretrain_image_embed(
    image_config_path: str,
    training_mode: str = 'synthetic',  # 'synthetic', 'real', or 'mixed'
    real_images_path: str = None,
    resume_from: str = None,
    embedding_name: str = 'SPATIAL_CRYO_FFT_FILTER',
    device: str = 'cuda',
    embedding_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 2e-4,
    simulation_batch_size: int = 1024,
    save_path: str = 'pretrained_image_embed.pt',
    check_frequency: int = 5,
    n_batches_per_epoch: int = 100,
    l2_weight: float = 0.0,
    mix_ratio: float = 0.5
):
    """
    Unsupervised pre-training with flexible data sources
    
    Args:
        image_config_path: Path to image config JSON
        training_mode: 'synthetic', 'real', or 'mixed'
        real_images_path: Path to real images MRC file (required if mode='real' or 'mixed')
        resume_from: Path to full model checkpoint to resume from (optional)
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
        mix_ratio: 0.0 = all real, 1.0 = all synthetic
    
    Returns:
        model: Trained model
        final_loss: Final total loss
    """
    
    # Validate training mode
    if training_mode not in ['synthetic', 'real', 'mixed']:
        raise ValueError(f"training_mode must be 'synthetic', 'real', or 'mixed', got '{training_mode}'")
    
    if training_mode in ['real', 'mixed'] and real_images_path is None:
        raise ValueError(f"training_mode='{training_mode}' requires --real_images")

    # Only relevant for mixed mode
    if training_mode == 'mixed':
        if not 0.0 <= mix_ratio <= 1.0:
           raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")
    
    print("\n" + "="*70)
    print(f"PRETRAINING: {embedding_name}")
    print(f"Training mode: {training_mode.upper()}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("="*70)
    
    # Load image config
    image_config = json.load(open(image_config_path))
    image_size = image_config["N_PIXELS"]
    
    # Setup synthetic data if needed
    synthetic_loader = None
    synthetic_iter = None
    models = None
    
    if training_mode in ['synthetic', 'mixed']:
        print("\nLoading conformational models...")
        if image_config["MODEL_FILE"].endswith("npy"):
            models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(device).float()
        else:
            models = torch.load(image_config["MODEL_FILE"]).to(device).float()
        
        n_conformations = len(models)
        print(f"  Number of conformations: {n_conformations}")
        print(f"  Image size: {image_size}x{image_size}")
        
        # Setup synthetic data generation
        image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
        synthetic_loader = PriorLoader(
            image_prior, 
            batch_size=simulation_batch_size, 
            num_workers=4
        )
        # create iterator
        synthetic_iter = iter(synthetic_loader)
 
    # Setup real data if needed
    real_loader = None
    real_iter = None
    
    if training_mode in ['real', 'mixed']:
        print("\nLoading real images...")
        try:
            real_loader = create_real_image_loader(
                real_images_path, 
                batch_size=simulation_batch_size,
                num_workers=4
            )
            # create iterator
            real_iter = iter(real_loader)
        except Exception as e:
            print(f"❌ Error loading real images: {e}")
            return None, 0.0
    
    # Build or load model
    print(f"\nBuilding model with {embedding_name}...")
    try:
        model = ImageEmbedPretrainModel(
            embedding_name, 
            embedding_dim, 
            image_size
        ).to(device)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return None, 0.0
    
    # Load checkpoint if resuming
    if resume_from:
        print(f"\nLoading checkpoint from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded successfully")
    
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
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Setup simulation parameters if needed
    if training_mode in ['synthetic', 'mixed']:
        num_pixels = torch.tensor(image_config["N_PIXELS"], dtype=torch.float32, device=device)
        pixel_size = torch.tensor(image_config["PIXEL_SIZE"], dtype=torch.float32, device=device)
        voltage = image_config.get("VOLTAGE", 300.0)
        cs = image_config.get("SPHERICAL_ABERRATION", 0.0)
    
    print("\nTraining configuration:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Training mode: {training_mode}")
    if training_mode == 'mixed':
        print(f"  Mix ratio: {mix_ratio:.2f} (synthetic:{mix_ratio:.0%}, real:{1-mix_ratio:.0%})")
    print(f"  L2 regularization weight: {l2_weight}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    if training_mode in ['synthetic', 'mixed']:
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
    
    # Training loop
    print("\nStarting training...\n")

    with tqdm(range(epochs), desc="Pretraining") as tq:
        for epoch in tq:

            model.train()

            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_l2_loss = 0
            n_batches = 0
 
            for batch_idx in range(n_batches_per_epoch):
                
                # Get batch based on mode
                if training_mode in ['synthetic', 'mixed']:

                    try:
                        parameters = next(synthetic_iter)
                    except StopIteration:
                        synthetic_iter = iter(synthetic_loader)
                        parameters = next(synthetic_iter)

                    (indices, quaternions, res, shift, defocus, b_factor, amp, snr) = parameters

                    # get synthetic images
                    syn_images, _ = cryo_em_simulator(
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
                
                if training_mode in ['real', 'mixed']:

                    try:
                        real_images = next(real_iter)
                    except StopIteration:
                        real_iter = iter(real_loader)
                        real_images = next(real_iter)

                    # put real images on device
                    real_images = real_images.to(device, non_blocking=True)
                
                # Combine based on mode
                if training_mode == 'synthetic':
                    images = syn_images
                elif training_mode == 'real':
                    images = real_images
                else:  # mixed mode
                    # calculate number of images
                    total_samples = min(len(syn_images), len(real_images))
                    # Calculate samples based on mix_ratio
                    # mix_ratio: 0.0 = all real, 1.0 = all synthetic, 0.5 = 50/50
                    n_syn = int(total_samples * mix_ratio)
                    n_real = total_samples - n_syn

                    # Take samples
                    combined = torch.cat([
                        syn_images[:n_syn],
                        real_images[:n_real]
                    ], dim=0)

                    # Shuffle to mix synthetic and real randomly
                    perm = torch.randperm(len(combined), device=combined.device)
                    images = combined[perm]

                # Train on mini-batches
                for batch_images in images.split(batch_size):
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings, reconstruction = model(batch_images)
                    
                    # Reconstruction loss
                    recon_loss = F.mse_loss(reconstruction.squeeze(1), batch_images)
                    
                    # L2 regularization - per-sample norm
                    l2_loss = (torch.norm(embeddings, dim=1) ** 2).mean()
                    
                    # Total loss
                    loss = recon_loss + l2_weight * l2_loss
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
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
                
                history['emb_std'].append(emb_std)
                history['emb_dist'].append(emb_dist)
                
                print(f"\n  Epoch {epoch:3d}:")
                print(f"    Total loss: {avg_loss:.6f}")
                print(f"    Reconstruction loss: {avg_recon_loss:.6f}")
                print(f"    L2 loss: {avg_l2_loss:.4f}")
                print(f"    Reconstruction error (test): {recon_error:.6f}")
                print(f"    Embedding std: {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")
                
                model.train()

    
    # Final embedding health check
    print("\nComputing final embedding statistics...")
    model.eval()
    with torch.no_grad():
        test_imgs = batch_images[:20]
        final_embs, final_recon = model(test_imgs)
        final_emb_std, final_emb_dist = check_embedding_health(final_embs, device)
    
    # Update history with final values
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
    print(f"  Training mode: {training_mode}")
    print(f"  Total loss: {final_loss:.6f}")
    print(f"  Reconstruction loss: {final_recon:.6f}")
    print(f"  Embedding std: {final_std:.6f}")
    print(f"  Embedding dist: {final_dist:.6f}")
    
    # Quality assessment
    print("\nQuality assessment:")
    
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
    
    # Save full model (encoder+decoder)
    full_model_path = save_path.replace('.pt', '_full_model.pt')
    torch.save(model.state_dict(), full_model_path)
    print(f"✅ Full model (encoder+decoder): {full_model_path}")
    
    # Save training history
    history_path = save_path.replace('.pt', '_history.pt')
    history['embedding_name'] = embedding_name
    history['embedding_dim'] = embedding_dim
    history['image_size'] = image_size
    history['training_mode'] = training_mode
    history['encoder_params'] = params['encoder']
    history['decoder_params'] = params['decoder']
    history['l2_weight'] = l2_weight
    history['resumed_from'] = resume_from
    torch.save(history, history_path)
    print(f"✅ Training history: {history_path}")
    
    print("="*70 + "\n")
    
    return model, final_loss

