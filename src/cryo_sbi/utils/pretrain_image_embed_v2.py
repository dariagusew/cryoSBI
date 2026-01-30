# "pretrain_image_embed_v2.py"
"""
pretrain_image_embed_v2.py

Simplified unsupervised pre-training of image encoder on synthetic data.

This version trains on SYNTHETIC data only.

Usage (with classification loss):
    python pretrain_image_embed.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --epochs 100 \
        --batch_size 256 \

Usage (reconstruction only):
    python pretrain_image_embed.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --epochs 100 \
        --batch_size 256 \
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param


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
                nn.LeakyReLU(0.2, inplace=True)
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


def generate_validation_set(prior_loader, models, simulation_param, val_size, device):
    """
    Generates a fixed set of validation images with a specific noise model.
    """
    print(f"\nGenerating {val_size} validation images with Gaussian noise...")
    
    val_images = []
    generated_count = 0
    val_iter = iter(prior_loader)

    with tqdm(total=val_size, desc="  Generating val set") as pbar:
        while generated_count < val_size:
            try:
                parameters = next(val_iter)
            except StopIteration:
                val_iter = iter(prior_loader) # Reset if we run out
                parameters = next(val_iter)
            
            (indices, quaternions, shift, defocus, b_factor, amp, snr) = parameters
            
            images, _ = cryo_em_simulator(
                models,
                indices.to(device, non_blocking=True),
                quaternions.to(device, non_blocking=True),
                shift.to(device, non_blocking=True),
                defocus.to(device, non_blocking=True),
                b_factor.to(device, non_blocking=True),
                amp.to(device, non_blocking=True),
                snr.to(device, non_blocking=True),
                simulation_param,
                "Gaussian"
            )
            
            val_images.append(images)
            generated_count += len(images)
            pbar.update(len(images))

    # Concatenate all generated images
    all_images = torch.cat(val_images, dim=0)
    # Compute memory
    val_mem_gb = all_images.nelement() * all_images.element_size() / 1024**3

    print(f"✅ Validation set created with {len(all_images)} images, consuming {val_mem_gb:.2f} GB of VRAM.")
    return all_images


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def pretrain_image_embed(
    image_config_path: str,
    resume_from: str = None,
    embedding_name: str = 'SPATIAL_CRYO',
    device: str = 'cuda',
    embedding_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-4,
    simulation_batch_size: int = 1024,
    save_path: str = 'pretrained_image_embed.pt',
    check_frequency: int = 5,
    n_batches_per_epoch: int = 100,
    l2_weight: float = 0.0
):
    """
    Unsupervised pre-training on synthetic data
    
    Args:
        image_config_path: Path to image config JSON
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

    Returns:
        model: Trained model
        final_loss: Final total loss
    """
    
    print("\n" + "="*70)
    print(f"PRETRAINING: {embedding_name}")
    print(f"Training mode: SYNTHETIC ONLY")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("="*70)
    
    # Load image config
    image_config = json.load(open(image_config_path))
    image_size = image_config["N_PIXELS"]
    
    # Setup synthetic data
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
    synthetic_iter = iter(synthetic_loader)

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
    
    # Setup simulation parameters
    simulation_param = create_simulation_param(image_config, models, device=device)

    # Generate a fixed validation set before training loop
    n_val_images = 10 * simulation_batch_size
    validation_images = generate_validation_set(
        synthetic_loader, models, simulation_param, n_val_images, device
    )

    print("\nTraining configuration:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  L2 regularization weight: {l2_weight}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Simulation batch size: {simulation_batch_size}")
    print(f"  Batches per epoch: {n_batches_per_epoch}")
    print(f"  Samples per epoch: {n_batches_per_epoch * simulation_batch_size:,}")
    print(f"  Validation samples: {len(validation_images):,}")
    print("="*70)
    
    # Training history
    history = {
        'loss': [],
        'recon_loss': [],
        'l2_loss': [],
        'val_loss' : [],
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
                
                try:
                    parameters = next(synthetic_iter)
                except StopIteration:
                    synthetic_iter = iter(synthetic_loader)
                    parameters = next(synthetic_iter)

                (indices, quaternions, shift, defocus, b_factor, amp, snr) = parameters

                # get synthetic images
                images, _ = cryo_em_simulator(
                    models,
                    indices.to(device, non_blocking=True),
                    quaternions.to(device, non_blocking=True),
                    shift.to(device, non_blocking=True),
                    defocus.to(device, non_blocking=True),
                    b_factor.to(device, non_blocking=True),
                    amp.to(device, non_blocking=True),
                    snr.to(device, non_blocking=True),
                    simulation_param,
                    simulation_param["noise"]
                )

                # Train on mini-batches
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size]
                    
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

            # Validation step at the end of each epoch
            model.eval()
            with torch.no_grad():
                 val_embeddings, val_reconstruction = model(validation_images)
                 val_loss = F.mse_loss(val_reconstruction.squeeze(1), validation_images)
                 val_loss = val_loss.item()

            # add to dictionary
            history['val_loss'].append(val_loss)
            # Set back to training mode
            model.train()
            
            # Update progress bar
            postfix_dict = {
                "loss": f"{avg_loss:.4f}",
                "recon": f"{avg_recon_loss:.4f}",
                "l2": f"{avg_l2_loss:.4f}",
                "val_recon": f"{val_loss:.4f}"
            }
            tq.set_postfix(postfix_dict)
            
            # Check embeddings and detailed printout every N epochs
            if epoch % check_frequency == 0:
                model.eval()
                with torch.no_grad():
                    # check embedding health
                    emb_std, emb_dist = check_embedding_health(embeddings, device)
                
                history['emb_std'].append(emb_std)
                history['emb_dist'].append(emb_dist)
                
                print(f"\n  Epoch {epoch:3d}:")
                print(f"    Total loss: {avg_loss:.6f}")
                print(f"    Reconstruction loss: {avg_recon_loss:.6f}")
                print(f"    L2 loss: {avg_l2_loss:.4f}")
                print(f"    Validation reconstruction loss: {val_loss:.6f}")
                print(f"    Embedding std: {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")
                
                model.train()

    
    # Final embedding health check
    print("\nComputing final embedding statistics...")
    model.eval()
    with torch.no_grad():
        final_embs, final_recon = model(batch_images)
        final_emb_std, final_emb_dist = check_embedding_health(final_embs, device)
    
    # Update history with final values
    if (epochs - 1) % check_frequency != 0:
        history['emb_std'].append(final_emb_std)
        history['emb_dist'].append(final_emb_dist)
    
    # Final evaluation
    print("\n" + "="*70)
    print("PRETRAINING COMPLETE")
    print("="*70)
    
    final_loss = history['loss'][-1]
    final_recon = history['recon_loss'][-1]
    final_l2 = history['l2_loss'][-1]
    final_valid = history['val_loss'][-1]
    final_std = history['emb_std'][-1]
    final_dist = history['emb_dist'][-1]
    
    print(f"\nFinal metrics:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Total loss: {final_loss:.6f}")
    print(f"  Reconstruction loss: {final_recon:.6f}")
    print(f"  L2 loss: {final_l2:.6f}")
    print(f"  Validation reconstruction loss: {final_valid:.6f}")
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
    print(f"✅ Full model (for resuming): {full_model_path}")
    
    # Save training history
    history_path = save_path.replace('.pt', '_history.pt')
    history['embedding_name'] = embedding_name
    history['embedding_dim'] = embedding_dim
    history['image_size'] = image_size
    history['encoder_params'] = params['encoder']
    history['decoder_params'] = params['decoder']
    history['l2_weight'] = l2_weight
    history['resumed_from'] = resume_from
    torch.save(history, history_path)
    print(f"✅ Training history: {history_path}")
    
    print("="*70 + "\n")
    
    return model, final_loss

