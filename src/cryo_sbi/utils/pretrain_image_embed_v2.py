# "pretrain_image_embed_v2.py"
"""
pretrain_image_embed_v2.py

Simplified unsupervised pre-training of image encoder on synthetic data.
Now includes an optional classifier head to enforce separation between
different conformations in the latent space.

This version trains on SYNTHETIC data only.

Usage (with classification loss):
    python pretrain_image_embed.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --epochs 100 \
        --batch_size 256 \
        --classifier_weight 0.5

Usage (reconstruction only):
    python pretrain_image_embed.py \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --epochs 100 \
        --batch_size 256 \
        --classifier_weight 0.0
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
    def __init__(self, embedding_dim, image_size, gn_groups=8):
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
                nn.GroupNorm(min(gn_groups, out_channels), out_channels),
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
    Image encoder + decoder for pretraining, with an optional classifier head.
    """
    def __init__(self, embedding_name, embedding_dim, image_size, n_classes=None):
        super().__init__()
        
        self.embedding_name = embedding_name
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.n_classes = n_classes
        
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

        # Add Classifier Head
        if n_classes is not None and n_classes > 0:
            self.classifier = nn.Linear(embedding_dim, n_classes)
            print(f"  Classifier Head: Enabled for {n_classes} classes")
        else:
            self.classifier = None
            print(f"  Classifier Head: Disabled")
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W]
        Returns:
            embeddings: [B, embedding_dim]
            reconstruction: [B, 1, H, W]
            logits: [B, n_classes] or None
        """
        embeddings = self.encoder(x)
        reconstruction = self.decoder(embeddings)
        
        # Get classification logits
        logits = None
        if self.classifier is not None:
            logits = self.classifier(embeddings)
        
        return embeddings, reconstruction, logits


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
    
    classifier = 0
    if hasattr(model, 'classifier') and model.classifier is not None:
        classifier = sum(p.numel() for p in model.classifier.parameters())

    return {
        'total': total,
        'trainable': trainable,
        'encoder': encoder,
        'decoder': decoder,
        'classifier': classifier
    }


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
    l2_weight: float = 0.0,
    classifier_weight: float = 0.0,
    mse_loss: str = 'noisy'
):
    """
    Unsupervised pre-training on synthetic data with optional classification loss.
    
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
        classifier_weight: Weight for classification loss on synthetic data (0 to disable)
        mse_loss: Compare against noisy or clean images

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
            image_size,
            n_classes=n_conformations if classifier_weight > 0 else None
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
    if params['classifier'] > 0:
        print(f"  Classifier parameters: {params['classifier']:,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Setup loss for classifier
    if classifier_weight > 0.0:
        classifier_criterion = nn.CrossEntropyLoss()

    # Setup simulation parameters
    simulation_param = create_simulation_param(image_config, models, device=device)

    print("\nTraining configuration:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  MSE loss type: {mse_loss}")
    print(f"  L2 regularization weight: {l2_weight}")
    if classifier_weight > 0.0:
        print(f"  Classification loss weight: {classifier_weight}")
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
        'class_loss': [],
        'class_acc': [],
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
            epoch_class_loss = 0
            epoch_correct_preds = 0
            epoch_total_preds = 0
            n_batches = 0
 
            for batch_idx in range(n_batches_per_epoch):
                
                try:
                    parameters = next(synthetic_iter)
                except StopIteration:
                    synthetic_iter = iter(synthetic_loader)
                    parameters = next(synthetic_iter)

                (indices, quaternions, shift, defocus, b_factor, amp, snr) = parameters
                labels = indices.round().long().squeeze(-1).to(device, non_blocking=True)

                # get synthetic images
                images, images_clean = cryo_em_simulator(
                    models,
                    indices.to(device, non_blocking=True),
                    quaternions.to(device, non_blocking=True),
                    shift.to(device, non_blocking=True),
                    defocus.to(device, non_blocking=True),
                    b_factor.to(device, non_blocking=True),
                    amp.to(device, non_blocking=True),
                    snr.to(device, non_blocking=True),
                    simulation_param
                )

                # Train on mini-batches
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size]
                    batch_images_clean = images_clean[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings, reconstruction, logits = model(batch_images)
                    
                    # Reconstruction loss
                    if mse_loss=='clean':
                       recon_loss = F.mse_loss(reconstruction.squeeze(1), batch_images_clean) 
                    else:
                       recon_loss = F.mse_loss(reconstruction.squeeze(1), batch_images)
                    
                    # L2 regularization - per-sample norm
                    l2_loss = (torch.norm(embeddings, dim=1) ** 2).mean()
                    
                    # Classification loss
                    class_loss = torch.tensor(0.0, device=device)
                    if classifier_weight > 0.0 and logits is not None:
                        class_loss = classifier_criterion(logits, batch_labels)
                        
                        # Track accuracy
                        with torch.no_grad():
                            preds = torch.argmax(logits, dim=1)
                            epoch_correct_preds += (preds == batch_labels).sum().item()
                            epoch_total_preds += len(batch_labels)

                    # Total loss
                    loss = recon_loss + l2_weight * l2_loss + classifier_weight * class_loss
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_l2_loss += l2_loss.item()
                    epoch_class_loss += class_loss.item()
                    n_batches += 1
            
            # Epoch statistics
            avg_loss = epoch_loss / n_batches
            avg_recon_loss = epoch_recon_loss / n_batches
            avg_l2_loss = epoch_l2_loss / n_batches
            avg_class_loss = epoch_class_loss / n_batches
            accuracy = (epoch_correct_preds / epoch_total_preds) if epoch_total_preds > 0 else 0.0
            
            history['loss'].append(avg_loss)
            history['recon_loss'].append(avg_recon_loss)
            history['l2_loss'].append(avg_l2_loss)
            history['class_loss'].append(avg_class_loss)
            history['class_acc'].append(accuracy)
            
            # Update progress bar
            postfix_dict = {
                "loss": f"{avg_loss:.4f}",
                "recon": f"{avg_recon_loss:.4f}",
                "l2": f"{avg_l2_loss:.4f}"
            }
            if classifier_weight > 0.0:
                postfix_dict["class"] = f"{avg_class_loss:.4f}"
                postfix_dict["acc"] = f"{accuracy:.2%}"
            tq.set_postfix(postfix_dict)
            
            # Detailed check every N epochs
            if epoch % check_frequency == 0:
                model.eval()
                with torch.no_grad():
                    # Check on last batch
                    test_imgs = batch_images[:20]
                    test_embs, test_recon, _ = model(test_imgs)
                    
                    emb_std, emb_dist = check_embedding_health(test_embs, device)
                    recon_error = F.mse_loss(test_recon.squeeze(1), test_imgs).item()
                
                history['emb_std'].append(emb_std)
                history['emb_dist'].append(emb_dist)
                
                print(f"\n  Epoch {epoch:3d}:")
                print(f"    Total loss: {avg_loss:.6f}")
                print(f"    Reconstruction loss: {avg_recon_loss:.6f}")
                print(f"    L2 loss: {avg_l2_loss:.4f}")
                if classifier_weight > 0.0:
                    print(f"    Classification loss: {avg_class_loss:.4f}")
                    print(f"    Classification accuracy: {accuracy:.2%}")
                print(f"    Reconstruction error (test): {recon_error:.6f}")
                print(f"    Embedding std: {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")
                
                model.train()

    
    # Final embedding health check
    print("\nComputing final embedding statistics...")
    model.eval()
    with torch.no_grad():
        test_imgs = batch_images[:20]
        final_embs, final_recon, _ = model(test_imgs)
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
    print(f"  Total loss: {final_loss:.6f}")
    print(f"  Reconstruction loss: {final_recon:.6f}")
    if classifier_weight > 0.0:
        final_class_loss = history['class_loss'][-1]
        final_acc = history['class_acc'][-1]
        print(f"  Classification loss: {final_class_loss:.6f}")
        print(f"  Classification accuracy: {final_acc:.2%}")
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
    
    # Save full model (encoder+decoder+classifier)
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
    history['classifier_weight'] = classifier_weight
    history['resumed_from'] = resume_from
    torch.save(history, history_path)
    print(f"✅ Training history: {history_path}")
    
    print("="*70 + "\n")
    
    return model, final_loss


def main():
    """
    Parses command-line arguments and runs the pre-training script.
    """
    parser = argparse.ArgumentParser(
        description='Pre-training for image encoder on synthetic data with optional classification loss.'
    )

    # --- Required Arguments ---
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to image config JSON')

    # --- Model & Architecture ---
    parser.add_argument('--embedding', type=str, default='SPATIAL_CRYO',
                       choices=['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER', 'SPATIAL_CRYO_GAUSS_FFT_FILTER', 'RESNET18', 'RESNET18_FFT_FILTER'],
                       help='Embedding network architecture to use')
    parser.add_argument('--embedding_dim', type=int, default=16,
                       help='Output dimension of the embedding network (default: 16)')

    # MSE loss type
    parser.add_argument('--mse_loss', type=str, default='noisy',
                       choices=['clean', 'noisy'],
                       help='Images to use for MSE loss (default: noisy)')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Training batch size for the optimizer step (default: 256)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate for the AdamW optimizer (default: 2e-4)')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                       help='Weight for L2 regularization on embeddings (default: 0.0)')
    parser.add_argument('--classifier_weight', type=float, default=0.0,
                       help='Weight for the classification loss. Set > 0 to enable. A good starting '
                            'point for noisy images is ~0.4 (default: 0.0)')

    # --- I/O and Checkpoints ---
    parser.add_argument('--save_path', type=str, default='pretrained_image_embed.pt',
                       help='Output path for the saved encoder weights')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a full model checkpoint to resume training from')

    # --- Execution & Other ---
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training: "cpu", "cuda", "cuda:0", etc. (default: "cuda")')
    parser.add_argument('--simulation_batch_size', type=int, default=1024,
                       help='Number of images to simulate at once (default: 1024)')
    parser.add_argument('--n_batches_per_epoch', type=int, default=100,
                       help='Number of simulation batches to generate per epoch (default: 100)')
    parser.add_argument('--check_frequency', type=int, default=5,
                       help='How often to print detailed stats, in epochs (default: 5)')

    args = parser.parse_args()

    # --- Validate Device ---
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"❌ CUDA not available! Falling back to CPU")
            args.device = 'cpu'
        else:
            if ':' in args.device:
                try:
                    gpu_id = int(args.device.split(':')[1])
                    if gpu_id >= torch.cuda.device_count():
                        print(f"❌ GPU {gpu_id} not available! Available GPUs: 0-{torch.cuda.device_count()-1}")
                        print(f"   Falling back to cuda:0")
                        args.device = 'cuda:0'
                except ValueError:
                    print(f"❌ Invalid CUDA device format: {args.device}. Using default cuda:0.")
                    args.device = 'cuda:0'

            print(f"✅ Using device: {args.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(args.device)}")

    # --- Run Training ---
    pretrain_image_embed(
        image_config_path=args.image_config,
        resume_from=args.resume_from,
        embedding_name=args.embedding,
        device=args.device,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        simulation_batch_size=args.simulation_batch_size,
        save_path=args.save_path,
        check_frequency=args.check_frequency,
        n_batches_per_epoch=args.n_batches_per_epoch,
        l2_weight=args.l2_weight,
        classifier_weight=args.classifier_weight,
        mse_loss=args.mse_loss
    )


if __name__ == '__main__':
    main()
