"""
pretrain_resnet18_unsupervised.py

Unsupervised pre-training of embedding networks for cryo-EM.
Uses reconstruction/denoising task (cryo-EM images are inherently noisy).

Usage:
    python pretrain_resnet18_unsupervised.py \
        --image_config config.json \
        --embedding RESNET18 \
        --epochs 150 \
        --l2_weight 0.001 \
        --output pretrained.pt
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

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator


# ============================================================================
# MODEL
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


class UnsupervisedPretrainModel(nn.Module):
    """Encoder with reconstruction/denoising head"""
    def __init__(self, embedding_name, embedding_dim, image_size):
        super().__init__()
        
        self.embedding_name = embedding_name
        self.embedding = EMBEDDING_NETS[embedding_name](embedding_dim)
        self.decoder = SimpleDecoder(embedding_dim, image_size)
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W]
        Returns:
            embeddings: [B, embedding_dim]
            reconstruction: [B, 1, H, W]
        """
        embeddings = self.embedding(x)
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


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def pretrain_unsupervised(
    image_config_path: str,
    embedding_name: str = 'RESNET18',
    device: str = 'cuda',
    embedding_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-4,
    simulation_batch_size: int = 1024,
    save_path: str = 'pretrained_embedding.pt',
    check_frequency: int = 5,
    n_batches_per_epoch: int = 100,
    l2_weight: float = 0.0,
):
    """
    Unsupervised pre-training using reconstruction/denoising
    
    Args:
        image_config_path: Path to image config JSON
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
        final_loss: Final reconstruction loss
    """
    
    print("\n" + "="*70)
    print(f"UNSUPERVISED PRETRAINING: {embedding_name}")
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
    
    # Build model
    print(f"\nBuilding model with {embedding_name}...")
    try:
        model = UnsupervisedPretrainModel(
            embedding_name, 
            embedding_dim, 
            image_size
        ).to(device)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return None, 0.0
    
    model.train()
    
    # Configure BatchNorm for stability
    print("  Configuring BatchNorm momentum = 0.01")
    for module in model.embedding.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.01
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    print("\nSetting up training...")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
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
    print(f"  Task: Reconstruction + L2 regularization")
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
    
    # Training loop
    print("\nStarting training...\n")
    
    with tqdm(range(epochs), desc="Pretraining") as tq:
        for epoch in tq:
            
            model.train()
            
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_l2_loss = 0
            n_batches = 0
            
            # Train on multiple simulation batches per epoch
            for parameters in islice(prior_loader, n_batches_per_epoch):
                (indices, quaternions, res, shift, defocus, b_factor, amp, snr) = parameters
                
                # Simulate images (inherently noisy from SNR)
                images, _ = cryo_em_simulator(
                    models,
                    indices.to(device),
                    quaternions.to(device),
                    res.to(device),
                    shift.to(device),
                    defocus.to(device),
                    b_factor.to(device),
                    amp.to(device),
                    snr.to(device),
                    num_pixels,
                    pixel_size,
                    voltage,
                    cs
                )
                
                # Train on mini-batches
                for batch_images in images.split(batch_size):
                    batch_images = batch_images.to(device)
                    
                    # Forward pass
                    embeddings, reconstruction = model(batch_images)
                    
                    # Reconstruction loss
                    recon_loss = F.mse_loss(reconstruction.squeeze(1), batch_images)
                    
                    # L2 regularization on embeddings
                    l2_loss = (embeddings ** 2).mean()
                    
                    # Total loss
                    loss = recon_loss + l2_weight * l2_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
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
                    test_imgs = images[:20]
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
                print(f"    Reconstruction error (test): {recon_error:.6f}")
                print(f"    Embedding std: {emb_std:.6f}")
                print(f"    Embedding dist: {emb_dist:.6f}")
                print(f"    Embedding L2 norm (test): {test_l2:.6f}")
                
                model.train()
    
    # Final evaluation
    print("\n" + "="*70)
    print("PRETRAINING COMPLETE")
    print("="*70)
    
    final_loss = history['loss'][-1]
    final_recon = history['recon_loss'][-1]
    final_l2 = history['l2_loss'][-1]
    final_std = history['emb_std'][-1] if history['emb_std'] else 0
    final_dist = history['emb_dist'][-1] if history['emb_dist'] else 0
    
    print(f"\nFinal metrics:")
    print(f"  Embedding: {embedding_name}")
    print(f"  Total loss: {final_loss:.6f}")
    print(f"  Reconstruction loss: {final_recon:.6f}")
    print(f"  L2 loss: {final_l2:.4f}")
    print(f"  Embedding std: {final_std:.6f}")
    print(f"  Embedding dist: {final_dist:.6f}")
    
    # Quality assessment
    print("\nQuality assessment:")
    if final_std < 0.01:
        print("  ❌ WARNING: Low embedding diversity (possible collapse)")
    elif final_std < 0.1:
        print("  ⚠️  Embedding diversity is moderate")
    else:
        print("  ✅ Good embedding diversity")
    
    if final_recon > 0.1:
        print("  ⚠️  High reconstruction error - may need more training")
    elif final_recon > 0.01:
        print("  ✅ Moderate reconstruction quality")
    else:
        print("  ✅ Excellent reconstruction quality")
    
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
    
    # Save embedding weights only (for NLE training)
    torch.save(model.embedding.state_dict(), save_path)
    print(f"✅ Embedding weights: {save_path}")
    
    # Save full model (including decoder)
    full_model_path = save_path.replace('.pt', '_full_model.pt')
    torch.save(model.state_dict(), full_model_path)
    print(f"✅ Full model: {full_model_path}")
    
    # Save training history
    history_path = save_path.replace('.pt', '_history.pt')
    history['embedding_name'] = embedding_name
    history['embedding_dim'] = embedding_dim
    history['l2_weight'] = l2_weight
    torch.save(history, history_path)
    print(f"✅ Training history: {history_path}")
    
    print("="*70 + "\n")
    
    return model, final_loss


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description='Unsupervised pre-training for cryo-EM embeddings (reconstruction task)'
#     )
    
#     # Required arguments
#     parser.add_argument('--image_config', type=str, required=True,
#                        help='Path to image config JSON')
    
#     # Embedding architecture
#     parser.add_argument('--embedding', type=str, default='RESNET18',
#                        choices=['RESNET18', 'RESNET18_FFT_FILTER'],
#                        help='Embedding architecture. Choices: RESNET18, RESNET18_FFT_FILTER')

#     # Training arguments
#     parser.add_argument('--epochs', type=int, default=100,
#                        help='Number of training epochs (default: 100)')
#     parser.add_argument('--batch_size', type=int, default=256,
#                        help='Training batch size (default: 256)')
#     parser.add_argument('--lr', type=float, default=2e-4,
#                        help='Learning rate (default: 2e-4)')
#     parser.add_argument('--embedding_dim', type=int, default=256,
#                        help='Embedding dimension (default: 256)')
#     parser.add_argument('--l2_weight', type=float, default=0.0,
#                        help='L2 regularization weight on embeddings (default: 0.0)')
    
#     # Output arguments
#     parser.add_argument('--output', type=str, default='pretrained_embedding.pt',
#                        help='Output path for pretrained weights')
    
#     # Device
#     parser.add_argument('--device', type=str, default='cuda',
#                        help='Device: "cpu", "cuda", "cuda:0", "cuda:1", etc.')
    
#     # Other
#     parser.add_argument('--simulation_batch_size', type=int, default=1024,
#                        help='Simulation batch size (default: 1024)')
#     parser.add_argument('--batches_per_epoch', type=int, default=100,
#                        help='Number of simulation batches per epoch (default: 100)')
#     parser.add_argument('--check_frequency', type=int, default=5,
#                        help='Print detailed stats every N epochs (default: 5)')
    
#     args = parser.parse_args()
    
#     # Validate device
#     if args.device.startswith('cuda'):
#         if not torch.cuda.is_available():
#             print(f"❌ CUDA not available! Falling back to CPU")
#             args.device = 'cpu'
#         else:
#             if ':' in args.device:
#                 gpu_id = int(args.device.split(':')[1])
#                 if gpu_id >= torch.cuda.device_count():
#                     print(f"❌ GPU {gpu_id} not available!")
#                     print(f"   Available GPUs: 0-{torch.cuda.device_count()-1}")
#                     print(f"   Falling back to cuda:0")
#                     args.device = 'cuda:0'
            
#             print(f"✅ Using device: {args.device}")
#             if torch.cuda.is_available():
#                 print(f"   GPU: {torch.cuda.get_device_name(args.device)}")
    
#     # Run pretraining
#     model, final_loss = pretrain_unsupervised(
#         image_config_path=args.image_config,
#         embedding_name=args.embedding,
#         device=args.device,
#         embedding_dim=args.embedding_dim,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         simulation_batch_size=args.simulation_batch_size,
#         save_path=args.output,
#         n_batches_per_epoch=args.batches_per_epoch,
#         check_frequency=args.check_frequency,
#         l2_weight=args.l2_weight,
#     )
    
#     if model is None:
#         return 1
    
#     print(f"\n✅ Unsupervised pre-training complete!")
#     print(f"   Embedding: {args.embedding}")
#     print(f"   L2 regularization weight: {args.l2_weight}")
#     print(f"   Final reconstruction loss: {final_loss:.6f}")
#     print(f"   Weights saved to: {args.output}")
    
#     return 0


# if __name__ == "__main__":
#     exit(main())
