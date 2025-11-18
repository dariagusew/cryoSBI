"""
pretrain_gnn.py

Pre-train GNN embedding network to distinguish between different conformations.
This creates discriminative embeddings that can then be used in NLE training.

Usage:
    python pretrain_gnn.py --model_file models.npy --output pretrained_gnn.pt --cutoff 30.0 --hidden_dim 256
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation

from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


class RotatedConformationDataset(Dataset):
    """Dataset that applies random rotations on-the-fly to conformational models"""
    
    def __init__(self, models, samples_per_model=20, add_noise=False, noise_std=0.1):
        """
        Args:
            models: torch.Tensor [num_models, num_beads, 3]
            samples_per_model: int, number of random rotations per model
            add_noise: bool, whether to add Gaussian noise to coordinates
            noise_std: float, standard deviation of Gaussian noise
        """
        self.models = models
        self.num_models = len(models)
        self.samples_per_model = samples_per_model
        self.add_noise = add_noise
        self.noise_std = noise_std
        
    def __len__(self):
        return self.num_models * self.samples_per_model
    
    def __getitem__(self, idx):
        model_idx = idx // self.samples_per_model
        coords = self.models[model_idx].clone()
        
        # Apply random rotation
        coords = self.random_rotate(coords)
        
        # Center coordinates
        coords = coords - coords.mean(dim=0, keepdim=True)
        
        # Optionally add noise
        if self.add_noise:
            coords = coords + torch.randn_like(coords) * self.noise_std
        
        return coords, model_idx
    
    @staticmethod
    def random_rotate(coords):
        """Apply random 3D rotation uniformly sampled from SO(3)"""
        # scipy's random method uses uniform sampling from SO(3)
        R = Rotation.random().as_matrix()
        R_torch = torch.from_numpy(R).to(dtype=coords.dtype)
        return coords @ R_torch.T


class ConformationClassifier(nn.Module):
    """Classifier head for pre-training embedding network"""
    
    def __init__(self, embedding_dim, num_classes, hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, embeddings):
        return self.classifier(embeddings)


def load_models(model_file):
    """Load conformational models from file"""
    print(f"Loading models from {model_file}")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    if model_file.endswith(".npy"):
        models = torch.from_numpy(np.load(model_file)).float()
    elif model_file.endswith(".pt") or model_file.endswith(".pth"):
        models = torch.load(model_file).float()
    else:
        raise ValueError(f"Unsupported file format: {model_file}. Use .npy, .pt, or .pth")
    
    print(f"Loaded models with shape: {models.shape}")
    
    # Ensure shape is [num_models, num_beads, 3]
    if models.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {models.shape}")
    
    if models.shape[1] == 3 and models.shape[2] != 3:
        print("Transposing from [num_models, 3, num_beads] to [num_models, num_beads, 3]")
        models = models.transpose(1, 2)
    
    if models.shape[2] != 3:
        raise ValueError(f"Last dimension should be 3 (x,y,z coordinates), got {models.shape[2]}")
    
    return models


def analyze_models(models, cutoff):
    """Print statistics about the models"""
    num_models, num_beads, _ = models.shape
    
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(f"Number of models: {num_models}")
    print(f"Number of beads per model: {num_beads}")
    print(f"Coordinate statistics:")
    print(f"  Mean: {models.mean():.4f}")
    print(f"  Std: {models.std():.4f}")
    print(f"  Min: {models.min():.4f}")
    print(f"  Max: {models.max():.4f}")
    
    # Check diversity between models
    print(f"\nModel diversity (mean coordinate difference):")
    diversities = []
    for i in range(min(5, num_models - 1)):
        diff = (models[i] - models[i+1]).abs().mean().item()
        diversities.append(diff)
        print(f"  Model {i} vs {i+1}: {diff:.4f}")
    
    # Check typical distances within a model
    print(f"\nTypical pairwise distances (model 0):")
    model = models[0]
    dists = torch.cdist(model.unsqueeze(0), model.unsqueeze(0))[0]
    dists = dists[dists > 0]  # Remove diagonal
    print(f"  Min: {dists.min():.2f}")
    print(f"  Max: {dists.max():.2f}")
    print(f"  Mean: {dists.mean():.2f}")
    print(f"  Median: {dists.median():.2f}")
    print(f"  25th percentile: {dists.quantile(0.25):.2f}")
    print(f"  75th percentile: {dists.quantile(0.75):.2f}")
    
    # Check cutoff appropriateness
    print(f"\nCutoff analysis (cutoff = {cutoff:.2f}):")
    below_cutoff = (dists < cutoff).float().mean().item() * 100
    print(f"  Percentage of distances below cutoff: {below_cutoff:.1f}%")
    if below_cutoff < 10:
        print(f"  ⚠ WARNING: Cutoff is too small! Most edges will be missing.")
    elif below_cutoff > 90:
        print(f"  ⚠ WARNING: Cutoff is too large! Graph will be too dense.")
    else:
        print(f"  ✓ Cutoff seems appropriate")
    
    print("="*60 + "\n")


def pretrain_gnn(
    model_file: str,
    output_weights: str,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    cutoff: float = 30.0,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    device: str = "cuda",
    samples_per_model: int = 20,
    add_noise: bool = False,
    noise_std: float = 0.1,
    num_workers: int = 4,
    save_frequency: int = 20,
    seed: int = 42
):
    """
    Pre-train GNN to distinguish between different conformations.
    
    Args:
        model_file: Path to model file (.npy or .pt)
        output_weights: Path to save trained GNN weights
        embedding_dim: Output dimension of GNN
        hidden_dim: Hidden dimension of GNN
        cutoff: Distance cutoff for graph construction
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device for training ('cuda' or 'cpu')
        samples_per_model: Number of augmented samples per model
        add_noise: Whether to add noise to coordinates
        noise_std: Standard deviation of Gaussian noise
        num_workers: Number of data loading workers
        save_frequency: Save checkpoint every N epochs
        seed: Random seed
    """
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load and analyze models
    models = load_models(model_file)
    analyze_models(models, cutoff)
    
    num_models = models.shape[0]
    num_beads = models.shape[1]
    
    # Create dataset
    print(f"Creating dataset with {samples_per_model} rotations per model...")
    dataset = RotatedConformationDataset(
        models, 
        samples_per_model=samples_per_model,
        add_noise=add_noise,
        noise_std=noise_std
    )
    print(f"Total samples: {len(dataset)}")
    
    # Test dataset
    print("\nTesting augmentation...")
    sample1, label1 = dataset[0]
    sample2, label2 = dataset[1]
    print(f"Sample 1: label={label1}, mean={sample1.mean():.4f}, std={sample1.std():.4f}")
    print(f"Sample 2: label={label2}, mean={sample2.mean():.4f}, std={sample2.std():.4f}")
    if label1 == label2:
        diff = (sample1 - sample2).abs().mean()
        print(f"Difference between augmentations of same model: {diff:.4f}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"\nTrain samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    # Initialize GNN
    print(f"\n{'='*60}")
    print("INITIALIZING NETWORK")
    print(f"{'='*60}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Cutoff: {cutoff}")
    
    gnn = EMBEDDING_NETS["GNN"](
        output_dimension=embedding_dim,
        hidden_dim=hidden_dim,
        cutoff=cutoff
    ).to(device)
    
    # Count parameters
    gnn_params = sum(p.numel() for p in gnn.parameters())
    print(f"GNN parameters: {gnn_params:,}")
    
    # Test GNN
    print("\nTesting GNN forward pass...")
    test_batch = torch.randn(4, num_beads, 3).to(device)
    with torch.no_grad():
        test_output = gnn(test_batch)
        print(f"  Input shape: {test_batch.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output mean: {test_output.mean():.4f}, std: {test_output.std():.4f}")
        print(f"  Output std across batch: {test_output.std(dim=0).mean():.4f}")
        
        # Test graph construction
        x, edge_index, edge_attr, batch_idx = gnn.batch_coords_to_graph(test_batch, cutoff)
        print(f"\n  Graph statistics:")
        print(f"    Total nodes: {x.shape[0]}")
        print(f"    Total edges: {edge_index.shape[1]}")
        print(f"    Edges per graph: {edge_index.shape[1] / test_batch.shape[0]:.1f}")
        print(f"    Edge distances: min={edge_attr.min():.2f}, max={edge_attr.max():.2f}, mean={edge_attr.mean():.2f}")
    
    # Initialize classifier
    classifier = ConformationClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_models,
        hidden_dims=[512, 256],
        dropout=0.2
    ).to(device)
    
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total parameters: {gnn_params + classifier_params:,}")
    print(f"{'='*60}\n")
    
    # Training setup
    params = list(gnn.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"{'='*60}")
    print(f"STARTING TRAINING FOR {epochs} EPOCHS")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training phase
        gnn.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for coords, labels_batch in pbar:
            coords = coords.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            embeddings = gnn(coords)
            logits = classifier(embeddings)
            loss = criterion(logits, labels_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        gnn.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        embedding_stds = []
        
        with torch.no_grad():
            for coords, labels_batch in val_loader:
                coords = coords.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                embeddings = gnn(coords)
                logits = classifier(embeddings)
                loss = criterion(logits, labels_batch)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
                
                embedding_stds.append(embeddings.std(dim=0).mean().item())
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        avg_emb_std = np.mean(embedding_stds)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}% | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}% | "
                  f"Emb Std={avg_emb_std:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': gnn.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'cutoff': cutoff,
            }, output_weights)
            if epoch % 10 == 0:
                print(f"  → Saved best model (val_acc={val_acc:.1f}%)")
        
        # Save periodic checkpoints
        if (epoch + 1) % save_frequency == 0:
            checkpoint_path = output_weights.replace('.pt', f'_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': gnn.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'cutoff': cutoff,
            }, checkpoint_path)
        
        scheduler.step()
    
    # Training complete
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Best model saved to: {output_weights}")
    
    # Final embedding quality check
    print(f"\n{'='*60}")
    print("FINAL EMBEDDING QUALITY CHECK")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(output_weights)
    gnn.load_state_dict(checkpoint['gnn_state_dict'])
    gnn.eval()
    
    with torch.no_grad():
        # Check embeddings for original models (no rotation)
        test_models = models[:min(10, num_models)].to(device)
        test_embeddings = gnn(test_models)
        
        print(f"Embeddings for {len(test_models)} models:")
        print(f"  Mean: {test_embeddings.mean():.4f}")
        print(f"  Std: {test_embeddings.std():.4f}")
        print(f"  Std across models (diversity): {test_embeddings.std(dim=0).mean():.4f}")
        
        pairwise_dist = torch.cdist(test_embeddings, test_embeddings)
        pairwise_dist_no_diag = pairwise_dist.clone()
        pairwise_dist_no_diag.fill_diagonal_(float('inf'))
        
        print(f"\nPairwise distances:")
        print(f"  Mean: {pairwise_dist.mean():.4f}")
        print(f"  Min (excluding diagonal): {pairwise_dist_no_diag.min():.4f}")
        print(f"  Max: {pairwise_dist.max():.4f}")
        
        # Check if embeddings are discriminative
        if test_embeddings.std(dim=0).mean() > 0.1:
            print("\n✓ Embeddings are diverse and discriminative!")
        else:
            print("\n✗ WARNING: Embeddings have low diversity")
        
        if pairwise_dist_no_diag.min() > 0.5:
            print("✓ Models are well-separated in embedding space!")
        else:
            print("✗ WARNING: Some models have very similar embeddings")
    
    print(f"{'='*60}\n")
    
    return gnn, train_losses, val_losses, train_accs, val_accs


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train GNN embedding network for conformation discrimination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_file", 
        type=str, 
        required=True,
        help="Path to model file (.npy or .pt) with shape [num_models, 3, num_beads] or [num_models, num_beads, 3]"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save trained GNN weights (.pt file)"
    )
    
    # Model architecture
    parser.add_argument(
        "--embedding_dim", 
        type=int, 
        default=256,
        help="Output dimension of GNN embedding"
    )
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=256,
        help="Hidden dimension of GNN"
    )
    parser.add_argument(
        "--cutoff", 
        type=float, 
        default=30.0,
        help="Distance cutoff for graph construction (in Angstroms)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    
    # Data augmentation
    parser.add_argument(
        "--samples_per_model", 
        type=int, 
        default=20,
        help="Number of random rotations per model for data augmentation"
    )
    parser.add_argument(
        "--add_noise", 
        action="store_true",
        help="Add Gaussian noise to coordinates"
    )
    parser.add_argument(
        "--noise_std", 
        type=float, 
        default=0.1,
        help="Standard deviation of Gaussian noise (if --add_noise is set)"
    )
    
    # System parameters
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save_frequency", 
        type=int, 
        default=20,
        help="Save checkpoint every N epochs"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Run pre-training
    try:
        pretrain_gnn(
            model_file=args.model_file,
            output_weights=args.output,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            cutoff=args.cutoff,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            samples_per_model=args.samples_per_model,
            add_noise=args.add_noise,
            noise_std=args.noise_std,
            num_workers=args.num_workers,
            save_frequency=args.save_frequency,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING TRAINING")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
