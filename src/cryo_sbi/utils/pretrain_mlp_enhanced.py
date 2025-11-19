"""
pretrain_mlp_enhanced.py

Pre-train MLP_Enhanced embedding network to distinguish between different conformations.
Uses enhanced features + contrastive loss for better discrimination of similar conformations.

Usage:
    python pretrain_mlp_enhanced.py --model_file models.npy --output pretrained_mlp_enhanced.pt
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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to explicitly separate similar conformations.
    Pulls together embeddings from same conformation, pushes apart different conformations.
    """
    
    def __init__(self, margin=2.0, temperature=0.5):
        """
        Args:
            margin: minimum distance between different conformations
            temperature: temperature for scaling distances
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D] - batch of embeddings
            labels: [B] - conformation labels
        
        Returns:
            loss: scalar contrastive loss
        """
        B = embeddings.shape[0]
        
        # Compute pairwise distances in embedding space
        dist_matrix = torch.cdist(embeddings, embeddings)  # [B, B]
        
        # Create masks for same/different conformations
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        labels_equal.fill_diagonal_(False)  # Ignore self-comparisons
        
        # Positive pairs (same conformation) - should be close
        pos_mask = labels_equal.float()
        pos_count = pos_mask.sum()
        if pos_count > 0:
            pos_loss = (dist_matrix * pos_mask).sum() / pos_count
        else:
            pos_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Negative pairs (different conformations) - should be far (at least margin apart)
        neg_mask = (~labels_equal).float()
        neg_mask.fill_diagonal_(0.0)
        neg_count = neg_mask.sum()
        if neg_count > 0:
            neg_loss = torch.relu(self.margin - dist_matrix)  # Hinge loss
            neg_loss = (neg_loss * neg_mask).sum() / neg_count
        else:
            neg_loss = torch.tensor(0.0, device=embeddings.device)
        
        return pos_loss + neg_loss


class TripletLoss(nn.Module):
    """
    Triplet loss: for each anchor, find hardest positive and negative.
    Even better for fine-grained discrimination.
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: minimum distance between positive and negative
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D]
            labels: [B]
        
        Returns:
            loss: scalar triplet loss
        """
        B = embeddings.shape[0]
        device = embeddings.device
        
        # Compute all pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings)  # [B, B]
        
        losses = []
        for i in range(B):
            anchor_label = labels[i]
            
            # Positive: same conformation, different sample
            pos_mask = (labels == anchor_label) & (torch.arange(B, device=device) != i)
            if pos_mask.sum() == 0:
                continue
            pos_dists = dist_matrix[i, pos_mask]
            
            # Negative: different conformation
            neg_mask = labels != anchor_label
            if neg_mask.sum() == 0:
                continue
            neg_dists = dist_matrix[i, neg_mask]
            
            # Hardest positive (furthest same-class sample)
            hardest_pos = pos_dists.max()
            
            # Hardest negative (closest different-class sample)
            hardest_neg = neg_dists.min()
            
            # Triplet loss: want pos < neg by at least margin
            loss = torch.relu(hardest_pos - hardest_neg + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()


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


def analyze_models(models):
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
    
    if len(diversities) > 0:
        avg_diversity = np.mean(diversities)
        if avg_diversity < 0.5:
            print(f"  ⚠ WARNING: Models are very similar (avg diff={avg_diversity:.4f})")
            print(f"  → Contrastive/triplet loss will be especially helpful!")
        else:
            print(f"  ✓ Models have good diversity (avg diff={avg_diversity:.4f})")
    
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
    print("="*60 + "\n")


def pretrain_mlp_enhanced(
    model_file: str,
    output_weights: str,
    embedding_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    device: str = "cuda",  # Now accepts "cuda:0", "cuda:1", etc.
    samples_per_model: int = 20,
    add_noise: bool = False,
    noise_std: float = 0.1,
    num_workers: int = 4,
    save_frequency: int = 20,
    seed: int = 42,
    use_contrastive: bool = True,
    use_triplet: bool = False,
    contrastive_weight: float = 0.5,
    triplet_weight: float = 0.3,
    contrastive_margin: float = 2.0,
    triplet_margin: float = 1.0
):
    """
    Pre-train MLP_Enhanced to distinguish between different conformations.
    
    Args:
        model_file: Path to model file (.npy or .pt)
        output_weights: Path to save trained MLP weights
        embedding_dim: Output dimension of MLP
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
        use_contrastive: Whether to use contrastive loss
        use_triplet: Whether to use triplet loss
        contrastive_weight: Weight for contrastive loss
        triplet_weight: Weight for triplet loss
        contrastive_margin: Margin for contrastive loss
        triplet_margin: Margin for triplet loss
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
    analyze_models(models)
    
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
    
    # Initialize MLP_Enhanced
    print(f"\n{'='*60}")
    print("INITIALIZING NETWORK")
    print(f"{'='*60}")
    print(f"Using MLP_Enhanced with richer distance features")
    print(f"Embedding dimension: {embedding_dim}")
    
    mlp = EMBEDDING_NETS["MLP_Enhanced"](embedding_dim).to(device)
    
    # Count parameters
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP parameters: {mlp_params:,}")
    
    # Test MLP
    print("\nTesting MLP_Enhanced forward pass...")
    test_batch = torch.randn(4, num_beads, 3).to(device)
    with torch.no_grad():
        test_output = mlp(test_batch)
        print(f"  Input shape: {test_batch.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output mean: {test_output.mean():.4f}, std: {test_output.std():.4f}")
        print(f"  Output std across batch: {test_output.std(dim=0).mean():.4f}")
    
    # Initialize classifier
    classifier = ConformationClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_models,
        hidden_dims=[512, 256],
        dropout=0.2
    ).to(device)
    
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total parameters: {mlp_params + classifier_params:,}")
    
    # Loss functions
    print(f"\n{'='*60}")
    print("LOSS CONFIGURATION")
    print(f"{'='*60}")
    criterion_cls = nn.CrossEntropyLoss()
    print(f"✓ Classification loss: CrossEntropy")
    
    losses_used = ["Classification"]
    
    if use_contrastive:
        criterion_contrastive = ContrastiveLoss(margin=contrastive_margin)
        print(f"✓ Contrastive loss: margin={contrastive_margin}, weight={contrastive_weight}")
        losses_used.append("Contrastive")
    
    if use_triplet:
        criterion_triplet = TripletLoss(margin=triplet_margin)
        print(f"✓ Triplet loss: margin={triplet_margin}, weight={triplet_weight}")
        losses_used.append("Triplet")
    
    print(f"\nTotal losses: {' + '.join(losses_used)}")
    print(f"{'='*60}\n")
    
    # Training setup
    params = list(mlp.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
        mlp.train()
        classifier.train()
        train_loss = 0.0
        train_loss_cls = 0.0
        train_loss_contrast = 0.0
        train_loss_triplet = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for coords, labels_batch in pbar:
            coords = coords.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = mlp(coords)
            logits = classifier(embeddings)
            
            # Classification loss
            loss_cls = criterion_cls(logits, labels_batch)
            loss = loss_cls
            train_loss_cls += loss_cls.item()
            
            # Contrastive loss
            if use_contrastive:
                loss_contrast = criterion_contrastive(embeddings, labels_batch)
                loss = loss + contrastive_weight * loss_contrast
                train_loss_contrast += loss_contrast.item()
            
            # Triplet loss
            if use_triplet:
                loss_triplet = criterion_triplet(embeddings, labels_batch)
                loss = loss + triplet_weight * loss_triplet
                train_loss_triplet += loss_triplet.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_loss_cls /= len(train_loader)
        train_loss_contrast /= len(train_loader)
        train_loss_triplet /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        mlp.eval()
        classifier.eval()
        val_loss = 0.0
        val_loss_cls = 0.0
        val_loss_contrast = 0.0
        val_loss_triplet = 0.0
        val_correct = 0
        val_total = 0
        embedding_stds = []
        embedding_separations = []
        
        with torch.no_grad():
            for coords, labels_batch in val_loader:
                coords = coords.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                embeddings = mlp(coords)
                logits = classifier(embeddings)
                
                # Classification loss
                loss_cls = criterion_cls(logits, labels_batch)
                loss = loss_cls
                val_loss_cls += loss_cls.item()
                
                # Contrastive loss
                if use_contrastive:
                    loss_contrast = criterion_contrastive(embeddings, labels_batch)
                    loss = loss + contrastive_weight * loss_contrast
                    val_loss_contrast += loss_contrast.item()
                
                # Triplet loss
                if use_triplet:
                    loss_triplet = criterion_triplet(embeddings, labels_batch)
                    loss = loss + triplet_weight * loss_triplet
                    val_loss_triplet += loss_triplet.item()
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
                
                # Track embedding quality
                embedding_stds.append(embeddings.std(dim=0).mean().item())
                
                # Measure separation between different conformations
                if embeddings.shape[0] > 1:
                    dist_matrix = torch.cdist(embeddings, embeddings)
                    labels_diff = labels_batch.unsqueeze(0) != labels_batch.unsqueeze(1)
                    if labels_diff.sum() > 0:
                        inter_class_dist = dist_matrix[labels_diff].mean().item()
                        embedding_separations.append(inter_class_dist)
        
        val_loss /= len(val_loader)
        val_loss_cls /= len(val_loader)
        val_loss_contrast /= len(val_loader)
        val_loss_triplet /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        avg_emb_std = np.mean(embedding_stds)
        avg_separation = np.mean(embedding_separations) if embedding_separations else 0.0
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}:")
            print(f"  Train: Loss={train_loss:.4f} (cls={train_loss_cls:.4f}", end="")
            if use_contrastive:
                print(f", contrast={train_loss_contrast:.4f}", end="")
            if use_triplet:
                print(f", triplet={train_loss_triplet:.4f}", end="")
            print(f"), Acc={train_acc:.1f}%")
            
            print(f"  Val:   Loss={val_loss:.4f} (cls={val_loss_cls:.4f}", end="")
            if use_contrastive:
                print(f", contrast={val_loss_contrast:.4f}", end="")
            if use_triplet:
                print(f", triplet={val_loss_triplet:.4f}", end="")
            print(f"), Acc={val_acc:.1f}%")
            
            print(f"  Embedding: Std={avg_emb_std:.4f}, Separation={avg_separation:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'mlp_state_dict': mlp.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': embedding_dim,
                'avg_emb_std': avg_emb_std,
                'avg_separation': avg_separation,
            }, output_weights)
            if epoch % 10 == 0:
                print(f"  → Saved best model (val_acc={val_acc:.1f}%)")
        
        # Save periodic checkpoints
        if (epoch + 1) % save_frequency == 0:
            checkpoint_path = output_weights.replace('.pt', f'_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'mlp_state_dict': mlp.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': embedding_dim,
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
    mlp.load_state_dict(checkpoint['mlp_state_dict'])
    mlp.eval()
    
    with torch.no_grad():
        # Check embeddings for original models (no rotation)
        test_models = models[:min(10, num_models)].to(device)
        test_embeddings = mlp(test_models)
        
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
        
        # Compare with stored metrics
        print(f"\nMetrics from best epoch:")
        print(f"  Embedding std: {checkpoint.get('avg_emb_std', 'N/A'):.4f}" if 'avg_emb_std' in checkpoint else "")
        print(f"  Average separation: {checkpoint.get('avg_separation', 'N/A'):.4f}" if 'avg_separation' in checkpoint else "")
        
        # Check if embeddings are discriminative
        std_threshold = 0.1
        sep_threshold = 0.5
        
        print(f"\n{'Quality Assessment':^60}")
        print("-"*60)
        
        emb_std = test_embeddings.std(dim=0).mean().item()
        min_dist = pairwise_dist_no_diag.min().item()
        
        if emb_std > std_threshold:
            print(f"✓ PASS: Embedding diversity = {emb_std:.4f} > {std_threshold}")
        else:
            print(f"✗ FAIL: Embedding diversity = {emb_std:.4f} < {std_threshold}")
        
        if min_dist > sep_threshold:
            print(f"✓ PASS: Min separation = {min_dist:.4f} > {sep_threshold}")
        else:
            print(f"⚠ WARNING: Min separation = {min_dist:.4f} < {sep_threshold}")
        
        if emb_std > std_threshold and min_dist > sep_threshold:
            print(f"\n✓✓✓ EXCELLENT: Embeddings are highly discriminative!")
        elif emb_std > std_threshold/2:
            print(f"\n✓✓ GOOD: Embeddings show reasonable discrimination")
        else:
            print(f"\n✗✗ POOR: Embeddings may have collapsed")
    
    print(f"{'='*60}\n")
    
    return mlp, train_losses, val_losses, train_accs, val_accs


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train MLP_Enhanced embedding with contrastive/triplet loss for similar conformations",
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
        help="Path to save trained MLP weights (.pt file)"
    )
    
    # Model architecture
    parser.add_argument(
        "--embedding_dim", 
        type=int, 
        default=256,
        help="Output dimension of MLP embedding"
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
    
    # Loss configuration
    parser.add_argument(
        "--use_contrastive",
        action="store_true",
        default=True,
        help="Use contrastive loss (recommended for similar conformations)"
    )
    parser.add_argument(
        "--no_contrastive",
        action="store_true",
        help="Disable contrastive loss"
    )
    parser.add_argument(
        "--use_triplet",
        action="store_true",
        help="Use triplet loss (can combine with contrastive)"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.5,
        help="Weight for contrastive loss"
    )
    parser.add_argument(
        "--triplet_weight",
        type=float,
        default=0.3,
        help="Weight for triplet loss"
    )
    parser.add_argument(
        "--contrastive_margin",
        type=float,
        default=2.0,
        help="Margin for contrastive loss (minimum distance between different conformations)"
    )
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=1.0,
        help="Margin for triplet loss"
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
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (0, 1, etc.). If not specified, uses default GPU or CPU"
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
    
    # Handle GPU selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            if args.gpu is not None:
                if args.gpu >= torch.cuda.device_count():
                    print(f"ERROR: GPU {args.gpu} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
                    sys.exit(1)
                device = f"cuda:{args.gpu}"
                torch.cuda.set_device(args.gpu)
                print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
            else:
                device = "cuda"
                print(f"Using default GPU: {torch.cuda.get_device_name(0)}")
            
            # Print GPU info
            print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
    
    # Handle contrastive flag
    use_contrastive = args.use_contrastive and not args.no_contrastive
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Model file: {args.model_file}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"\nLoss configuration:")
    print(f"  Classification loss: ✓ (always enabled)")
    if use_contrastive:
        print(f"  Contrastive loss: ✓ (weight={args.contrastive_weight}, margin={args.contrastive_margin})")
    else:
        print(f"  Contrastive loss: ✗ (disabled)")
    if args.use_triplet:
        print(f"  Triplet loss: ✓ (weight={args.triplet_weight}, margin={args.triplet_margin})")
    else:
        print(f"  Triplet loss: ✗ (disabled)")
    print(f"\nData augmentation:")
    print(f"  Samples per model: {args.samples_per_model}")
    print(f"  Add noise: {args.add_noise}")
    if args.add_noise:
        print(f"  Noise std: {args.noise_std}")
    print("="*60 + "\n")
    
    # Run pre-training
    try:
        pretrain_mlp_enhanced(
            model_file=args.model_file,
            output_weights=args.output,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,  # Pass the full device string
            samples_per_model=args.samples_per_model,
            add_noise=args.add_noise,
            noise_std=args.noise_std,
            num_workers=args.num_workers,
            save_frequency=args.save_frequency,
            seed=args.seed,
            use_contrastive=use_contrastive,
            use_triplet=args.use_triplet,
            contrastive_weight=args.contrastive_weight,
            triplet_weight=args.triplet_weight,
            contrastive_margin=args.contrastive_margin,
            triplet_margin=args.triplet_margin
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
