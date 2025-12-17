"""
compare_embeddings.py

Compare embeddings of synthetic and real cryo-EM images.
Checks for overlap in latent space using multiple visualization and quantification methods.

FIXED VERSION: Metrics now correctly detect distribution separation that matches visual plots!

Usage:
    python compare_embeddings.py \
        --pretrained_weights pretrained_spatial_cryo.pt \
        --image_config config.json \
        --real_images real_particles.mrcs \
        --embedding SPATIAL_CRYO \
        --embedding_dim 256 \
        --n_synthetic 5000 \
        --n_real 5000 \
        --output_dir embedding_comparison
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde, spearmanr
import mrcfile

# Optional: UMAP for better visualization
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not available. Install with: pip install umap-learn")

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pretrained_encoder(weights_path, embedding_name, embedding_dim, image_size, device):
    """Load pretrained encoder"""
    print(f"\nLoading pretrained encoder...")
    print(f"  Architecture: {embedding_name}")
    print(f"  Weights: {weights_path}")
    
    if embedding_name not in EMBEDDING_NETS:
        raise ValueError(f"Unknown embedding: {embedding_name}")
    
    encoder = EMBEDDING_NETS[embedding_name](embedding_dim, D=image_size).to(device)
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.eval()
    
    print(f"  ✅ Loaded successfully")
    return encoder


def normalize_images(images, method='per_image'):
    """
    Ensure consistent normalization between synthetic and real images
    
    Args:
        images: torch.Tensor of shape (N, H, W)
        method: 'per_image', 'global', or 'none'
    """
    if method == 'per_image':
        # Per-image standardization (common in cryo-EM)
        mean = images.mean(dim=(1, 2), keepdim=True)
        std = images.std(dim=(1, 2), keepdim=True)
        return (images - mean) / (std + 1e-8)
    elif method == 'global':
        # Global standardization
        return (images - images.mean()) / (images.std() + 1e-8)
    elif method == 'none':
        return images
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def generate_synthetic_images(image_config, models, n_images, device):
    """Generate synthetic images using the simulator"""
    print(f"\nGenerating {n_images} synthetic images...")
    
    # Setup
    image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    prior_loader = PriorLoader(image_prior, batch_size=min(1024, n_images), num_workers=4)
    
    num_pixels = torch.tensor(image_config["N_PIXELS"], dtype=torch.float32, device=device)
    pixel_size = torch.tensor(image_config["PIXEL_SIZE"], dtype=torch.float32, device=device)
    voltage = image_config.get("VOLTAGE", 300.0)
    cs = image_config.get("SPHERICAL_ABERRATION", 0.0)
    
    all_images = []
    all_params = []
    
    with torch.no_grad():
        for parameters in tqdm(prior_loader, desc="Simulating"):
            (indices, quaternions, res, shift, defocus, b_factor, amp, snr) = parameters
            
            # Simulate batch
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
            
            all_images.append(images.cpu())
            all_params.append({
                'indices': indices,
                'quaternions': quaternions,
                'defocus': defocus,
                'snr': snr
            })
            
            if sum(img.shape[0] for img in all_images) >= n_images:
                break
    
    # Concatenate and truncate
    all_images = torch.cat(all_images, dim=0)[:n_images]
    
    print(f"  ✅ Generated {len(all_images)} images")
    print(f"  Shape: {all_images.shape}")
    print(f"  Range: [{all_images.min():.3f}, {all_images.max():.3f}]")
    
    return all_images, all_params


def load_real_images(mrc_path, n_images=None, normalize=False):
    """Load real images from MRC particle stack"""
    print(f"\nLoading real images from: {mrc_path}")
    
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        images = mrc.data
        
        # Handle different MRC formats
        if images.ndim == 2:
            images = images[np.newaxis, ...]  # Single image
        
        print(f"  Total particles in stack: {len(images)}")
        
        # Subsample if requested
        if n_images is not None and n_images < len(images):
            indices = np.random.choice(len(images), n_images, replace=False)
            images = images[indices]
            print(f"  Randomly selected: {n_images}")
        
        # Convert to torch
        images = torch.from_numpy(images.copy()).float()
        
        print(f"  ✅ Loaded {len(images)} images")
        print(f"  Shape: {images.shape}")
        print(f"  Range (before norm): [{images.min():.3f}, {images.max():.3f}]")
        
        return images


# ============================================================================
# EMBEDDING
# ============================================================================

def embed_images(encoder, images, batch_size=512, device='cuda'):
    """Embed images in batches"""
    print(f"\nEmbedding {len(images)} images...")
    
    embeddings = []
    encoder.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding"):
            batch = images[i:i+batch_size].to(device)
            emb = encoder(batch)
            embeddings.append(emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    
    print(f"  ✅ Embedding shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    
    return embeddings


# ============================================================================
# NEW: SEPARATION METRICS (FIXED!)
# ============================================================================

def compute_separation_metrics(synthetic_emb, real_emb):
    """
    Compute metrics that actually detect if distributions occupy different regions
    (matching what you see in PCA/t-SNE/UMAP plots!)
    """
    print("\n" + "="*70)
    print("SEPARATION ANALYSIS (VISUAL-ALIGNED METRICS)")
    print("="*70)
    
    metrics = {}
    
    # Prepare data
    X_syn = synthetic_emb.numpy()
    X_real = real_emb.numpy()
    X_combined = np.vstack([X_syn, X_real])
    y = np.array([0]*len(X_syn) + [1]*len(X_real))
    
    # ============================================================
    # 1. CLASSIFIER-BASED SEPARATION (Gold Standard!)
    # ============================================================
    print("\n1. 🎯 Classifier Two-Sample Test")
    print("   (Can a linear classifier distinguish synthetic from real?)")
    
    # Shuffle
    indices = np.random.permutation(len(X_combined))
    X_shuffled, y_shuffled = X_combined[indices], y[indices]
    
    # Train classifier with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_shuffled, y_shuffled, cv=5, scoring='accuracy')
    
    accuracy = scores.mean()
    accuracy_std = scores.std()
    metrics['classifier_accuracy'] = accuracy
    metrics['classifier_std'] = accuracy_std
    
    print(f"\n   Accuracy: {accuracy*100:.1f}% ± {accuracy_std*100:.1f}% (baseline=50%)")
    
    if accuracy < 0.55:
        print("   ✅ EXCELLENT - Cannot distinguish (< 55%)")
        separation_level = "NONE"
    elif accuracy < 0.65:
        print("   🟡 GOOD - Slight distinguishability (55-65%)")
        separation_level = "SLIGHT"
    elif accuracy < 0.80:
        print("   ⚠️  MODERATE - Clearly distinguishable (65-80%)")
        separation_level = "MODERATE"
    else:
        print("   ❌ STRONG - Highly separated (> 80%)")
        separation_level = "STRONG"
    
    metrics['separation_level'] = separation_level
    
    # ============================================================
    # 2. PCA-BASED SEPARATION (What you see in plots!)
    # ============================================================
    print("\n2. 📊 PCA Separation Analysis")
    
    # Fit PCA on combined data
    pca = PCA(n_components=min(10, X_combined.shape[1]))
    X_pca = pca.fit_transform(X_combined)
    
    # Split back
    X_syn_pca = X_pca[:len(X_syn)]
    X_real_pca = X_pca[len(X_syn):]
    
    # Compute separation in top PCs
    separations = {}
    for n_components in [2, 5, 10]:
        if n_components > X_pca.shape[1]:
            continue
            
        mean_syn = X_syn_pca[:, :n_components].mean(axis=0)
        mean_real = X_real_pca[:, :n_components].mean(axis=0)
        
        std_syn = X_syn_pca[:, :n_components].std()
        std_real = X_real_pca[:, :n_components].std()
        
        # Normalized separation (in units of std)
        centroid_distance = np.linalg.norm(mean_syn - mean_real)
        pooled_std = (std_syn + std_real) / 2
        
        normalized_separation = centroid_distance / pooled_std
        separations[f'pc{n_components}'] = normalized_separation
        
        print(f"   PC1-{n_components}: {normalized_separation:.2f} standard deviations")
    
    metrics['pca2d_separation'] = separations.get('pc2', 0)
    metrics['pca_explained_variance'] = pca.explained_variance_ratio_[:2].sum()
    
    # Interpretation for 2D
    sep_2d = metrics['pca2d_separation']
    if sep_2d < 0.5:
        print(f"\n   ✅ Overlapping in 2D PCA (< 0.5σ)")
    elif sep_2d < 1.0:
        print(f"\n   🟡 Adjacent in 2D PCA (0.5-1.0σ)")
    elif sep_2d < 2.0:
        print(f"\n   ⚠️  Separated in 2D PCA (1.0-2.0σ)")
    else:
        print(f"\n   ❌ Strongly separated in 2D PCA (> 2.0σ)")
    
    # ============================================================
    # 3. SPATIAL OVERLAP (Distribution overlap in embedding space)
    # ============================================================
    print("\n3. 🗺️  Spatial Overlap Analysis")
    
    # For each point, find nearest neighbor in full dataset
    # Check if NN is from same or different distribution
    
    nn_all = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_all.fit(X_combined)
    
    # Check synthetic points (use subset for speed)
    n_samples = min(1000, len(X_syn))
    distances_syn, indices_syn = nn_all.kneighbors(X_syn[:n_samples])
    # Is nearest neighbor (excluding self) from real distribution?
    nn_is_real = indices_syn[:, 1] >= len(X_syn)
    overlap_from_syn = nn_is_real.mean()
    
    # Check real points
    n_samples = min(1000, len(X_real))
    distances_real, indices_real = nn_all.kneighbors(X_real[:n_samples])
    nn_is_syn = indices_real[:, 1] < len(X_syn)
    overlap_from_real = nn_is_syn.mean()
    
    overlap_score = (overlap_from_syn + overlap_from_real) / 2
    metrics['spatial_overlap'] = overlap_score
    
    print(f"\n   Synthetic points with real NN: {overlap_from_syn*100:.1f}%")
    print(f"   Real points with synthetic NN: {overlap_from_real*100:.1f}%")
    print(f"   Average spatial overlap: {overlap_score*100:.1f}%")
    
    if overlap_score > 0.7:
        print("   ✅ High spatial overlap (> 70%)")
    elif overlap_score > 0.4:
        print("   🟡 Moderate spatial overlap (40-70%)")
    elif overlap_score > 0.2:
        print("   ⚠️  Low spatial overlap (20-40%)")
    else:
        print("   ❌ Minimal spatial overlap (< 20%)")
    
    # ============================================================
    # 4. MAXIMUM MEAN DISCREPANCY (Distribution distance)
    # ============================================================
    print("\n4. 📏 Maximum Mean Discrepancy (MMD)")
    
    def compute_mmd_rbf(X, Y, gamma=None):
        """MMD with RBF kernel"""
        n, m = len(X), len(Y)
        
        if gamma is None:
            # Median heuristic
            combined = np.vstack([X, Y])
            gamma = 1.0 / (2 * np.median(np.var(combined, axis=0)) + 1e-8)
        
        def rbf_kernel(A, B, gamma):
            """Compute RBF kernel efficiently"""
            # Subsample if too large
            if len(A) > 500:
                A = A[np.random.choice(len(A), 500, replace=False)]
            if len(B) > 500:
                B = B[np.random.choice(len(B), 500, replace=False)]
            
            dist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * dist)
        
        XX = rbf_kernel(X, X, gamma).mean()
        YY = rbf_kernel(Y, Y, gamma).mean()
        XY = rbf_kernel(X, Y, gamma).mean()
        
        return XX + YY - 2 * XY
    
    # Use PCA space for efficiency
    mmd = compute_mmd_rbf(X_syn_pca[:1000], X_real_pca[:1000])
    metrics['mmd'] = mmd
    
    print(f"\n   MMD: {mmd:.6f}")
    print(f"   (Lower is better, 0 = identical distributions)")
    
    # ============================================================
    # 5. 2D VISUAL OVERLAP (Bounding box in 2D projection)
    # ============================================================
    print("\n5. 👁️  Visual Overlap (2D PCA bounding boxes)")
    
    # This directly measures what you see in scatter plots!
    syn_min = X_syn_pca[:, :2].min(axis=0)
    syn_max = X_syn_pca[:, :2].max(axis=0)
    real_min = X_real_pca[:, :2].min(axis=0)
    real_max = X_real_pca[:, :2].max(axis=0)
    
    # Overlap in each dimension
    overlap_dims = []
    for i in range(2):
        overlap = max(0, min(syn_max[i], real_max[i]) - max(syn_min[i], real_min[i]))
        total_range = max(syn_max[i], real_max[i]) - min(syn_min[i], real_min[i])
        overlap_ratio = overlap / total_range if total_range > 0 else 0
        overlap_dims.append(overlap_ratio)
    
    visual_overlap = np.mean(overlap_dims)
    metrics['visual_overlap_2d'] = visual_overlap
    
    print(f"\n   PC1 overlap: {overlap_dims[0]*100:.1f}%")
    print(f"   PC2 overlap: {overlap_dims[1]*100:.1f}%")
    print(f"   Average 2D visual overlap: {visual_overlap*100:.1f}%")
    
    if visual_overlap > 0.6:
        print("   ✅ Strong visual overlap")
    elif visual_overlap > 0.3:
        print("   🟡 Partial visual overlap")
    else:
        print("   ❌ Minimal visual overlap (explains separation in plots!)")
    
    return metrics


# ============================================================================
# LEGACY METRICS (for comparison)
# ============================================================================

def compute_legacy_statistics(synthetic_emb, real_emb):
    """OLD distance-based metrics (less reliable for detecting separation)"""
    print("\n" + "="*70)
    print("LEGACY DISTANCE METRICS")
    print("(Note: These can be misleading for separated distributions)")
    print("="*70)
    
    stats = {}
    
    # Basic statistics
    print(f"\nBasic statistics:")
    print(f"  Synthetic - Mean: {synthetic_emb.mean():.4f}, Std: {synthetic_emb.std():.4f}")
    print(f"  Real      - Mean: {real_emb.mean():.4f}, Std: {real_emb.std():.4f}")
    
    # Distance statistics
    print(f"\nPairwise distance statistics:")
    
    # Within-group distances (sample for speed)
    n_samples = min(1000, len(synthetic_emb), len(real_emb))
    
    syn_sample = synthetic_emb[:n_samples]
    syn_dists = torch.cdist(syn_sample, syn_sample)
    syn_dists = syn_dists[torch.triu(torch.ones_like(syn_dists), diagonal=1) == 1]
    stats['synthetic_dist_mean'] = syn_dists.mean().item()
    stats['synthetic_dist_std'] = syn_dists.std().item()
    
    real_sample = real_emb[:n_samples]
    real_dists = torch.cdist(real_sample, real_sample)
    real_dists = real_dists[torch.triu(torch.ones_like(real_dists), diagonal=1) == 1]
    stats['real_dist_mean'] = real_dists.mean().item()
    stats['real_dist_std'] = real_dists.std().item()
    
    print(f"  Within synthetic: {stats['synthetic_dist_mean']:.4f} ± {stats['synthetic_dist_std']:.4f}")
    print(f"  Within real:      {stats['real_dist_mean']:.4f} ± {stats['real_dist_std']:.4f}")
    
    # Cross-group distances
    cross_dists = torch.cdist(syn_sample, real_sample)
    stats['cross_dist_mean'] = cross_dists.mean().item()
    stats['cross_dist_std'] = cross_dists.std().item()
    
    print(f"  Synthetic-Real:   {stats['cross_dist_mean']:.4f} ± {stats['cross_dist_std']:.4f}")
    
    # OLD overlap ratio (can be misleading!)
    within_mean = np.mean([stats['synthetic_dist_mean'], stats['real_dist_mean']])
    overlap_ratio = stats['cross_dist_mean'] / within_mean
    stats['legacy_overlap_ratio'] = overlap_ratio
    
    print(f"\n⚠️  Legacy overlap ratio: {overlap_ratio:.4f}")
    print(f"    (This metric can miss spatial separation!)")
    
    return stats


# ============================================================================
# ENHANCED VISUALIZATIONS
# ============================================================================

def create_enhanced_visualizations(synthetic_emb, real_emb, output_dir):
    """Create visualizations with separation metrics overlaid"""
    print("\n" + "="*70)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_syn = synthetic_emb.numpy()
    X_real = real_emb.numpy()
    X_combined = np.vstack([X_syn, X_real])
    labels = np.array(['Synthetic']*len(X_syn) + ['Real']*len(X_real))
    
    viz_metrics = {}
    
    # ============================================================
    # 1. Enhanced PCA with quantitative annotations
    # ============================================================
    print("\n1. Enhanced PCA visualization...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    X_syn_pca = X_pca[:len(X_syn)]
    X_real_pca = X_pca[len(X_syn):]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Panel 1: Standard scatter with separation line
    ax = axes[0]
    
    ax.scatter(X_syn_pca[:, 0], X_syn_pca[:, 1], 
               alpha=0.3, s=1, label='Synthetic', c='blue')
    ax.scatter(X_real_pca[:, 0], X_real_pca[:, 1], 
               alpha=0.3, s=1, label='Real', c='red')
    
    # Centroids
    centroid_syn = X_syn_pca.mean(axis=0)
    centroid_real = X_real_pca.mean(axis=0)
    
    ax.scatter(*centroid_syn, s=300, c='blue', marker='X', 
               edgecolors='black', linewidths=2, label='Syn Centroid', zorder=5)
    ax.scatter(*centroid_real, s=300, c='red', marker='X', 
               edgecolors='black', linewidths=2, label='Real Centroid', zorder=5)
    
    # Separation line
    ax.plot([centroid_syn[0], centroid_real[0]], 
            [centroid_syn[1], centroid_real[1]], 
            'k--', linewidth=2, alpha=0.7)
    
    # Calculate separation
    separation = np.linalg.norm(centroid_syn - centroid_real)
    pooled_std = (X_syn_pca.std() + X_real_pca.std()) / 2
    normalized_sep = separation / pooled_std
    
    viz_metrics['pca_separation'] = normalized_sep
    
    # Annotate
    ax.text(0.05, 0.95, 
            f'Separation: {normalized_sep:.2f}σ\nExplained: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%', 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA: Centroid Separation')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Density visualization
    ax = axes[1]
    
    try:
        from scipy.stats import gaussian_kde
        
        # Synthetic density
        if len(X_syn_pca) > 5000:
            idx = np.random.choice(len(X_syn_pca), 5000, replace=False)
            xy_syn = X_syn_pca[idx].T
        else:
            xy_syn = X_syn_pca.T
        
        z_syn = gaussian_kde(xy_syn)(xy_syn)
        scatter_syn = ax.scatter(xy_syn[0], xy_syn[1], c=z_syn, s=1, 
                                 cmap='Blues', alpha=0.5, label='Synthetic')
        
        # Real density
        if len(X_real_pca) > 5000:
            idx = np.random.choice(len(X_real_pca), 5000, replace=False)
            xy_real = X_real_pca[idx].T
        else:
            xy_real = X_real_pca.T
        
        z_real = gaussian_kde(xy_real)(xy_real)
        scatter_real = ax.scatter(xy_real[0], xy_real[1], c=z_real, s=1, 
                                  cmap='Reds', alpha=0.5, label='Real')
        
    except Exception as e:
        print(f"  ⚠️  Density plot failed: {e}")
        ax.scatter(X_syn_pca[:, 0], X_syn_pca[:, 1], alpha=0.3, s=1, c='blue')
        ax.scatter(X_real_pca[:, 0], X_real_pca[:, 1], alpha=0.3, s=1, c='red')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA: Density Visualization')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Marginal distributions
    ax = axes[2]
    ax.hist(X_syn_pca[:, 0], bins=50, alpha=0.5, label='Syn PC1', 
            density=True, color='blue')
    ax.hist(X_real_pca[:, 0], bins=50, alpha=0.5, label='Real PC1', 
            density=True, color='red')
    ax.hist(X_syn_pca[:, 1], bins=50, alpha=0.5, label='Syn PC2', 
            density=True, color='lightblue', linestyle='--')
    ax.hist(X_real_pca[:, 1], bins=50, alpha=0.5, label='Real PC2', 
            density=True, color='lightcoral', linestyle='--')
    ax.set_xlabel('PC Value')
    ax.set_ylabel('Density')
    ax.set_title('Marginal Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: pca_enhanced.png")
    print(f"     Separation: {normalized_sep:.2f}σ")

    # ============================================================
    # 2. t-SNE with overlap quantification
    # ============================================================
    print("\n2. t-SNE visualization...")
    
    # Use subset for speed
    n_samples_per_group = min(5000, len(X_syn), len(X_real))
    
    syn_idx = np.random.choice(len(X_syn), n_samples_per_group, replace=False)
    real_idx = np.random.choice(len(X_real), n_samples_per_group, replace=False)
    
    X_subset = np.vstack([X_syn[syn_idx], X_real[real_idx]])
    labels_subset = np.array(['Synthetic']*n_samples_per_group + 
                             ['Real']*n_samples_per_group)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Standard visualization
    ax = axes[0]
    mask_syn = labels_subset == 'Synthetic'
    ax.scatter(X_tsne[mask_syn, 0], X_tsne[mask_syn, 1],
               alpha=0.3, s=1, label='Synthetic', c='blue')
    ax.scatter(X_tsne[~mask_syn, 0], X_tsne[~mask_syn, 1],
               alpha=0.3, s=1, label='Real', c='red')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Projection')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: With convex hulls
    ax = axes[1]
    ax.scatter(X_tsne[mask_syn, 0], X_tsne[mask_syn, 1],
               alpha=0.3, s=1, label='Synthetic', c='blue')
    ax.scatter(X_tsne[~mask_syn, 0], X_tsne[~mask_syn, 1],
               alpha=0.3, s=1, label='Real', c='red')
    
    # Draw convex hulls
    try:
        from scipy.spatial import ConvexHull
        
        syn_tsne = X_tsne[mask_syn]
        real_tsne = X_tsne[~mask_syn]
        
        # Subsample for convex hull
        if len(syn_tsne) > 1000:
            hull_idx = np.random.choice(len(syn_tsne), 1000, replace=False)
            hull_syn = ConvexHull(syn_tsne[hull_idx])
            for simplex in hull_syn.simplices:
                ax.plot(syn_tsne[hull_idx][simplex, 0], 
                       syn_tsne[hull_idx][simplex, 1], 'b-', alpha=0.3)
        
        if len(real_tsne) > 1000:
            hull_idx = np.random.choice(len(real_tsne), 1000, replace=False)
            hull_real = ConvexHull(real_tsne[hull_idx])
            for simplex in hull_real.simplices:
                ax.plot(real_tsne[hull_idx][simplex, 0], 
                       real_tsne[hull_idx][simplex, 1], 'r-', alpha=0.3)
    except Exception as e:
        print(f"  ⚠️  Convex hull failed: {e}")
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE with Convex Hulls')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: tsne_projection.png")
    
    # ============================================================
    # 3. UMAP (if available)
    # ============================================================
    if HAS_UMAP:
        print("\n3. UMAP visualization...")
        
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        X_umap = reducer.fit_transform(X_subset)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.scatter(X_umap[mask_syn, 0], X_umap[mask_syn, 1],
                   alpha=0.3, s=1, label='Synthetic', c='blue')
        ax.scatter(X_umap[~mask_syn, 0], X_umap[~mask_syn, 1],
                   alpha=0.3, s=1, label='Real', c='red')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Projection')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_projection.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: umap_projection.png")
    
    # ============================================================
    # 4. Distance distributions
    # ============================================================
    print("\n4. Distance distribution comparison...")
    
    # Sample for speed
    n_dist_samples = min(2000, len(synthetic_emb), len(real_emb))
    syn_sample = synthetic_emb[:n_dist_samples]
    real_sample = real_emb[:n_dist_samples]
    
    # Within-group distances
    syn_dists = torch.cdist(syn_sample, syn_sample)
    syn_dists = syn_dists[torch.triu(torch.ones_like(syn_dists), diagonal=1) == 1].numpy()
    
    real_dists = torch.cdist(real_sample, real_sample)
    real_dists = real_dists[torch.triu(torch.ones_like(real_dists), diagonal=1) == 1].numpy()
    
    # Cross-group distances
    cross_dists = torch.cdist(syn_sample, real_sample).flatten().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Overlaid histograms
    ax = axes[0]
    ax.hist(syn_dists, bins=50, alpha=0.5, label='Within Synthetic', 
            density=True, color='blue')
    ax.hist(real_dists, bins=50, alpha=0.5, label='Within Real', 
            density=True, color='red')
    ax.hist(cross_dists, bins=50, alpha=0.5, label='Synthetic-Real', 
            density=True, color='purple')
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Density')
    ax.set_title('Pairwise Distance Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Box plots
    ax = axes[1]
    data_to_plot = [syn_dists, real_dists, cross_dists]
    bp = ax.boxplot(data_to_plot, labels=['Within\nSynthetic', 'Within\nReal', 'Cross\nSyn-Real'],
                    patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Distance Distribution Summary')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: distance_distributions.png")
    
    # ============================================================
    # 5. Per-dimension statistics
    # ============================================================
    print("\n5. Per-dimension comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    syn_means = synthetic_emb.mean(dim=0).numpy()
    real_means = real_emb.mean(dim=0).numpy()
    syn_stds = synthetic_emb.std(dim=0).numpy()
    real_stds = real_emb.std(dim=0).numpy()
    
    # Means scatter
    ax = axes[0, 0]
    ax.scatter(syn_means, real_means, alpha=0.5, s=10)
    lim = max(np.abs(syn_means).max(), np.abs(real_means).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('Synthetic Mean')
    ax.set_ylabel('Real Mean')
    ax.set_title('Per-Dimension Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stds scatter
    ax = axes[0, 1]
    ax.scatter(syn_stds, real_stds, alpha=0.5, s=10)
    lim = max(syn_stds.max(), real_stds.max())
    ax.plot([0, lim], [0, lim], 'r--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('Synthetic Std')
    ax.set_ylabel('Real Std')
    ax.set_title('Per-Dimension Standard Deviations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean histogram
    ax = axes[1, 0]
    ax.hist(syn_means, bins=30, alpha=0.5, label='Synthetic', color='blue')
    ax.hist(real_means, bins=30, alpha=0.5, label='Real', color='red')
    ax.set_xlabel('Mean Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dimension Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std histogram
    ax = axes[1, 1]
    ax.hist(syn_stds, bins=30, alpha=0.5, label='Synthetic', color='blue')
    ax.hist(real_stds, bins=30, alpha=0.5, label='Real', color='red')
    ax.set_xlabel('Std Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dimension Stds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: dimension_statistics.png")
    
    # ============================================================
    # 6. PCA scree plot
    # ============================================================
    print("\n6. PCA variance analysis...")
    
    pca_full = PCA(n_components=min(50, X_combined.shape[1]))
    pca_full.fit(X_combined)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variance explained
    ax = axes[0]
    ax.plot(range(1, len(pca_full.explained_variance_ratio_)+1), 
            pca_full.explained_variance_ratio_, 'bo-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('Scree Plot')
    ax.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax = axes[1]
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum)+1), cumsum, 'ro-')
    ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Cumulative Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_scree.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: pca_scree.png")
    
    return viz_metrics


def visualize_sample_images(synthetic_imgs, real_imgs, output_dir, n_samples=10):
    """Show sample images from each group"""
    print("\n7. Sample images...")
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    
    # Synthetic
    for i in range(n_samples):
        axes[0, i].imshow(synthetic_imgs[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Synthetic', fontsize=12, loc='left')
    
    # Real
    for i in range(n_samples):
        axes[1, i].imshow(real_imgs[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Real', fontsize=12, loc='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: sample_images.png")


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(separation_metrics, viz_metrics, legacy_stats, output_dir):
    """Generate a comprehensive markdown report"""
    
    report = f"""# Cryo-EM Embedding Comparison Report

## Executive Summary

### 🎯 Primary Metrics (Most Reliable)

**Classifier Two-Sample Test**
- Accuracy: **{separation_metrics['classifier_accuracy']*100:.1f}%** ± {separation_metrics['classifier_std']*100:.1f}%
- Baseline: 50% (random guessing)
- Result: **{separation_metrics['separation_level']}** separation

**Interpretation:**
"""
    
    acc = separation_metrics['classifier_accuracy']
    if acc < 0.55:
        report += "✅ **EXCELLENT** - Synthetic and real embeddings are indistinguishable\n"
        report += "   The simulator produces realistic data that matches the real distribution.\n"
    elif acc < 0.65:
        report += "🟡 **GOOD** - Minor differences exist but substantial overlap\n"
        report += "   The simulator is reasonable with some distributional mismatch.\n"
    elif acc < 0.80:
        report += "⚠️  **POOR** - Clear separation between synthetic and real\n"
        report += "   The simulator has significant discrepancies from real data.\n"
    else:
        report += "❌ **FAILED** - Synthetic and real are strongly separated\n"
        report += "   Major mismatch - check simulator parameters and preprocessing.\n"
    
    report += f"""

---

## Detailed Metrics

### Spatial Separation Analysis

**PCA-Based Separation**
- 2D Centroid Distance: **{separation_metrics['pca2d_separation']:.2f}σ**
- Variance Explained (PC1-2): {separation_metrics['pca_explained_variance']*100:.1f}%

Interpretation: 
"""
    
    sep = separation_metrics['pca2d_separation']
    if sep < 0.5:
        report += "✅ Centroids overlap in PCA space\n"
    elif sep < 1.0:
        report += "🟡 Centroids are adjacent but distinguishable\n"
    elif sep < 2.0:
        report += "⚠️  Centroids are clearly separated (matches visual observation!)\n"
    else:
        report += "❌ Centroids are far apart in PCA space\n"
    
    report += f"""

**Spatial Overlap**
- Average Spatial Overlap: **{separation_metrics['spatial_overlap']*100:.1f}%**
- (Fraction of points with nearest neighbor from other distribution)

"""
    
    if separation_metrics['spatial_overlap'] > 0.7:
        report += "✅ High spatial overlap - distributions occupy same regions\n"
    elif separation_metrics['spatial_overlap'] > 0.4:
        report += "🟡 Moderate spatial overlap\n"
    else:
        report += "❌ Low spatial overlap - distributions occupy different regions\n"
    
    report += f"""

**Visual Overlap (2D Projections)**
- 2D Bounding Box Overlap: **{separation_metrics['visual_overlap_2d']*100:.1f}%**

"""
    
    if separation_metrics['visual_overlap_2d'] > 0.6:
        report += "✅ Strong visual overlap in scatter plots\n"
    elif separation_metrics['visual_overlap_2d'] > 0.3:
        report += "🟡 Partial visual overlap\n"
    else:
        report += "❌ Minimal visual overlap (explains separation you see in plots!)\n"
    
    report += f"""

**Maximum Mean Discrepancy**
- MMD (RBF kernel): **{separation_metrics['mmd']:.6f}**
- (Lower is better, 0 = identical distributions)

---

## Legacy Distance Metrics
*(Note: These can be misleading for spatially separated distributions)*

- Within Synthetic: {legacy_stats['synthetic_dist_mean']:.4f} ± {legacy_stats['synthetic_dist_std']:.4f}
- Within Real: {legacy_stats['real_dist_mean']:.4f} ± {legacy_stats['real_dist_std']:.4f}
- Cross (Syn-Real): {legacy_stats['cross_dist_mean']:.4f} ± {legacy_stats['cross_dist_std']:.4f}
- Legacy Overlap Ratio: {legacy_stats['legacy_overlap_ratio']:.4f}

⚠️  **Warning:** The legacy overlap ratio can show "good overlap" even when distributions 
are spatially separated! Use the classifier accuracy and PCA separation metrics instead.

---

## Recommendations

"""
    
    if acc < 0.60 and sep < 1.0:
        report += """
✅ **Simulator is working well!**
- Proceed with inference using synthetic data
- The learned embeddings capture real data characteristics
"""
    elif acc < 0.70:
        report += """
🟡 **Simulator is acceptable but could be improved**
- Check if preprocessing (normalization, CTF correction) matches between synthetic and real
- Consider retraining the embedding network with mixed synthetic-real data
- Results should be validated carefully
"""
    else:
        report += """
❌ **Simulator needs significant improvement**

Possible issues:
1. **Preprocessing mismatch**: Different normalization between synthetic and real
2. **CTF parameters**: Defocus range or CTF correction differs
3. **Noise model**: SNR or noise characteristics don't match real data
4. **Missing effects**: Real data may have artifacts not in simulator
5. **Conformational sampling**: Synthetic may not cover full conformational space

Recommended actions:
1. Verify preprocessing pipeline matches exactly
2. Check CTF parameter ranges match real data
3. Inspect sample images side-by-side
4. Consider domain adaptation techniques
5. Retrain embedding with real data if possible
"""
    
    report += f"""

---

## Visualizations

See the following files in the output directory:
- `pca_enhanced.png` - PCA projection with separation metrics
- `tsne_projection.png` - t-SNE visualization
"""
    
    if HAS_UMAP:
        report += "- `umap_projection.png` - UMAP visualization\n"
    
    report += """- `distance_distributions.png` - Pairwise distance distributions
- `dimension_statistics.png` - Per-dimension statistical comparison
- `pca_scree.png` - Variance explained by principal components
- `sample_images.png` - Side-by-side image comparison

---

*Report generated by compare_embeddings.py (Fixed Version)*
"""
    
    # Save report
    report_path = os.path.join(output_dir, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n  ✅ Saved comprehensive report: REPORT.md")
    
    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare embeddings of synthetic and real cryo-EM images (FIXED VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compare_embeddings.py \\
      --pretrained_weights weights.pt \\
      --image_config config.json \\
      --real_images particles.mrcs
  
  # With custom parameters
  python compare_embeddings.py \\
      --pretrained_weights weights.pt \\
      --image_config config.json \\
      --real_images particles.mrcs \\
      --n_synthetic 10000 \\
      --n_real 10000 \\
      --normalization per_image \\
      --output_dir results
        """
    )
    
    # Required
    parser.add_argument('--pretrained_weights', type=str, required=True,
                       help='Path to pretrained encoder weights')
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to image config JSON')
    parser.add_argument('--real_images', type=str, required=True,
                       help='Path to real particle stack (MRC/MRCS)')
    
    # Model
    parser.add_argument('--embedding', type=str, default='SPATIAL_CRYO_FFT_FILTER',
                       help='Embedding architecture')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    
    # Data
    parser.add_argument('--n_synthetic', type=int, default=5000,
                       help='Number of synthetic images to generate')
    parser.add_argument('--n_real', type=int, default=5000,
                       help='Number of real images to use (None for all)')
    parser.add_argument('--normalization', type=str, default='per_image',
                       choices=['per_image', 'global', 'none'],
                       help='Image normalization method')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='embedding_comparison',
                       help='Output directory for results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cpu, cuda, cuda:0, etc.')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EMBEDDING COMPARISON: SYNTHETIC vs REAL (FIXED VERSION)")
    print("="*70)
    print("\n🔧 This version uses metrics that correctly detect spatial separation!")
    
    # Load image config
    print(f"\nLoading configuration from: {args.image_config}")
    image_config = json.load(open(args.image_config))
    image_size = image_config["N_PIXELS"]
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Pixel size: {image_config['PIXEL_SIZE']} Å")
    
    # Load conformational models
    print("\nLoading conformational models...")
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(args.device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(args.device).float()
    
    print(f"  ✅ Loaded {len(models)} conformational states")
    print(f"  Model shape: {models.shape}")
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(
        args.pretrained_weights,
        args.embedding,
        args.embedding_dim,
        image_size,
        args.device
    )
    
    # Generate synthetic images
    synthetic_images, synthetic_params = generate_synthetic_images(
        image_config, models, args.n_synthetic, args.device
    )
    
    # Load real images
    real_images = load_real_images(args.real_images, args.n_real, normalize=False)
    
    # Check size compatibility
    if real_images.shape[1:] != synthetic_images.shape[1:]:
        print(f"\n⚠️  Image size mismatch!")
        print(f"   Synthetic: {synthetic_images.shape[1:]}")
        print(f"   Real: {real_images.shape[1:]}")
        print(f"   Resizing real images to {image_size}x{image_size}...")
        
        real_images = F.interpolate(
            real_images.unsqueeze(1),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    
    # Apply consistent normalization
    print(f"\n🔧 Applying {args.normalization} normalization to both datasets...")
    synthetic_images = normalize_images(synthetic_images, method=args.normalization)
    real_images = normalize_images(real_images, method=args.normalization)
    
    print(f"  Synthetic - Mean: {synthetic_images.mean():.4f}, Std: {synthetic_images.std():.4f}")
    print(f"  Real      - Mean: {real_images.mean():.4f}, Std: {real_images.std():.4f}")
    
    # Visualize sample images
    visualize_sample_images(
        synthetic_images.numpy(),
        real_images.numpy(),
        args.output_dir
    )
    
    # Embed both sets
    synthetic_emb = embed_images(encoder, synthetic_images, device=args.device)
    real_emb = embed_images(encoder, real_images, device=args.device)
    
    # ============================================================
    # NEW ANALYSIS PIPELINE
    # ============================================================
    
    # 1. Separation metrics (PRIMARY - these are reliable!)
    separation_metrics = compute_separation_metrics(synthetic_emb, real_emb)
    
    # 2. Enhanced visualizations
    viz_metrics = create_enhanced_visualizations(synthetic_emb, real_emb, args.output_dir)
    
    # 3. Legacy metrics (for comparison only)
    legacy_stats = compute_legacy_statistics(synthetic_emb, real_emb)
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    torch.save({
        'synthetic_embeddings': synthetic_emb,
        'real_embeddings': real_emb,
        'separation_metrics': separation_metrics,
        'viz_metrics': viz_metrics,
        'legacy_stats': legacy_stats,
        'config': vars(args)
    }, os.path.join(args.output_dir, 'embeddings.pt'))
    
    print(f"\n  ✅ Saved embeddings and metrics: embeddings.pt")
    
    # Generate comprehensive report
    report = generate_report(separation_metrics, viz_metrics, legacy_stats, args.output_dir)
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    acc = separation_metrics['classifier_accuracy']
    sep_2d = separation_metrics['pca2d_separation']
    spatial_overlap = separation_metrics['spatial_overlap']
    visual_overlap = separation_metrics['visual_overlap_2d']
    
    print(f"\n🎯 Key Metrics:")
    print(f"   Classifier Accuracy:  {acc*100:.1f}% (baseline 50%)")
    print(f"   PCA 2D Separation:    {sep_2d:.2f}σ")
    print(f"   Spatial Overlap:      {spatial_overlap*100:.1f}%")
    print(f"   Visual Overlap:       {visual_overlap*100:.1f}%")
    
    print(f"\n📊 Overall Assessment: ", end="")
    
    if acc < 0.55 and sep_2d < 0.5:
        print("✅ EXCELLENT")
        print("\n   Synthetic and real distributions are virtually indistinguishable!")
        print("   The simulator accurately captures real data characteristics.")
        print("   ✓ Safe to use for inference")
    elif acc < 0.65 and sep_2d < 1.0:
        print("🟡 GOOD")
        print("\n   Minor differences exist but distributions substantially overlap.")
        print("   The simulator is reasonable with slight mismatch.")
        print("   ⚠ Proceed with caution - validate results")
    elif acc < 0.80 or sep_2d < 2.0:
        print("⚠️  POOR")
        print("\n   Clear separation detected between synthetic and real.")
        print("   ⚠ Significant simulator issues - results may be unreliable")
    else:
        print("❌ FAILED")
        print("\n   Strong separation - distributions occupy different regions.")
        print("   ❌ Do not use for inference - fix simulator first")
    
    if visual_overlap < 0.3 and acc > 0.7:
        print(f"\n🔍 Diagnosis:")
        print(f"   Low visual overlap ({visual_overlap*100:.1f}%) + high classifier accuracy ({acc*100:.1f}%)")
        print(f"   = Distributions occupy different regions (what you see in plots!)")
    
    print("\n📁 Output files:")
    print(f"   {args.output_dir}/")
    print(f"   ├── REPORT.md                    ← Read this first!")
    print(f"   ├── embeddings.pt                ← Saved embeddings & metrics")
    print(f"   ├── pca_enhanced.png             ← PCA with separation")
    print(f"   ├── tsne_projection.png          ← t-SNE visualization")
    if HAS_UMAP:
        print(f"   ├── umap_projection.png          ← UMAP visualization")
    print(f"   ├── distance_distributions.png   ← Distance histograms")
    print(f"   ├── dimension_statistics.png     ← Per-dimension analysis")
    print(f"   ├── pca_scree.png                ← Variance explained")
    print(f"   └── sample_images.png            ← Image samples")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📖 See {args.output_dir}/REPORT.md for detailed interpretation\n")


if __name__ == "__main__":
    import sys
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
