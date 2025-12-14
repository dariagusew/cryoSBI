"""
compare_embeddings.py

Compare embeddings of synthetic and real cryo-EM images.
Checks for overlap in latent space using multiple visualization and quantification methods.

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
            images = cryo_em_simulator(
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


def load_real_images(mrc_path, n_images=None, normalize=True):
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
        
        # Normalize
        if normalize:
            # Per-image normalization (common in cryo-EM)
            images = (images - images.mean(dim=(1,2), keepdim=True)) / (images.std(dim=(1,2), keepdim=True) + 1e-8)
        
        print(f"  ✅ Loaded {len(images)} images")
        print(f"  Shape: {images.shape}")
        print(f"  Range: [{images.min():.3f}, {images.max():.3f}]")
        
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
# ANALYSIS & VISUALIZATION
# ============================================================================

def compute_statistics(synthetic_emb, real_emb):
    """Compute statistical comparison of embeddings"""
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    stats = {}
    
    # Basic statistics
    stats['synthetic_mean'] = synthetic_emb.mean(dim=0)
    stats['synthetic_std'] = synthetic_emb.std(dim=0)
    stats['real_mean'] = real_emb.mean(dim=0)
    stats['real_std'] = real_emb.std(dim=0)
    
    # Overall statistics
    print(f"\nSynthetic embeddings:")
    print(f"  Mean: {stats['synthetic_mean'].mean():.4f}")
    print(f"  Std:  {stats['synthetic_std'].mean():.4f}")
    print(f"  Range: [{synthetic_emb.min():.4f}, {synthetic_emb.max():.4f}]")
    
    print(f"\nReal embeddings:")
    print(f"  Mean: {stats['real_mean'].mean():.4f}")
    print(f"  Std:  {stats['real_std'].mean():.4f}")
    print(f"  Range: [{real_emb.min():.4f}, {real_emb.max():.4f}]")
    
    # Distance statistics
    print(f"\nDistance statistics:")
    
    # Within-group distances
    syn_dists = torch.cdist(synthetic_emb[:1000], synthetic_emb[:1000])
    syn_dists = syn_dists[torch.triu(torch.ones_like(syn_dists), diagonal=1) == 1]
    stats['synthetic_dist_mean'] = syn_dists.mean().item()
    stats['synthetic_dist_std'] = syn_dists.std().item()
    
    real_dists = torch.cdist(real_emb[:1000], real_emb[:1000])
    real_dists = real_dists[torch.triu(torch.ones_like(real_dists), diagonal=1) == 1]
    stats['real_dist_mean'] = real_dists.mean().item()
    stats['real_dist_std'] = real_dists.std().item()
    
    print(f"  Within synthetic: {stats['synthetic_dist_mean']:.4f} ± {stats['synthetic_dist_std']:.4f}")
    print(f"  Within real:      {stats['real_dist_mean']:.4f} ± {stats['real_dist_std']:.4f}")
    
    # Cross-group distances
    cross_dists = torch.cdist(synthetic_emb[:1000], real_emb[:1000])
    stats['cross_dist_mean'] = cross_dists.mean().item()
    stats['cross_dist_std'] = cross_dists.std().item()
    
    print(f"  Synthetic-Real:   {stats['cross_dist_mean']:.4f} ± {stats['cross_dist_std']:.4f}")
    
    # Overlap metric
    # If cross-distance is similar to within-group distances → good overlap
    overlap_ratio = stats['cross_dist_mean'] / np.mean([stats['synthetic_dist_mean'], stats['real_dist_mean']])
    stats['overlap_ratio'] = overlap_ratio
    
    print(f"\nOverlap metric (cross/within ratio): {overlap_ratio:.4f}")
    if overlap_ratio < 1.2:
        print("  ✅ Excellent overlap - distributions are well-aligned")
    elif overlap_ratio < 1.5:
        print("  🟡 Good overlap - minor distribution shift")
    elif overlap_ratio < 2.0:
        print("  ⚠️  Moderate overlap - noticeable distribution shift")
    else:
        print("  ❌ Poor overlap - distributions are separated")
    
    return stats


def nearest_neighbor_analysis(synthetic_emb, real_emb, k=5):
    """Analyze nearest neighbors between synthetic and real"""
    print("\n" + "="*70)
    print("NEAREST NEIGHBOR ANALYSIS")
    print("="*70)
    
    n_samples = min(1000, len(synthetic_emb), len(real_emb))
    
    syn_sample = synthetic_emb[:n_samples].numpy()
    real_sample = real_emb[:n_samples].numpy()
    
    print(f"\nAnalyzing {n_samples} samples per group (k={k})...")
    
    # For each real image, find k nearest synthetic images
    distances = cdist(real_sample, syn_sample, metric='euclidean')
    nn_indices = np.argsort(distances, axis=1)[:, :k]
    nn_distances = np.take_along_axis(distances, nn_indices, axis=1)
    
    mean_nn_dist = nn_distances.mean()
    median_nn_dist = np.median(nn_distances)
    
    print(f"\nReal → Synthetic NN distances:")
    print(f"  Mean:   {mean_nn_dist:.4f}")
    print(f"  Median: {median_nn_dist:.4f}")
    print(f"  Std:    {nn_distances.std():.4f}")
    
    # For each synthetic image, find k nearest real images
    distances_rev = cdist(syn_sample, real_sample, metric='euclidean')
    nn_indices_rev = np.argsort(distances_rev, axis=1)[:, :k]
    nn_distances_rev = np.take_along_axis(distances_rev, nn_indices_rev, axis=1)
    
    mean_nn_dist_rev = nn_distances_rev.mean()
    median_nn_dist_rev = np.median(nn_distances_rev)
    
    print(f"\nSynthetic → Real NN distances:")
    print(f"  Mean:   {mean_nn_dist_rev:.4f}")
    print(f"  Median: {median_nn_dist_rev:.4f}")
    print(f"  Std:    {nn_distances_rev.std():.4f}")
    
    # Symmetry check
    symmetry_ratio = mean_nn_dist / mean_nn_dist_rev
    print(f"\nSymmetry ratio: {symmetry_ratio:.4f}")
    if 0.9 < symmetry_ratio < 1.1:
        print("  ✅ Symmetric - balanced coverage")
    else:
        print("  ⚠️  Asymmetric - one distribution may be subset of other")
    
    return {
        'real_to_syn_dist': mean_nn_dist,
        'syn_to_real_dist': mean_nn_dist_rev,
        'symmetry_ratio': symmetry_ratio
    }


def visualize_embeddings(synthetic_emb, real_emb, output_dir):
    """Create visualization plots"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine for dimensionality reduction
    all_emb = torch.cat([synthetic_emb, real_emb], dim=0).numpy()
    labels = np.array(['Synthetic']*len(synthetic_emb) + ['Real']*len(real_emb))
    
    # 1. PCA
    print("\n1. PCA projection...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(all_emb)
    
    plt.figure(figsize=(10, 8))

    plt.scatter(emb_pca[labels=='Synthetic', 0], emb_pca[labels=='Synthetic', 1], 
                alpha=0.5, s=1, label='Synthetic', c='blue')
    
    plt.scatter(emb_pca[labels=='Real', 0], emb_pca[labels=='Real', 1], 
                alpha=0.5, s=1, label='Real', c='red')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA: Synthetic vs Real Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_projection.png'), dpi=300)
    plt.close()
    print(f"  ✅ Saved: pca_projection.png")
    
    # 2. t-SNE
    print("\n2. t-SNE projection...")
    # Use subset for speed
    n_samples = min(10000, len(all_emb))
    indices = np.random.choice(len(all_emb), n_samples, replace=False)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(all_emb[indices])
    #emb_tsne = tsne.fit_transform(all_emb)  
  
    plt.figure(figsize=(10, 8))
    mask_syn = labels[indices] == 'Synthetic'
    plt.scatter(emb_tsne[mask_syn, 0], emb_tsne[mask_syn, 1],
                alpha=0.5, s=1, label='Synthetic', c='blue')
    plt.scatter(emb_tsne[~mask_syn, 0], emb_tsne[~mask_syn, 1],
                alpha=0.5, s=1, label='Real', c='red')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE: Synthetic vs Real Embeddings')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_projection.png'), dpi=300)
    plt.close()
    print(f"  ✅ Saved: tsne_projection.png")
    
    # 3. UMAP (if available)
    if HAS_UMAP:
        print("\n3. UMAP projection...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb_umap = reducer.fit_transform(all_emb[indices])
        #emb_umap = reducer.fit_transform(all_emb)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(emb_umap[mask_syn, 0], emb_umap[mask_syn, 1],
                    alpha=0.5, s=1, label='Synthetic', c='blue')
        plt.scatter(emb_umap[~mask_syn, 0], emb_umap[~mask_syn, 1],
                    alpha=0.5, s=1, label='Real', c='red')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP: Synthetic vs Real Embeddings')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_projection.png'), dpi=300)
        plt.close()
        print(f"  ✅ Saved: umap_projection.png")
    
    # 4. Distance distributions
    print("\n4. Distance distributions...")
    
    # Sample for speed
    n_samples = min(10000, len(synthetic_emb), len(real_emb))
    syn_sample = synthetic_emb[:n_samples]
    real_sample = real_emb[:n_samples]
    
    # Within-group distances
    syn_dists = torch.cdist(syn_sample, syn_sample)
    syn_dists = syn_dists[torch.triu(torch.ones_like(syn_dists), diagonal=1) == 1].numpy()
    
    real_dists = torch.cdist(real_sample, real_sample)
    real_dists = real_dists[torch.triu(torch.ones_like(real_dists), diagonal=1) == 1].numpy()
    
    # Cross-group distances
    cross_dists = torch.cdist(syn_sample, real_sample).flatten().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.hist(syn_dists, bins=50, alpha=0.5, label='Within Synthetic', density=True)
    plt.hist(real_dists, bins=50, alpha=0.5, label='Within Real', density=True)
    plt.hist(cross_dists, bins=50, alpha=0.5, label='Synthetic-Real', density=True)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Pairwise Distances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_distributions.png'), dpi=300)
    plt.close()
    print(f"  ✅ Saved: distance_distributions.png")
    
    # 5. Per-dimension statistics
    print("\n5. Per-dimension comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Means
    ax = axes[0, 0]
    syn_means = synthetic_emb.mean(dim=0).numpy()
    real_means = real_emb.mean(dim=0).numpy()
    ax.scatter(syn_means, real_means, alpha=0.5, s=1)
    lim = max(np.abs(syn_means).max(), np.abs(real_means).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)
    ax.set_xlabel('Synthetic Mean')
    ax.set_ylabel('Real Mean')
    ax.set_title('Per-Dimension Means')
    ax.grid(True, alpha=0.3)
    
    # Stds
    ax = axes[0, 1]
    syn_stds = synthetic_emb.std(dim=0).numpy()
    real_stds = real_emb.std(dim=0).numpy()
    ax.scatter(syn_stds, real_stds, alpha=0.5, s=1)
    lim = max(syn_stds.max(), real_stds.max())
    ax.plot([0, lim], [0, lim], 'r--', alpha=0.5)
    ax.set_xlabel('Synthetic Std')
    ax.set_ylabel('Real Std')
    ax.set_title('Per-Dimension Standard Deviations')
    ax.grid(True, alpha=0.3)
    
    # Mean histogram
    ax = axes[1, 0]
    ax.hist(syn_means, bins=30, alpha=0.5, label='Synthetic')
    ax.hist(real_means, bins=30, alpha=0.5, label='Real')
    ax.set_xlabel('Mean Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dimension Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std histogram
    ax = axes[1, 1]
    ax.hist(syn_stds, bins=30, alpha=0.5, label='Synthetic')
    ax.hist(real_stds, bins=30, alpha=0.5, label='Real')
    ax.set_xlabel('Std Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dimension Stds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_statistics.png'), dpi=300)
    plt.close()
    print(f"  ✅ Saved: dimension_statistics.png")


def visualize_sample_images(synthetic_imgs, real_imgs, output_dir, n_samples=10):
    """Show sample images from each group"""
    print("\n6. Sample images...")
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    
    # Synthetic
    for i in range(n_samples):
        axes[0, i].imshow(synthetic_imgs[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Synthetic', fontsize=12)
    
    # Real
    for i in range(n_samples):
        axes[1, i].imshow(real_imgs[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Real', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=300)
    plt.close()
    print(f"  ✅ Saved: sample_images.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare embeddings of synthetic and real cryo-EM images'
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
                       help='Number of real images to use (or None for all)')
    
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
    print("EMBEDDING COMPARISON: SYNTHETIC vs REAL")
    print("="*70)
    
    # Load image config
    image_config = json.load(open(args.image_config))
    image_size = image_config["N_PIXELS"]
    
    # Load conformational models
    print("\nLoading conformational models...")
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(args.device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(args.device).float()
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(
        args.pretrained_weights,
        args.embedding,
        args.embedding_dim,
        image_size,
        args.device
    )
    
    # Generate synthetic images
    synthetic_images, _ = generate_synthetic_images(
        image_config, models, args.n_synthetic, args.device
    )
    
    # Load real images
    real_images = load_real_images(args.real_images, args.n_real)
    
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
    
    # Visualize samples
    visualize_sample_images(
        synthetic_images.numpy(),
        real_images.numpy(),
        args.output_dir
    )
    
    # Embed both sets
    synthetic_emb = embed_images(encoder, synthetic_images, device=args.device)
    real_emb = embed_images(encoder, real_images, device=args.device)
    
    # Compute statistics
    stats = compute_statistics(synthetic_emb, real_emb)
    
    # Nearest neighbor analysis
    nn_stats = nearest_neighbor_analysis(synthetic_emb, real_emb)
    
    # Create visualizations
    visualize_embeddings(synthetic_emb, real_emb, args.output_dir)
    
    # Save embeddings
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    torch.save({
        'synthetic_embeddings': synthetic_emb,
        'real_embeddings': real_emb,
        'statistics': stats,
        'nn_statistics': nn_stats,
        'config': vars(args)
    }, os.path.join(args.output_dir, 'embeddings.pt'))
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    print(f"   - embeddings.pt")
    print(f"   - pca_projection.png")
    print(f"   - tsne_projection.png")
    if HAS_UMAP:
        print(f"   - umap_projection.png")
    print(f"   - distance_distributions.png")
    print(f"   - dimension_statistics.png")
    print(f"   - sample_images.png")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    overlap_ratio = stats['overlap_ratio']
    print(f"\n📊 Overlap metric: {overlap_ratio:.4f}")
    
    if overlap_ratio < 1.2:
        print("\n✅ EXCELLENT OVERLAP")
        print("   Synthetic and real distributions are well-aligned")
        print("   → Simulator captures real data characteristics well")
    elif overlap_ratio < 1.5:
        print("\n🟡 GOOD OVERLAP")
        print("   Minor distribution shift detected")
        print("   → Simulator is reasonable but has some mismatch")
    elif overlap_ratio < 2.0:
        print("\n⚠️  MODERATE OVERLAP")
        print("   Noticeable distribution shift")
        print("   → Check simulator parameters or preprocessing")
    else:
        print("\n❌ POOR OVERLAP")
        print("   Distributions are separated")
        print("   → Major mismatch between synthetic and real")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70 + "\n")


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
