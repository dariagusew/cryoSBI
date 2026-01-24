"""
analyze_embedding_space.py

A script to visualize the latent space of a pre-trained image encoder.

It works by:
1. Loading a trained encoder model.
2. Simulating a new set of images with known conformation labels and SNR.
3. Passing these images through the encoder to get their embeddings.
4. Using UMAP to reduce the embedding dimensions to 2D.
5. Creating a scatter plot of the 2D embeddings, where:
   - Color represents the conformation label.
   - Size and transparency (alpha) represent the Signal-to-Noise Ratio (SNR).

This allows for a rich visual assessment of the latent space, helping to
verify if cluster overlaps correspond to low-SNR (ambiguous) images.

Usage:
    python analyze_embedding_space.py \
        --encoder_weights pretrained_image_embed.pt \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --num_images 10000 \
        --output_plot umap_visualization_with_snr.png
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import umap
except ImportError:
    print("❌ UMAP not found. Please install it:")
    print("   pip install umap-learn")
    exit(1)

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param


def generate_embeddings(encoder, image_config_path, num_images, batch_size, device):
    """
    Generates a specified number of new images and computes their embeddings and SNRs.

    Args:
        encoder (torch.nn.Module): The trained encoder model.
        image_config_path (str): Path to the image config JSON.
        num_images (int): Total number of images to generate.
        batch_size (int): Batch size for simulation and inference.
        device (str): The device to run on ('cuda' or 'cpu').

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - A NumPy array of embeddings (num_images, embedding_dim).
            - A NumPy array of corresponding labels (num_images,).
            - A NumPy array of corresponding SNRs (num_images,).
    """
    print("--- Step 1: Setting up simulation environment ---")
    
    image_config = json.load(open(image_config_path))
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(device).float()
    
    simulation_param = create_simulation_param(image_config, models, device=device)

    image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    data_loader = PriorLoader(image_prior, batch_size=batch_size, num_workers=2)
    data_iter = iter(data_loader)
    
    all_embeddings = []
    all_labels = []
    all_snrs = [] # <-- NEW: Store SNR values

    encoder.eval()

    print(f"--- Step 2: Generating {num_images} images and computing embeddings ---")
    with torch.no_grad(), tqdm(total=num_images, desc="Generating Embeddings") as pbar:
        while len(all_labels) < num_images:
            try:
                parameters = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                parameters = next(data_iter)

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
                simulation_param
            )

            embeddings = encoder(images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(indices.round().long().squeeze().numpy())
            all_snrs.extend(snr.squeeze().numpy()) # <-- NEW: Save SNR values
            
            pbar.update(images.size(0))

    final_embeddings = np.vstack(all_embeddings)[:num_images]
    final_labels = np.array(all_labels)[:num_images]
    final_snrs = np.array(all_snrs)[:num_images] # <-- NEW: Final SNR array
    
    return final_embeddings, final_labels, final_snrs


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize the latent space of a trained image encoder.'
    )

    parser.add_argument('--encoder_weights', type=str, required=True,
                       help='Path to the saved .pt file for the trained encoder.')
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to the image config JSON used during training.')
    parser.add_argument('--embedding', type=str, required=True,
                       choices=['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER', 'SPATIAL_CRYO_GAUSS_FFT_FILTER', 'RESNET18', 'RESNET18_FFT_FILTER'],
                       help='The embedding network architecture that was trained.')
    parser.add_argument('--embedding_dim', type=int, required=True,
                       help='The output dimension of the trained embedding network.')
    parser.add_argument('--num_images', type=int, default=10000,
                       help='Number of images to generate for the visualization (default: 10000).')
    parser.add_argument('--output_plot', type=str, default='umap_visualization_with_snr.png',
                       help='File path to save the output plot (default: umap_visualization_with_snr.png).')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for generating images and running inference (default: 512).')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: "cpu", "cuda", "cuda:0", etc. (default: "cuda").')
    
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️ CUDA not available! Falling back to CPU.")
        args.device = 'cpu'
    print(f"✅ Using device: {args.device}")

    print(f"\n--- Loading encoder: {args.embedding} with dim={args.embedding_dim} ---")
    image_config = json.load(open(args.image_config))
    image_size = image_config["N_PIXELS"]
    encoder = EMBEDDING_NETS[args.embedding](args.embedding_dim, D=image_size)
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location=args.device))
    encoder.to(args.device)
    print("✅ Encoder loaded successfully.")

    # --- Step 2: Generate data, compute embeddings, AND get SNRs ---
    embeddings, labels, snrs = generate_embeddings(
        encoder=encoder,
        image_config_path=args.image_config,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=args.device
    )
    print(f"✅ Generated {len(labels)} embeddings, labels, and SNRs.")

    print("\n--- Step 3: Running UMAP to reduce embeddings to 2D ---")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    print("✅ UMAP complete.")

    # --- Step 4: Create and save the plot ---
    print(f"\n--- Step 4: Creating visualization and saving to {args.output_plot} ---")
    num_classes = len(np.unique(labels))
    
    # --- NEW: Normalize SNR for better visual mapping ---
    # We use a percentile clip to avoid outliers dominating the scaling
    snr_min, snr_max = np.percentile(snrs, [5, 95])
    snrs_clipped = np.clip(snrs, snr_min, snr_max)
    snrs_normalized = (snrs_clipped - snr_min) / (snr_max - snr_min)

    # Map normalized SNR to size and alpha
    # Low SNR -> small, transparent. High SNR -> large, opaque.
    min_size = 5
    max_size = 50
    min_alpha = 0.2
    max_alpha = 0.9

    point_sizes = min_size + (snrs_normalized * (max_size - min_size))
    point_alphas = min_alpha + (snrs_normalized * (max_alpha - min_alpha))

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    cmap = plt.cm.get_cmap('turbo', num_classes)

    # Plot points one by one to control alpha individually
    # This is more flexible than passing an array of alphas to scatter
    for i in range(len(embedding_2d)):
        ax.scatter(
            embedding_2d[i, 0],
            embedding_2d[i, 1],
            c=[labels[i]],  # Pass color as a list for cmap
            cmap=cmap,
            vmin=0,
            vmax=num_classes-1,
            s=point_sizes[i],
            alpha=point_alphas[i]
        )

    ax.set_title(f'UMAP Visualization of Image Embeddings (Color: Label, Size/Alpha: SNR)', fontsize=20)
    ax.set_xlabel('UMAP Component 1', fontsize=14)
    ax.set_ylabel('UMAP Component 2', fontsize=14)
    
    # --- Create a custom legend ---
    # Color legend for labels
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=cmap(i), markersize=10,
                                 label=f'Conf. {i}') for i in np.unique(labels)]
    
    # Size/Alpha legend for SNR
    snr_legend_handles = [
        plt.scatter([], [], s=min_size, alpha=min_alpha, c='gray', label=f'Low SNR (~{snr_min:.2f})'),
        plt.scatter([], [], s=(min_size+max_size)/2, alpha=(min_alpha+max_alpha)/2, c='gray', label='Mid SNR'),
        plt.scatter([], [], s=max_size, alpha=max_alpha, c='gray', label=f'High SNR (~{snr_max:.2f})')
    ]
    
    legend1 = ax.legend(handles=legend_handles, title="Conformation", loc='upper right')
    ax.add_artist(legend1)
    ax.legend(handles=snr_legend_handles, title="SNR", loc='lower right')
    
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully to {args.output_plot}")


if __name__ == '__main__':
    main()
