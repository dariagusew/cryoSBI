"""
analyze_embedding_space.py

A script to visualize the latent space of a pre-trained image encoder.

It works by:
1. Loading a trained encoder model.
2. Simulating a new set of images with known conformation labels.
3. Passing these images through the encoder to get their embeddings.
4. Using UMAP to reduce the embedding dimensions to 2D.
5. Creating a scatter plot of the 2D embeddings, colored by their
   ground-truth conformation label.

This provides a clear visual assessment of how well the encoder has learned
to separate different conformations.

Usage:
    python analyze_embedding_space.py \
        --encoder_weights pretrained_image_embed.pt \
        --image_config config.json \
        --embedding SPATIAL_CRYO_FFT_FILTER \
        --embedding_dim 16 \
        --num_images 10000 \
        --output_plot umap_visualization.png
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# UMAP is a powerful dimensionality reduction tool.
# Install with: pip install umap-learn
try:
    import umap
except ImportError:
    print("❌ UMAP not found. Please install it:")
    print("   pip install umap-learn")
    exit(1)


# --- Import necessary components from your cryo_sbi library ---
# (Ensure this script can access them, same as your training script)
from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param


def generate_embeddings(encoder, image_config_path, num_images, batch_size, device):
    """
    Generates a specified number of new images and computes their embeddings.

    Args:
        encoder (torch.nn.Module): The trained encoder model.
        image_config_path (str): Path to the image config JSON.
        num_images (int): Total number of images to generate.
        batch_size (int): Batch size for simulation and inference.
        device (str): The device to run on ('cuda' or 'cpu').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - A NumPy array of embeddings (num_images, embedding_dim).
            - A NumPy array of corresponding labels (num_images,).
    """
    print("--- Step 1: Setting up simulation environment ---")
    
    # Load image configuration and conformational models
    image_config = json.load(open(image_config_path))
    if image_config["MODEL_FILE"].endswith("npy"):
        models = torch.from_numpy(np.load(image_config["MODEL_FILE"])).to(device).float()
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(device).float()
    
    simulation_param = create_simulation_param(image_config, models, device=device)

    # Set up the data loader for generating simulation parameters
    image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    data_loader = PriorLoader(image_prior, batch_size=batch_size, num_workers=2)
    data_iter = iter(data_loader)
    
    # Store results here
    all_embeddings = []
    all_labels = []

    # Put encoder in evaluation mode
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
            
            # Simulate a batch of images
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

            # Get embeddings from the encoder
            embeddings = encoder(images)

            # Store the results
            all_embeddings.append(embeddings.cpu().numpy())
            # Squeeze labels to be 1D for coloring
            all_labels.extend(indices.round().long().squeeze().numpy())
            
            pbar.update(images.size(0))

    # Concatenate all batches into single NumPy arrays
    final_embeddings = np.vstack(all_embeddings)[:num_images]
    final_labels = np.array(all_labels)[:num_images]
    
    return final_embeddings, final_labels


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize the latent space of a trained image encoder.'
    )

    # --- Required Arguments ---
    parser.add_argument('--encoder_weights', type=str, required=True,
                       help='Path to the saved .pt file for the trained encoder.')
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to the image config JSON used during training.')

    # --- Model Architecture (must match the trained model) ---
    parser.add_argument('--embedding', type=str, required=True,
                       choices=['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER', 'SPATIAL_CRYO_GAUSS_FFT_FILTER', 'RESNET18', 'RESNET18_FFT_FILTER'],
                       help='The embedding network architecture that was trained.')
    parser.add_argument('--embedding_dim', type=int, required=True,
                       help='The output dimension of the trained embedding network.')

    # --- Analysis & Plotting Arguments ---
    parser.add_argument('--num_images', type=int, default=10000,
                       help='Number of images to generate for the visualization (default: 10000).')
    parser.add_argument('--output_plot', type=str, default='umap_visualization.png',
                       help='File path to save the output plot (default: umap_visualization.png).')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for generating images and running inference (default: 512).')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: "cpu", "cuda", "cuda:0", etc. (default: "cuda").')
    
    args = parser.parse_args()

    # --- Setup Device ---
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️ CUDA not available! Falling back to CPU.")
        args.device = 'cpu'
    print(f"✅ Using device: {args.device}")

    # --- Step 1: Load the trained encoder model ---
    print(f"\n--- Loading encoder: {args.embedding} with dim={args.embedding_dim} ---")
    image_config = json.load(open(args.image_config))
    image_size = image_config["N_PIXELS"]
    
    # Instantiate the correct encoder architecture
    encoder = EMBEDDING_NETS[args.embedding](args.embedding_dim, D=image_size)
    
    # Load the saved weights
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location=args.device))
    encoder.to(args.device)
    print("✅ Encoder loaded successfully.")

    # --- Step 2: Generate data and compute embeddings ---
    embeddings, labels = generate_embeddings(
        encoder=encoder,
        image_config_path=args.image_config,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=args.device
    )
    print(f"✅ Generated {len(labels)} embeddings of dimension {embeddings.shape[1]}.")

    # --- Step 3: Run UMAP for dimensionality reduction ---
    print("\n--- Step 3: Running UMAP to reduce embeddings to 2D ---")
    reducer = umap.UMAP(
        n_neighbors=30,      # Controls local vs. global structure. Higher values = more global.
        min_dist=0.1,        # Controls how tightly points are clustered.
        n_components=2,
        metric='euclidean',
        random_state=42      # For reproducibility
    )
    embedding_2d = reducer.fit_transform(embeddings)
    print("✅ UMAP complete.")

    # --- Step 4: Create and save the plot ---
    print(f"\n--- Step 4: Creating visualization and saving to {args.output_plot} ---")
    num_classes = len(np.unique(labels))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use a colormap that works well for discrete classes
    cmap = plt.cm.get_cmap('turbo', num_classes)

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap=cmap,
        s=5,  # Point size
        alpha=0.7 # Point transparency
    )

    ax.set_title(f'UMAP Visualization of Image Embeddings ({args.embedding}, dim={args.embedding_dim})', fontsize=18)
    ax.set_xlabel('UMAP Component 1', fontsize=12)
    ax.set_ylabel('UMAP Component 2', fontsize=12)

    # Create a colorbar with integer labels
    cbar = fig.colorbar(scatter, ticks=np.unique(labels))
    cbar.set_label('Conformation Label', fontsize=12)
    
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully to {args.output_plot}")
    
    # To display the plot directly if in an interactive environment
    # plt.show()


if __name__ == '__main__':
    main()
