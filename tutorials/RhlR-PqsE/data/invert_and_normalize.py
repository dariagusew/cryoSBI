#!/usr/bin/env python3
import argparse
import mrcfile
import torch
from tqdm import tqdm


def process_mrc(
    input_path: str,
    output_path: str,
    batch_size: int = 512,
    device: str = "cuda",
    eps: float = 1e-8,
):
    """Memory-efficient, GPU-accelerated sign inversion and per-image normalization."""
    # Ensure CUDA is available if requested
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Open source file via mmap to check shape and total images
    with mrcfile.mmap(input_path, mode="r", permissive=True) as src_mrc:
        total_images, height, width = src_mrc.data.shape

        print(
            f"Processing {total_images} images ({height}x{width}) on device [{device.upper()}]..."
        )

        # Allocate memory-mapped output file on disk
        out_mrc = mrcfile.new_mmap(
            output_path,
            shape=(total_images, height, width),
            mrc_mode=2,  # Mode 2 = float32
            overwrite=True,
        )

        # Batch processing loop
        for i in tqdm(
            range(0, total_images, batch_size), desc="Inverting & Normalizing"
        ):
            end_idx = min(i + batch_size, total_images)

            # Read slice from disk via memory map
            np_batch = src_mrc.data[i:end_idx]

            # Move batch to PyTorch GPU tensor
            batch = torch.from_numpy(np_batch.copy()).to(
                device, dtype=torch.float32
            )

            # 1. Flip data sign
            batch = -batch

            # 2. Normalize each image individually (mean=0, std=1)
            means = batch.mean(dim=(-2, -1), keepdim=True)
            stds = batch.std(dim=(-2, -1), keepdim=True)
            batch = (batch - means) / (stds + eps)

            # Stream result directly to disk via mmap
            out_mrc.data[i:end_idx] = batch.cpu().numpy()

        out_mrc.flush()
        out_mrc.close()

    print(f"Successfully written to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip sign and normalize (mean 0, std 1) an MRC image stack on GPU in batches."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input MRC file"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output MRC file"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for GPU processing",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_mrc(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
