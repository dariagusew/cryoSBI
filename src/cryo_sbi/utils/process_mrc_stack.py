# process_mrc_stack.py
"""
Process MRC particle stacks: fix headers and downsample using GPU.
"""

import argparse
import numpy as np
import torch
import mrcfile
from pathlib import Path
from tqdm import tqdm
import sys

# ============================================================================
# MRC FILE HANDLING
# ============================================================================

def check_mrc_file_size(filepath):
    """Check MRC file size in bytes and GB."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    file_size_gb = file_size / (1024**3)
    return file_size, file_size_gb


def validate_mrc_data(data):
    """Validate MRC data after reading."""
    if data is None:
        return False, "Data is None"
    if data.size == 0:
        return False, "Data is empty"
    if data.ndim not in [2, 3]:
        return False, f"Invalid dimensions: {data.ndim}D"
    try:
        if np.all(data == 0):
            return False, "All data is zero"
        if np.any(np.isnan(data)):
            return False, "Data contains NaN"
        if np.any(np.isinf(data)):
            return False, "Data contains inf"
        if np.std(data) == 0:
            return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"


def read_mrc_header_raw(filepath):
    """Read MRC header manually."""
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(1024)
            if len(header_bytes) < 1024:
                return None
            import struct
            nx, ny, nz = struct.unpack('iii', header_bytes[0:12])
            mode = struct.unpack('i', header_bytes[12:16])[0]
            return {'nx': nx, 'ny': ny, 'nz': nz, 'mode': mode, 'header_size': 1024}
    except:
        return None


def get_dtype_from_mode(mode):
    """Convert MRC mode to numpy dtype."""
    dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    return dtype_map.get(mode, np.float32)


def validate_mrc_dimensions(nx, ny, nz):
    """Check if dimensions are reasonable."""
    if nx <= 0 or ny <= 0 or nz <= 0:
        return False, f"Non-positive: {nz}×{ny}×{nx}"
    if nx > 8192 or ny > 8192:
        return False, f"Too large: {ny}×{nx}"
    if nz > 50000000:
        return False, f"Stack too large: {nz}"
    return True, "Valid"


def open_mrc_robust(filepath, max_size_gb=None):
    """Robustly open MRC file with fallback methods."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None, False, "File not found"
    
    file_size, file_size_gb = check_mrc_file_size(filepath)
    if max_size_gb is not None and file_size_gb > max_size_gb:
        return None, False, f"Too large: {file_size_gb:.2f} GB"
    
    # Method 1: Standard
    try:
        with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
            if mrc.data is not None and mrc.data.size > 0:
                is_valid, msg = validate_mrc_data(mrc.data)
                if is_valid:
                    data = np.array(mrc.data) if file_size_gb < 1.0 else mrc.data
                    return data, True, "Standard"
    except:
        pass
    
    # Method 2: Force-read
    try:
        header_info = read_mrc_header_raw(filepath)
        if header_info is not None:
            nx, ny, nz, mode = header_info['nx'], header_info['ny'], header_info['nz'], header_info['mode']
            is_valid, msg = validate_mrc_dimensions(nx, ny, nz)
            if not is_valid:
                return None, False, msg
            
            dtype = get_dtype_from_mode(mode)
            data = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            
            is_valid, msg = validate_mrc_data(data[0])
            if is_valid:
                return data, True, f"Force-read memmap"
    except Exception as e:
        return None, False, f"Failed: {str(e)[:100]}"
    
    return None, False, "All methods failed"




# ============================================================================
# GPU PROCESSING FUNCTIONS
# ============================================================================

def downsample_gpu(images, target_size):
    """
    Downsample images to target_size x target_size using GPU.
    
    Args:
        images: torch.Tensor of shape (N, H, W) on GPU
        target_size: int, target dimension
    
    Returns:
        Downsampled images as torch.Tensor
    """
    N, H, W = images.shape
    
    if H == target_size and W == target_size:
        return images
    
    # Add channel dimension for interpolate
    images_4d = images.unsqueeze(1)  # (N, 1, H, W)
    
    # Use area interpolation for downsampling (best quality)
    downsampled = torch.nn.functional.interpolate(
        images_4d,
        size=(target_size, target_size),
        mode='area'
    ).squeeze(1)
    
    return downsampled


def normalize_batch_gpu(images, method='per_particle'):
    """
    Normalize batch of images on GPU.
    
    Args:
        images: torch.Tensor of shape (N, H, W)
        method: 'per_particle' - Z-score normalization per particle (mean=0, std=1)
                'global' - Normalize using global mean/std across all particles
                'none' - No normalization
    
    Returns:
        Normalized torch.Tensor
    """
    if method == 'per_particle':
        # Z-score normalization: (x - mean) / std for each particle individually
        # Each particle will have mean=0 and std=1
        mean = images.mean(dim=(1, 2), keepdim=True)
        std = images.std(dim=(1, 2), keepdim=True).clamp(min=1e-10)
        return (images - mean) / std
    
    elif method == 'global':
        # Global Z-score: use mean/std computed across ALL particles
        # All particles normalized by same values
        mean = images.mean()
        std = images.std().clamp(min=1e-10)
        return (images - mean) / std
    
    else:  # 'none'
        return images


# ============================================================================
# MRC HEADER REPAIR
# ============================================================================

def fix_mrc_header_from_data(data, voxel_size=1.0):
    """
    Create correct MRC header parameters from data.
    
    Args:
        data: numpy array (nz, ny, nx)
        voxel_size: float, pixel size in Angstroms
    
    Returns:
        dict with header parameters
    """
    nz, ny, nx = data.shape
    
    header_params = {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'mx': nx,
        'my': ny,
        'mz': nz,
        'cella': {
            'x': nx * voxel_size,
            'y': ny * voxel_size,
            'z': nz * voxel_size
        },
        'mapc': 1,
        'mapr': 2,
        'maps': 3,
    }
    
    return header_params


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_mrc_stack(
    input_path,
    output_path,
    target_size,
    batch_size=32,
    normalize='per_particle',
    voxel_size=None,
    device='cuda',
    max_size_gb=None
):
    """
    Process MRC particle stack: fix header and downsample.
    
    Args:
        input_path: str or Path, input MRC file
        output_path: str or Path, output MRC file
        target_size: int, output dimension (npixel x npixel)
        batch_size: int, number of particles to process at once
        normalize: str, 'per_particle', 'global', or 'none'
        voxel_size: float or None, pixel size in Angstroms
        device: str, 'cuda' or 'cpu'
        max_size_gb: float or None, max file size to process
    """
    
    print(f"=" * 80)
    print(f"MRC STACK PROCESSING")
    print(f"=" * 80)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Check GPU availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        print(f"✓ Using device: {device}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Read input MRC with robust method
    print(f"\n📖 Reading input: {input_path.name}")
    file_size, file_size_gb = check_mrc_file_size(input_path)
    print(f"  File size: {file_size_gb:.2f} GB")
    
    data, success, msg = open_mrc_robust(input_path, max_size_gb=max_size_gb)
    
    if not success:
        print(f"❌ Failed to read MRC: {msg}")
        return False
    
    print(f"✓ Loaded successfully: {msg}")
    print(f"  Shape: {data.shape} (nz={data.shape[0]}, ny={data.shape[1]}, nx={data.shape[2]})")
    print(f"  Dtype: {data.dtype}")
    print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")
    
    nz, ny, nx = data.shape
    
    # Determine voxel size
    if voxel_size is None:
        try:
            with mrcfile.open(input_path, permissive=True, mode='r') as mrc:
                voxel_size = float(mrc.voxel_size.x)
                print(f"  Voxel size from header: {voxel_size:.3f} Å")
        except:
            voxel_size = 1.0
            print(f"  ⚠️  Could not read voxel size, using default: {voxel_size:.3f} Å")
    else:
        print(f"  Using provided voxel size: {voxel_size:.3f} Å")
    
    # Calculate output voxel size
    downsample_factor = max(ny, nx) / target_size
    output_voxel_size = voxel_size * downsample_factor
    print(f"  Output voxel size: {output_voxel_size:.3f} Å (downsample factor: {downsample_factor:.2f}x)")
    
    # Prepare output array
    print(f"\n⚙️  Processing {nz} particles...")
    print(f"  Batch size: {batch_size}")
    print(f"  Target size: {target_size}x{target_size}")
    
    # Explain normalization
    if normalize == 'per_particle':
        print(f"  Normalization: Per-particle Z-score (each particle: mean=0, std=1)")
    elif normalize == 'global':
        print(f"  Normalization: Global Z-score (all particles normalized by same mean/std)")
    else:
        print(f"  Normalization: None (original values preserved)")
    
    processed_data = np.zeros((nz, target_size, target_size), dtype=np.float32)
    
    # For global normalization, we need to collect all data first or compute statistics
    if normalize == 'global':
        print("  Computing global statistics...")
        global_mean = float(data.mean())
        global_std = float(data.std())
        print(f"    Global mean: {global_mean:.3f}, Global std: {global_std:.3f}")
    
    # Process in batches
    n_batches = (nz + batch_size - 1) // batch_size
    
    with tqdm(total=nz, desc="Processing", unit="particles") as pbar:
        for i in range(0, nz, batch_size):
            end_idx = min(i + batch_size, nz)
            batch = data[i:end_idx]
            
            # Convert to torch tensor and move to GPU
            batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(device)
            
            # Downsample
            batch_tensor = downsample_gpu(batch_tensor, target_size)
            
            # Normalize
            if normalize == 'global':
                # Use pre-computed global statistics
                batch_tensor = (batch_tensor - global_mean) / max(global_std, 1e-10)
            else:
                batch_tensor = normalize_batch_gpu(batch_tensor, method=normalize)
            
            # Copy back to CPU
            processed_data[i:end_idx] = batch_tensor.cpu().numpy()
            
            pbar.update(end_idx - i)
            
            # Clear GPU cache periodically
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    print(f"✓ Processing complete")
    print(f"  Output shape: {processed_data.shape}")
    print(f"  Output range: [{processed_data.min():.3f}, {processed_data.max():.3f}]")
    print(f"  Output mean: {processed_data.mean():.3f}, Output std: {processed_data.std():.3f}")
    
    # Write output MRC with corrected header
    print(f"\n💾 Writing output: {output_path.name}")
    
    try:
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(processed_data.astype(np.float32))
            
            # Set correct voxel size
            mrc.voxel_size = output_voxel_size
            
            # Update header from data (this fixes corrupted headers)
            mrc.update_header_from_data()
            mrc.update_header_stats()
            
            print(f"✓ File written successfully")
            print(f"  Output size: {output_path.stat().st_size / 1e6:.2f} MB")
            
    except Exception as e:
        print(f"❌ Failed to write MRC: {str(e)}")
        return False
    
    # Verify output
    print(f"\n🔍 Verifying output...")
    try:
        with mrcfile.open(output_path, mode='r') as mrc:
            print(f"✓ Output file is valid")
            print(f"  Shape: {mrc.data.shape}")
            print(f"  Voxel size: {mrc.voxel_size.x:.3f} Å")
            print(f"  Data range: [{mrc.data.min():.3f}, {mrc.data.max():.3f}]")
            print(f"  Data mean: {mrc.data.mean():.3f}, std: {mrc.data.std():.3f}")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify output: {str(e)}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ SUCCESS: Processing complete")
    print(f"{'=' * 80}\n")
    
    return True


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process MRC particle stack: fix header and downsample',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Normalization options:
  per_particle: Each particle normalized independently (mean=0, std=1)
                Formula: (x - mean_particle) / std_particle
                Use when: Particles have different contrast/intensity
                
  global:       All particles normalized by global statistics
                Formula: (x - mean_all) / std_all
                Use when: Want to preserve relative intensities between particles
                
  none:         No normalization (original values)
                Use when: Values are already normalized or you want raw data

Examples:
  # Basic usage with per-particle normalization
  python process_mrc_stack.py input.mrc output.mrc 128
  
  # No normalization (preserve original values)
  python process_mrc_stack.py input.mrc output.mrc 128 --normalize none
  
  # Global normalization
  python process_mrc_stack.py input.mrc output.mrc 64 --normalize global
  
  # With custom parameters
  python process_mrc_stack.py input.mrc output.mrc 64 --batch-size 64 --voxel-size 1.5
  
  # Use CPU instead of GPU
  python process_mrc_stack.py input.mrc output.mrc 128 --device cpu
        """
    )
    
    parser.add_argument('input', type=str, help='Input MRC file')
    parser.add_argument('output', type=str, help='Output MRC file')
    parser.add_argument('size', type=int, help='Output size (pixels)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU processing (default: 32)')
    parser.add_argument('--normalize', type=str, default='per_particle',
                       choices=['per_particle', 'global', 'none'],
                       help='Normalization method (default: per_particle)')
    parser.add_argument('--voxel-size', type=float, default=None,
                       help='Pixel size in Angstroms (default: read from header)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--max-size-gb', type=float, default=None,
                       help='Maximum file size to process in GB (default: no limit)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.size <= 0 or args.size > 2048:
        print(f"❌ Error: Invalid output size: {args.size} (must be 1-2048)")
        sys.exit(1)
    
    # Process
    success = process_mrc_stack(
        input_path=args.input,
        output_path=args.output,
        target_size=args.size,
        batch_size=args.batch_size,
        normalize=args.normalize,
        voxel_size=args.voxel_size,
        device=args.device,
        max_size_gb=args.max_size_gb
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
