# process_mrc_stack.py
"""
Process MRC particle stacks: fix headers and downsample using GPU.
Optimized for very large files (100+ GB).
"""

import argparse
import numpy as np
import torch
import mrcfile
from pathlib import Path
from tqdm import tqdm
import sys
import gc
import psutil
from contextlib import contextmanager
from typing import Tuple, Optional, Dict, Union

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def get_available_memory():
    """Get available system RAM and GPU memory in GB."""
    ram_available = psutil.virtual_memory().available / (1024**3)
    
    gpu_available = 0
    if torch.cuda.is_available():
        gpu_available = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated(0)) / (1024**3)
    
    return ram_available, gpu_available


def estimate_memory_requirements(nz, ny, nx, batch_size, target_size, stride=1):
    """
    Estimate memory requirements for processing.
    
    Returns:
        dict with memory estimates in GB
    """
    n_particles = (nz + stride - 1) // stride
    
    # Input batch in RAM (worst case, float32)
    input_batch_ram = batch_size * ny * nx * 4 / (1024**3)
    
    # GPU memory: input batch + output batch + overhead
    input_batch_gpu = batch_size * ny * nx * 4 / (1024**3)
    output_batch_gpu = batch_size * target_size * target_size * 4 / (1024**3)
    gpu_overhead = 1.0  # GB for PyTorch overhead
    total_gpu = input_batch_gpu + output_batch_gpu + gpu_overhead
    
    # Output array in RAM (full)
    output_array_ram = n_particles * target_size * target_size * 4 / (1024**3)
    
    # Peak RAM: output array + one batch
    peak_ram = output_array_ram + input_batch_ram + 2.0  # +2GB safety margin
    
    return {
        'input_batch_ram': input_batch_ram,
        'output_array_ram': output_array_ram,
        'peak_ram': peak_ram,
        'gpu_required': total_gpu,
        'n_output_particles': n_particles
    }


def check_memory_feasibility(mem_est, device='cuda'):
    """Check if we have enough memory to proceed."""
    ram_avail, gpu_avail = get_available_memory()
    
    issues = []
    warnings = []
    
    # Check RAM
    if mem_est['peak_ram'] > ram_avail:
        issues.append(f"Insufficient RAM: need {mem_est['peak_ram']:.1f} GB, "
                     f"have {ram_avail:.1f} GB")
    elif mem_est['peak_ram'] > ram_avail * 0.8:
        warnings.append(f"RAM usage will be high: {mem_est['peak_ram']:.1f} GB / "
                       f"{ram_avail:.1f} GB available")
    
    # Check GPU
    if device == 'cuda':
        if mem_est['gpu_required'] > gpu_avail:
            issues.append(f"Insufficient GPU memory: need {mem_est['gpu_required']:.1f} GB, "
                         f"have {gpu_avail:.1f} GB")
        elif mem_est['gpu_required'] > gpu_avail * 0.8:
            warnings.append(f"GPU usage will be high: {mem_est['gpu_required']:.1f} GB / "
                           f"{gpu_avail:.1f} GB available")
    
    return issues, warnings


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
        # For memmap, only check first particle to avoid loading all
        if isinstance(data, np.memmap):
            test_data = data[0] if data.ndim == 3 else data
        else:
            test_data = data
            
        if np.all(test_data == 0):
            return False, "All data is zero"
        if np.any(np.isnan(test_data)):
            return False, "Data contains NaN"
        if np.any(np.isinf(test_data)):
            return False, "Data contains inf"
        if np.std(test_data) == 0:
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
    if nz > 100000000:  # Increased limit for large stacks
        return False, f"Stack too large: {nz}"
    return True, "Valid"


@contextmanager
def open_mrc_memmap(filepath, max_size_gb=None):
    """
    Context manager for opening MRC as memmap (never loads into RAM).
    
    Yields:
        numpy.memmap or None
    """
    filepath = Path(filepath)
    memmap_obj = None
    
    try:
        if not filepath.exists():
            yield None, False, "File not found"
            return
        
        file_size, file_size_gb = check_mrc_file_size(filepath)
        if max_size_gb is not None and file_size_gb > max_size_gb:
            yield None, False, f"Too large: {file_size_gb:.2f} GB"
            return
        
        # Try standard mrcfile first (but don't load data)
        try:
            with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
                nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
                dtype = mrc.data.dtype
                
                # Close the mrcfile and open as pure memmap
                pass
            
            # Now open as memmap
            memmap_obj = np.memmap(
                filepath, 
                dtype=dtype, 
                mode='r', 
                offset=1024, 
                shape=(nz, ny, nx)
            )
            
            is_valid, msg = validate_mrc_data(memmap_obj)
            if is_valid:
                yield memmap_obj, True, "Memmap"
                return
                
        except Exception as e:
            pass
        
        # Fallback: force-read header
        header_info = read_mrc_header_raw(filepath)
        if header_info is not None:
            nx = header_info['nx']
            ny = header_info['ny']
            nz = header_info['nz']
            mode = header_info['mode']
            
            is_valid, msg = validate_mrc_dimensions(nx, ny, nz)
            if not is_valid:
                yield None, False, msg
                return
            
            dtype = get_dtype_from_mode(mode)
            memmap_obj = np.memmap(
                filepath, 
                dtype=dtype, 
                mode='r', 
                offset=1024, 
                shape=(nz, ny, nx)
            )
            
            is_valid, msg = validate_mrc_data(memmap_obj)
            if is_valid:
                yield memmap_obj, True, f"Force-read memmap"
                return
        
        yield None, False, "All methods failed"
        
    finally:
        # Cleanup
        if memmap_obj is not None:
            del memmap_obj
        gc.collect()


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


def normalize_batch_gpu(images, method='per_particle', global_stats=None):
    """
    Normalize batch of images on GPU.
    
    Args:
        images: torch.Tensor of shape (N, H, W)
        method: 'per_particle', 'global', or 'none'
        global_stats: dict with 'mean' and 'std' for global normalization
    
    Returns:
        Normalized torch.Tensor
    """
    if method == 'per_particle':
        mean = images.mean(dim=(1, 2), keepdim=True)
        std = images.std(dim=(1, 2), keepdim=True).clamp(min=1e-10)
        return (images - mean) / std
    
    elif method == 'global':
        if global_stats is None:
            raise ValueError("global_stats required for global normalization")
        mean = global_stats['mean']
        std = global_stats['std']
        return (images - mean) / std
    
    else:  # 'none'
        return images


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_global_stats_chunked(data, chunk_size=1000, stride=1):
    """
    Compute global mean and std in chunks without loading all data.
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        data: numpy memmap array (nz, ny, nx)
        chunk_size: number of particles per chunk
        stride: sample every Nth particle
    
    Returns:
        dict with 'mean' and 'std'
    """
    nz = data.shape[0]
    particle_indices = np.arange(0, nz, stride)
    n_particles = len(particle_indices)
    
    # Welford's algorithm for stable mean/variance computation
    count = 0
    mean = 0.0
    M2 = 0.0  # Sum of squared differences from mean
    
    print(f"  Computing global statistics from {n_particles} particles...")
    
    with tqdm(total=n_particles, desc="  Stats", unit="particles") as pbar:
        for i in range(0, n_particles, chunk_size):
            end_idx = min(i + chunk_size, n_particles)
            chunk_indices = particle_indices[i:end_idx]
            
            # Load chunk
            chunk = data[chunk_indices].astype(np.float64)  # Use float64 for precision
            
            # Update statistics using Welford's algorithm
            for particle in chunk:
                for value in particle.flat:
                    count += 1
                    delta = value - mean
                    mean += delta / count
                    delta2 = value - mean
                    M2 += delta * delta2
            
            pbar.update(end_idx - i)
            
            # Clean up chunk
            del chunk
    
    variance = M2 / count if count > 1 else 0.0
    std = np.sqrt(variance)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'count': count
    }


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
    max_size_gb=None,
    stride=1,
    validate_only=False
):
    """
    Process MRC particle stack: fix header and downsample.
    Optimized for very large files (100+ GB).
    
    Args:
        input_path: str or Path, input MRC file
        output_path: str or Path, output MRC file
        target_size: int, output dimension (npixel x npixel)
        batch_size: int, number of particles to process at once
        normalize: str, 'per_particle', 'global', or 'none'
        voxel_size: float or None, pixel size in Angstroms
        device: str, 'cuda' or 'cpu'
        max_size_gb: float or None, max file size to process
        stride: int, process every Nth particle
        validate_only: bool, only validate file without processing
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
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Total GPU memory: {gpu_mem:.1f} GB")
    
    # Show system memory
    ram_avail, gpu_avail = get_available_memory()
    print(f"  Available RAM: {ram_avail:.1f} GB")
    if device == 'cuda':
        print(f"  Available GPU memory: {gpu_avail:.1f} GB")
    
    # Read input MRC with memmap
    print(f"\n📖 Reading input: {input_path.name}")
    file_size, file_size_gb = check_mrc_file_size(input_path)
    print(f"  File size: {file_size_gb:.2f} GB")
    
    if file_size_gb > 10:
        print(f"  ⚠️  Large file detected - using memory-mapped I/O")
    
    # Open MRC as memmap (never loads into RAM)
    with open_mrc_memmap(input_path, max_size_gb=max_size_gb) as (data, success, msg):
        if not success:
            print(f"❌ Failed to read MRC: {msg}")
            return False
        
        print(f"✓ Loaded successfully: {msg}")
        print(f"  Shape: {data.shape} (nz={data.shape[0]}, ny={data.shape[1]}, nx={data.shape[2]})")
        print(f"  Dtype: {data.dtype}")
        
        # Sample a few particles to show range
        sample_indices = np.linspace(0, data.shape[0]-1, min(10, data.shape[0]), dtype=int)
        sample_data = data[sample_indices]
        print(f"  Sample range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        del sample_data
        
        nz, ny, nx = data.shape
        
        # Validate only mode
        if validate_only:
            print(f"\n✅ VALIDATION COMPLETE - File is readable")
            return True
        
        # Select particle indices based on stride
        particle_indices = np.arange(0, nz, stride)
        n_output_particles = len(particle_indices)
        
        print(f"\n⚙️  Processing configuration:")
        print(f"  Stride: {stride} (reading every {stride} particle(s))")
        print(f"  Output particles: {n_output_particles} / {nz}")
        print(f"  Batch size: {batch_size}")
        print(f"  Target size: {target_size}x{target_size}")
        
        # Estimate memory requirements
        print(f"\n💾 Memory estimation:")
        mem_est = estimate_memory_requirements(nz, ny, nx, batch_size, target_size, stride)
        print(f"  Input batch RAM: {mem_est['input_batch_ram']:.2f} GB")
        print(f"  Output array RAM: {mem_est['output_array_ram']:.2f} GB")
        print(f"  Peak RAM needed: {mem_est['peak_ram']:.2f} GB")
        if device == 'cuda':
            print(f"  GPU memory needed: {mem_est['gpu_required']:.2f} GB")
        
        # Check memory feasibility
        issues, warnings = check_memory_feasibility(mem_est, device)
        
        if issues:
            print(f"\n❌ Memory issues detected:")
            for issue in issues:
                print(f"  • {issue}")
            print(f"\nSuggestions:")
            print(f"  • Reduce batch size (current: {batch_size})")
            print(f"  • Increase stride (current: {stride})")
            print(f"  • Use smaller target size (current: {target_size})")
            return False
        
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
            print(f"  Proceeding anyway...")
        
        # Determine voxel size
        if voxel_size is None:
            try:
                with mrcfile.open(input_path, permissive=True, mode='r') as mrc:
                    voxel_size = float(mrc.voxel_size.x)
                    print(f"\n  Voxel size from header: {voxel_size:.3f} Å")
            except:
                voxel_size = 1.0
                print(f"\n  ⚠️  Could not read voxel size, using default: {voxel_size:.3f} Å")
        else:
            print(f"\n  Using provided voxel size: {voxel_size:.3f} Å")
        
        # Calculate output voxel size
        downsample_factor = max(ny, nx) / target_size
        output_voxel_size = voxel_size * downsample_factor
        print(f"  Output voxel size: {output_voxel_size:.3f} Å (downsample factor: {downsample_factor:.2f}x)")
        
        # Compute global statistics if needed (in chunks, never loads all data)
        global_stats = None
        if normalize == 'global':
            print(f"\n📊 Computing global statistics (this may take a while)...")
            global_stats = compute_global_stats_chunked(data, chunk_size=1000, stride=stride)
            print(f"  Global mean: {global_stats['mean']:.6f}")
            print(f"  Global std: {global_stats['std']:.6f}")
            print(f"  Total values: {global_stats['count']:,}")
        
        # Explain normalization
        print(f"\n🔧 Normalization:")
        if normalize == 'per_particle':
            print(f"  Method: Per-particle Z-score (each particle: mean=0, std=1)")
        elif normalize == 'global':
            print(f"  Method: Global Z-score (all particles normalized by same mean/std)")
        else:
            print(f"  Method: None (original values preserved)")
        
        # Create output MRC file and process directly to it
        print(f"\n⚙️  Creating output file ({mem_est['output_array_ram']:.2f} GB)...")
        try:
            with mrcfile.new(output_path, overwrite=True) as mrc:
                # Create empty array in the file
                mrc.set_data(np.zeros((n_output_particles, target_size, target_size), dtype=np.float32))
                
                print(f"✓ File created successfully")
                
                # Process in batches and write directly to mrc.data
                print(f"\n🚀 Processing particles...")
                n_batches = (n_output_particles + batch_size - 1) // batch_size
                
                try:
                    with tqdm(total=n_output_particles, desc="Processing", unit="particles") as pbar:
                        for i in range(0, n_output_particles, batch_size):
                            end_idx = min(i + batch_size, n_output_particles)
                            
                            # Get batch using stride indices (loads only this batch into RAM)
                            batch_indices = particle_indices[i:end_idx]
                            batch = data[batch_indices].astype(np.float32)  # Load batch
                            
                            # Convert to torch tensor and move to device
                            batch_tensor = torch.from_numpy(batch).to(device)
                            
                            # Downsample if needed
                            batch_tensor = downsample_gpu(batch_tensor, target_size)
                            
                            # Normalize
                            batch_tensor = normalize_batch_gpu(batch_tensor, method=normalize, 
                                                              global_stats=global_stats)
                            
                            # Write directly to mrc.data (no intermediate array)
                            mrc.data[i:end_idx] = batch_tensor.cpu().numpy()
                            
                            # Clean up
                            del batch, batch_tensor
                            
                            pbar.update(end_idx - i)
                            
                            # Clear GPU cache periodically
                            if device == 'cuda' and (i // batch_size) % 10 == 0:
                                torch.cuda.empty_cache()
                    
                    print(f"✓ Processing complete")
                    print(f"  Output shape: {mrc.data.shape}")
                    print(f"  Output range: [{mrc.data.min():.3f}, {mrc.data.max():.3f}]")
                    
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Processing interrupted by user")
                    return False
                except Exception as e:
                    print(f"\n\n❌ Error during processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                # Set correct voxel size
                mrc.voxel_size = output_voxel_size
                
                # Update header from data (this fixes corrupted headers)
                mrc.update_header_from_data()
                mrc.update_header_stats()
                
                print(f"\n✓ File written successfully")
                
        except Exception as e:
            print(f"❌ Failed to write MRC: {str(e)}")
            return False
    
    # Data memmap is now closed and cleaned up (exited context manager)
    gc.collect()
    
    # Verify output
    print(f"\n🔍 Verifying output...")
    try:
        with mrcfile.open(output_path, mode='r') as mrc:
            print(f"✓ Output file is valid")
            print(f"  Shape: {mrc.data.shape}")
            print(f"  Voxel size: {mrc.voxel_size.x:.3f} Å")
            print(f"  Data range: [{mrc.data.min():.3f}, {mrc.data.max():.3f}]")
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
        description='Process MRC particle stack: fix header and downsample (optimized for large files)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Normalization options:
  per_particle: Each particle normalized independently (mean=0, std=1)
                Formula: (x - mean_particle) / std_particle
                Use when: Particles have different contrast/intensity
                
  global:       All particles normalized by global statistics
                Formula: (x - mean_all) / std_all
                Use when: Want to preserve relative intensities between particles
                Note: For large files, global stats are computed in chunks
                
  none:         No normalization (original values)
                Use when: Values are already normalized or you want raw data

Memory optimization for large files (100+ GB):
  • File is never fully loaded into RAM (uses memory-mapped I/O)
  • Only processes batch_size particles at a time
  • Use --stride to subsample particles and reduce output size
  • Reduce --batch-size if running out of GPU memory
  • Global normalization computes statistics in chunks

Examples:
  # Validate file without processing
  python process_mrc_stack.py input.mrc output.mrc 128 --validate
  
  # Basic usage with per-particle normalization
  python process_mrc_stack.py input.mrc output.mrc 128
  
  # Process every 2nd particle (reduces output by 50%)
  python process_mrc_stack.py input.mrc output.mrc 128 --stride 2
  
  # Large file with reduced batch size
  python process_mrc_stack.py huge_400gb.mrc output.mrc 128 --batch-size 16 --stride 2
  
  # No normalization (preserve original values)
  python process_mrc_stack.py input.mrc output.mrc 128 --normalize none
  
  # Global normalization (maintains relative intensities)
  python process_mrc_stack.py input.mrc output.mrc 128 --normalize global
  
  # Use CPU instead of GPU
  python process_mrc_stack.py input.mrc output.mrc 128 --device cpu
  
  # Custom voxel size
  python process_mrc_stack.py input.mrc output.mrc 64 --voxel-size 1.5
        """
    )
    
    parser.add_argument('input', type=str, help='Input MRC file')
    parser.add_argument('output', type=str, help='Output MRC file')
    parser.add_argument('size', type=int, help='Output size (pixels)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU processing (default: 32, reduce if GPU OOM)')
    parser.add_argument('--normalize', type=str, default='per_particle',
                       choices=['per_particle', 'global', 'none'],
                       help='Normalization method (default: per_particle)')
    parser.add_argument('--voxel-size', type=float, default=None,
                       help='Pixel size in Angstroms (default: read from header)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Read every Nth particle (default: 1, read all)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--max-size-gb', type=float, default=None,
                       help='Maximum file size to process in GB (default: no limit)')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate input file without processing')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.size <= 0 or args.size > 2048:
        print(f"❌ Error: Invalid output size: {args.size} (must be 1-2048)")
        sys.exit(1)
    
    if args.stride < 1:
        print(f"❌ Error: Invalid stride: {args.stride} (must be >= 1)")
        sys.exit(1)
    
    if args.batch_size < 1:
        print(f"❌ Error: Invalid batch size: {args.batch_size} (must be >= 1)")
        sys.exit(1)
    
    # Check if output will overwrite existing file
    if Path(args.output).exists() and not args.validate:
        response = input(f"⚠️  Output file exists: {args.output}\n   Overwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Aborted by user")
            sys.exit(1)
    
    # Process
    try:
        success = process_mrc_stack(
            input_path=args.input,
            output_path=args.output,
            target_size=args.size,
            batch_size=args.batch_size,
            normalize=args.normalize,
            voxel_size=args.voxel_size,
            device=args.device,
            max_size_gb=args.max_size_gb,
            stride=args.stride,
            validate_only=args.validate
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
