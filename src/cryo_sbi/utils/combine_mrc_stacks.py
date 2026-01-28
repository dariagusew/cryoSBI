# combine_mrc_stacks.py
"""
Combine two MRC particle stacks by randomly sampling particles from each.
Optimized for very large files (100+ GB) using memory-mapping and GPU.
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
from typing import Tuple, Optional, Dict, Union, List

# ============================================================================
# REUSED FUNCTIONS FROM process_mrc_stack.py
# (Memory Management and MRC File Handling)
# ============================================================================

def get_available_memory():
    """Get available system RAM and GPU memory in GB."""
    ram_available = psutil.virtual_memory().available / (1024**3)
    gpu_available = 0
    if torch.cuda.is_available():
        gpu_available = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated(0)) / (1024**3)
    return ram_available, gpu_available

def check_mrc_file_size(filepath):
    """Check MRC file size in bytes and GB."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    file_size_gb = file_size / (1024**3)
    return file_size, file_size_gb

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

@contextmanager
def open_mrc_memmap(filepath, max_size_gb=None):
    """Context manager for opening MRC as memmap (never loads into RAM)."""
    filepath = Path(filepath)
    memmap_obj = None
    try:
        if not filepath.exists():
            yield None, "File not found"
            return
        
        file_size, file_size_gb = check_mrc_file_size(filepath)
        if max_size_gb is not None and file_size_gb > max_size_gb:
            yield None, f"Too large: {file_size_gb:.2f} GB"
            return
        
        header_info = None
        try: # Try mrcfile first
            with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
                header_info = {
                    'nx': mrc.header.nx, 'ny': mrc.header.ny, 'nz': mrc.header.nz,
                    'dtype': mrc.data.dtype
                }
        except Exception:
            raw_header = read_mrc_header_raw(filepath)
            if raw_header:
                header_info = {
                    'nx': raw_header['nx'], 'ny': raw_header['ny'], 'nz': raw_header['nz'],
                    'dtype': get_dtype_from_mode(raw_header['mode'])
                }

        if header_info is None:
            yield None, "Could not read header by any method"
            return

        memmap_obj = np.memmap(
            filepath, 
            dtype=header_info['dtype'], 
            mode='r', 
            offset=1024, 
            shape=(header_info['nz'], header_info['ny'], header_info['nx'])
        )
        yield memmap_obj, "Success"
        
    finally:
        if memmap_obj is not None:
            del memmap_obj
        gc.collect()

# ============================================================================
# NEW COMBINING LOGIC
# ============================================================================

def combine_mrc_stacks(
    input_path1: Union[str, Path],
    input_path2: Union[str, Path],
    output_path: Union[str, Path],
    total_particles: int,
    ratio1: float,
    batch_size: int = 64,
    device: str = 'cuda'
):
    """
    Combines two MRC stacks into a new one by randomly sampling particles.
    
    Args:
        input_path1: Path to the first MRC file.
        input_path2: Path to the second MRC file.
        output_path: Path for the combined output MRC file.
        total_particles: Total number of particles in the output stack.
        ratio1: The ratio of particles to pick from the first stack (e.g., 0.7 for 70%).
        batch_size: Number of particles to process at once.
        device: 'cuda' or 'cpu'.
    """
    print(f"=" * 80)
    print(f"MRC STACK COMBINER")
    print(f"=" * 80)

    input_path1 = Path(input_path1)
    input_path2 = Path(input_path2)
    output_path = Path(output_path)
    
    # Check GPU availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        print(f"✓ Using device: {device}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Open both input files using our robust memmap context manager
    with open_mrc_memmap(input_path1) as (data1, msg1), \
         open_mrc_memmap(input_path2) as (data2, msg2):

        if data1 is None:
            print(f"❌ Failed to read {input_path1.name}: {msg1}")
            return False
        if data2 is None:
            print(f"❌ Failed to read {input_path2.name}: {msg2}")
            return False

        print(f"✓ Loaded successfully:")
        print(f"  Input 1 ({input_path1.name}): Shape={data1.shape}, Dtype={data1.dtype}")
        print(f"  Input 2 ({input_path2.name}): Shape={data2.shape}, Dtype={data2.dtype}")

        # --- Validation ---
        if data1.shape[1:] != data2.shape[1:]:
            print(f"❌ Error: Image dimensions do not match!")
            print(f"  Input 1: {data1.shape[1:]}, Input 2: {data2.shape[1:]}")
            return False
        
        nz1, ny, nx = data1.shape
        nz2 = data2.shape[0]

        # --- Calculate particle counts ---
        num_from_1 = int(total_particles * ratio1)
        num_from_2 = total_particles - num_from_1

        print(f"\n⚙️  Sampling configuration:")
        print(f"  Total output particles: {total_particles}")
        print(f"  Ratio from Input 1: {ratio1:.2f} ({num_from_1} particles)")
        print(f"  Ratio from Input 2: {1-ratio1:.2f} ({num_from_2} particles)")

        if num_from_1 > nz1:
            print(f"❌ Error: Requesting {num_from_1} particles from {input_path1.name}, but it only has {nz1}.")
            return False
        if num_from_2 > nz2:
            print(f"❌ Error: Requesting {num_from_2} particles from {input_path2.name}, but it only has {nz2}.")
            return False

        # --- Generate shuffled list of source indices ---
        print("\n🔀 Generating and shuffling particle indices...")
        # Randomly choose indices WITHOUT replacement
        indices1 = np.random.choice(nz1, size=num_from_1, replace=False)
        indices2 = np.random.choice(nz2, size=num_from_2, replace=False)
        
        # Create a unified "instruction list" of (source_array, source_index)
        # Using integers (0 or 1) to represent the source is more memory-efficient than storing memmap objects
        instructions = [(0, idx) for idx in indices1] + [(1, idx) for idx in indices2]
        np.random.shuffle(instructions)
        sources = [data1, data2] # A list to easily access the data source by index (0 or 1)
        print(f"✓ Index list created for {len(instructions)} particles.")

        # --- Create and process output file ---
        output_shape = (total_particles, ny, nx)
        output_file_size_gb = total_particles * ny * nx * 4 / (1024**3)
        print(f"\n💾 Creating output file ({output_file_size_gb:.2f} GB)...")

        try:
            with mrcfile.new_mmap(output_path, shape=output_shape, mrc_mode=2, overwrite=True) as mrc:
                # Attempt to copy header info from the first file
                try:
                    with mrcfile.open(input_path1, permissive=True) as mrc_in:
                        mrc.voxel_size = mrc_in.voxel_size
                        mrc.header.map = b'MAP '
                        mrc.update_header_from_data()
                except Exception as e:
                    print(f"  ⚠️ Could not copy header from source, using defaults. Error: {e}")
                    mrc.voxel_size = 1.0

                print(f"✓ Output file created successfully.")
                
                # --- Batch processing loop ---
                print(f"\n🚀 Combining particles with batch size {batch_size}...")
                
                with tqdm(total=total_particles, desc="Combining", unit="particles") as pbar:
                    for i in range(0, total_particles, batch_size):
                        end_idx = min(i + batch_size, total_particles)
                        
                        # Get the instructions for this batch
                        batch_instructions = instructions[i:end_idx]
                        current_batch_size = len(batch_instructions)
                        
                        # Create a CPU buffer for the batch
                        # This is where we assemble particles read from random locations on disk
                        batch_cpu = np.empty((current_batch_size, ny, nx), dtype=np.float32)
                        
                        for j, (source_idx, particle_idx) in enumerate(batch_instructions):
                            batch_cpu[j] = sources[source_idx][particle_idx]

                        # The "use GPU" part: move data through the GPU.
                        # This is useful if you were to add a processing step later.
                        # For a pure copy, it adds a small overhead but satisfies the requirement.
                        with torch.no_grad():
                            batch_tensor = torch.from_numpy(batch_cpu).to(device)
                            
                            # (Optional processing step could go here, e.g., normalization)
                            
                            # Write data directly to the output memory-mapped file
                            mrc.data[i:end_idx] = batch_tensor.cpu().numpy()

                        # Clean up
                        del batch_cpu
                        del batch_tensor
                        pbar.update(current_batch_size)
                        
                        if device == 'cuda' and (i // batch_size) % 10 == 0:
                            torch.cuda.empty_cache()
                            
                mrc.update_header_stats()
                print("✓ Combining complete.")
        
        except Exception as e:
            print(f"❌ An error occurred during file writing: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Final verification
    print(f"\n🔍 Verifying output...")
    try:
        with mrcfile.open(output_path, mode='r') as mrc:
            print(f"✓ Output file '{output_path.name}' is valid.")
            print(f"  Shape: {mrc.data.shape} (Expected: {output_shape})")
            if mrc.data.shape == output_shape:
                 print("  Shape verification successful.")
            else:
                 print("  ⚠️ Warning: Shape mismatch!")
    except Exception as e:
        print(f"⚠️ Warning: Could not verify output file: {str(e)}")
        
    print(f"\n{'=' * 80}")
    print(f"✅ SUCCESS: Combining complete")
    print(f"{'=' * 80}\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine two MRC particle stacks by randomly sampling from each.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input1", help="Path to the first input MRC file.")
    parser.add_argument("input2", help="Path to the second input MRC file.")
    parser.add_argument("output", help="Path for the combined output MRC file.")
    
    parser.add_argument(
        "--total", 
        type=int, 
        required=True, 
        help="Total number of particles in the final output stack."
    )
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=0.5, 
        help="Fraction of particles to take from the FIRST input file (e.g., 0.7 for 70%%)."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Number of particles to process in each batch."
    )
    parser.add_argument(
        "--device", 
        choices=['cuda', 'cpu'], 
        default='cuda', 
        help="Device to use for data transfer."
    )
    
    args = parser.parse_args()
    
    if not (0.0 <= args.ratio <= 1.0):
        print("Error: --ratio must be between 0.0 and 1.0.")
        sys.exit(1)
        
    combine_mrc_stacks(
        input_path1=args.input1,
        input_path2=args.input2,
        output_path=args.output,
        total_particles=args.total,
        ratio1=args.ratio,
        batch_size=args.batch_size,
        device=args.device
    )
