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
from typing import Tuple, Optional, Dict, Union, List

from cryo_sbi.wpa_simulator.ctf import generate_ctf, remove_ctf_batch

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
    
    # The on-disk size of the final output file
    output_file_disk_size = n_particles * target_size * target_size * 4 / (1024**3)
    
    # Peak RAM: just one batch, since output is memory-mapped
    peak_ram = input_batch_ram + 2.0  # +2GB safety margin
    
    return {
        'input_batch_ram': input_batch_ram,
        'output_file_disk_size': output_file_disk_size,
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
    return file_size, file_size / (1024**3)

def validate_mrc_data(data):
    """
    Validate MRC data after reading. This version is 'memmap-aware' to avoid
    loading entire large files into memory for validation.
    """
    if data is None or data.size == 0 or data.ndim not in [2, 3]:
        return False, f"Invalid data shape or type: {data.shape if hasattr(data, 'shape') else 'None'}"
    try:
        # For memmap, only check the first particle to avoid loading all data.
        if isinstance(data, np.memmap):
            test_data = data[0] if data.ndim == 3 else data
        else:
            test_data = data
        if np.all(test_data == 0): return False, "All data is zero"
        if np.any(np.isnan(test_data)): return False, "Data contains NaN"
        if np.any(np.isinf(test_data)): return False, "Data contains inf"
        if np.std(test_data) == 0: return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def read_mrc_header_raw(filepath):
    """Read MRC header manually if standard methods fail."""
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(1024)
            if len(header_bytes) < 1024: return None
            import struct
            nx, ny, nz = struct.unpack('iii', header_bytes[0:12])
            mode = struct.unpack('i', header_bytes[12:16])[0]
            return {'nx': nx, 'ny': ny, 'nz': nz, 'mode': mode}
    except:
        return None

def get_dtype_from_mode(mode):
    """Convert MRC mode to numpy dtype."""
    dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    return dtype_map.get(mode, np.float32)

def validate_mrc_dimensions(nx, ny, nz):
    """Check if dimensions are reasonable."""
    if nx <= 0 or ny <= 0 or nz <= 0: return False, f"Non-positive: {nz}×{ny}×{nx}"
    if nx > 8192 or ny > 8192: return False, f"Too large: {ny}×{nx}"
    if nz > 50000000: return False, f"Stack too large: {nz}"
    return True, "Valid"

def open_mrc_robust(filepath, max_size_gb=None):
    """
    Robustly open MRC file with fallback methods, prioritizing memory-mapping
    to avoid loading large files into RAM.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None, False, "File not found"
    
    file_size, file_size_gb = check_mrc_file_size(filepath)
    if max_size_gb is not None and file_size_gb > max_size_gb:
        return None, False, f"Too large: {file_size_gb:.2f} GB"
    
    try:
        header_info = read_mrc_header_raw(filepath)
        if header_info is not None:
            nx, ny, nz, mode = header_info['nx'], header_info['ny'], header_info['nz'], header_info['mode']
            is_valid, msg = validate_mrc_dimensions(nx, ny, nz)
            if not is_valid:
                return None, False, msg
            
            dtype = get_dtype_from_mode(mode)
            data = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            
            is_valid, msg = validate_mrc_data(data)
            if is_valid:
                return data, True, f"Memmap via manual header read"
    except Exception as e:
        return None, False, f"Failed: {str(e)[:100]}"
    
    return None, False, "All MRC opening methods failed"


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
# STAR FILE PARSING
# ============================================================================

def parse_star_file_ctf(star_path: str) -> List[Dict]:
    """
    Parse a RELION STAR file and extract per-particle CTF parameters.

    Reads the following columns (all required unless noted):
        _rlnDefocusU          – defocus along U axis (Å)
        _rlnDefocusV          – defocus along V axis (Å)
        _rlnDefocusAngle      – astigmatism angle (degrees)
        _rlnVoltage           – accelerating voltage (kV)
        _rlnSphericalAberration – Cs (mm)
        _rlnAmplitudeContrast – amplitude contrast (fraction, e.g. 0.1)
        _rlnPhaseShift        – additional phase shift (degrees, optional, default 0)
        _rlnCtfBfactor        – B-factor envelope (Å², optional, default 0)
        _rlnCtfScalefactor    – overall scale factor (optional, default 1)

    Args:
        star_path: Path to the RELION STAR file.

    Returns:
        List of dicts, one per particle row, with keys:
            'defocus_u', 'defocus_v', 'defocus_angle',
            'voltage', 'cs', 'amplitude_contrast',
            'phase_shift', 'bfactor', 'scale_factor'
    """
    star_path = Path(star_path)
    if not star_path.exists():
        raise FileNotFoundError(f"STAR file not found: {star_path}")

    # Map from RELION column name → internal key and default value
    _COL_MAP = {
        '_rlnDefocusU':            ('defocus_u',          None),
        '_rlnDefocusV':            ('defocus_v',          None),
        '_rlnDefocusAngle':        ('defocus_angle',      None),
        '_rlnVoltage':             ('voltage',            None),
        '_rlnSphericalAberration': ('cs',                 None),
        '_rlnAmplitudeContrast':   ('amplitude_contrast', None),
        '_rlnPhaseShift':          ('phase_shift',        0.0),
        '_rlnCtfBfactor':          ('bfactor',            0.0),
        '_rlnCtfScalefactor':      ('scale_factor',       1.0),
    }
    _REQUIRED = {
        '_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle',
        '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast',
    }

    with open(star_path, 'r') as fh:
        lines = fh.readlines()

    # Locate the loop_ block and collect column headers
    col_index: Dict[str, int] = {}   # rln_name → column index (0-based)
    in_loop = False
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'loop_':
            in_loop = True
            col_index = {}
            continue
        if in_loop and stripped.startswith('_'):
            parts = stripped.split()
            col_name = parts[0]
            col_idx = len(col_index)
            col_index[col_name] = col_idx
            continue
        if in_loop and col_index and stripped and not stripped.startswith('_'):
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"No data rows found in STAR file: {star_path}")

    # Check required columns
    missing = _REQUIRED - set(col_index.keys())
    if missing:
        raise ValueError(
            f"STAR file is missing required CTF columns: {missing}\n"
            f"Found columns: {list(col_index.keys())}"
        )

    # Build index lookup for the columns we care about
    wanted: Dict[str, Tuple[str, float, int]] = {}  # rln_name → (key, default, col_idx)
    for rln_name, (key, default) in _COL_MAP.items():
        if rln_name in col_index:
            wanted[rln_name] = (key, default, col_index[rln_name])

    # Parse data rows
    particles = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('data_') or stripped.startswith('loop_'):
            continue
        tokens = stripped.split()
        entry: Dict[str, float] = {}
        for rln_name, (key, default, idx) in wanted.items():
            if idx < len(tokens):
                entry[key] = float(tokens[idx])
            elif default is not None:
                entry[key] = default
            else:
                raise ValueError(
                    f"Row has too few columns for required field '{rln_name}': {line!r}"
                )
        particles.append(entry)

    if not particles:
        raise ValueError(f"No particle rows parsed from STAR file: {star_path}")

    return particles


# ============================================================================
# CTF HELPERS (delegates to cryo_sbi.wpa_simulator.ctf)
# ============================================================================

def _build_ctf_from_params(
    ctf_params: List[Dict],
    image_size: int,
    pixel_size: float,
    device: str,
) -> torch.Tensor:
    """
    Build a batch CTF tensor from a list of per-particle parameter dicts
    (as returned by :func:`parse_star_file_ctf`) by delegating to
    :func:`cryo_sbi.wpa_simulator.ctf.generate_ctf`.

    Args:
        ctf_params: List of dicts with keys 'defocus_u', 'defocus_v',
                    'defocus_angle', 'voltage', 'cs',
                    'amplitude_contrast', 'bfactor'.
        image_size: Side length of the square image (pixels).
        pixel_size: Pixel size in Å.
        device:     Torch device string.

    Returns:
        CTF tensor of shape (N, image_size, image_size), DC at corner.
    """
    def _t(key):
        return torch.tensor(
            [p[key] for p in ctf_params], dtype=torch.float32, device=device
        )

    return generate_ctf(
        num_pixels=image_size,
        pixel_size=pixel_size,
        defocus_u=_t('defocus_u'),
        defocus_v=_t('defocus_v'),
        defocus_angle=_t('defocus_angle'),
        voltage=_t('voltage'),
        cs=_t('cs'),
        amplitude_contrast=_t('amplitude_contrast'),
        b_factor=_t('bfactor'),
        device=device,
    )


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
    validate_only=False,
    subtract_ctf=False,
    star_file=None,
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
        subtract_ctf: bool, if True subtract the CTF from each particle
        star_file: str or Path or None, RELION STAR file with CTF parameters
                   (required when subtract_ctf=True)
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
    
    # Open MRC as memmap (never loads into RAM) using the robust scheme
    data, success, msg = open_mrc_robust(input_path, max_size_gb=max_size_gb)
    
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

    # ---- CTF setup -------------------------------------------------------
    ctf_params_all = None
    if subtract_ctf:
        if star_file is None:
            print("❌ --subtract-ctf requires --star-file to be specified")
            return False
        print(f"\n🔬 CTF subtraction enabled")
        print(f"  Reading CTF parameters from: {star_file}")
        try:
            ctf_params_all = parse_star_file_ctf(star_file)
            print(f"  ✓ Loaded CTF parameters for {len(ctf_params_all)} particles")
            if len(ctf_params_all) < nz:
                print(f"  ⚠️  STAR file has fewer entries ({len(ctf_params_all)}) "
                        f"than MRC stack ({nz}). Extra particles will use the last entry.")
            elif len(ctf_params_all) > nz:
                print(f"  ⚠️  STAR file has more entries ({len(ctf_params_all)}) "
                        f"than MRC stack ({nz}). Extra entries will be ignored.")
        except Exception as e:
            print(f"❌ Failed to read STAR file: {e}")
            return False
    
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
    print(f"  RAM for one batch: {mem_est['input_batch_ram']:.2f} GB")
    print(f"  Estimated peak RAM usage: {mem_est['peak_ram']:.2f} GB")
    print(f"  Output file disk size: {mem_est['output_file_disk_size']:.2f} GB")
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

    # Warn if CTF subtraction is requested but pixel size is ambiguous
    if subtract_ctf:
        print(f"  CTF pixel size (after downsampling): {output_voxel_size:.3f} Å")
    
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
    print(f"  RAM for one batch: {mem_est['input_batch_ram']:.2f} GB")
    print(f"  Estimated peak RAM usage: {mem_est['peak_ram']:.2f} GB")
    print(f"  Output file disk size: {mem_est['output_file_disk_size']:.2f} GB")
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
            # Define the shape for the new memory-mapped file
            output_shape = (n_output_particles, target_size, target_size)

            # Use mrcfile.new_mmap for robust, memory-free file creation and allocation
            with mrcfile.new_mmap(output_path, shape=output_shape, mrc_mode=2, overwrite=True) as mrc:
                # Header dimensions (nx, ny, nz) and mode are set automatically by new_mmap.
                # We only need to set the remaining metadata.
                mrc.voxel_size = output_voxel_size
                mrc.header.map = b'MAP '
                mrc.header.cella.x = mrc.header.nx * mrc.voxel_size.x
                mrc.header.cella.y = mrc.header.ny * mrc.voxel_size.y
                mrc.header.cella.z = mrc.header.nz * mrc.voxel_size.z

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
                            
                            with torch.no_grad():
                                # Convert to torch tensor and move to device
                                batch_tensor = torch.from_numpy(batch).to(device)

                                # Downsample if needed
                                if ny != target_size or nx != target_size:
                                   batch_tensor = downsample_gpu(batch_tensor, target_size)

                                # CTF subtraction (after downsampling so pixel size is correct)
                                if subtract_ctf and ctf_params_all is not None:
                                    # Gather per-particle CTF params for this batch
                                    batch_ctf_params = [
                                        ctf_params_all[min(idx, len(ctf_params_all) - 1)]
                                        for idx in batch_indices
                                    ]
                                    ctf_batch = _build_ctf_from_params(
                                        batch_ctf_params,
                                        image_size=target_size,
                                        pixel_size=output_voxel_size,
                                        device=device,
                                    )
                                    batch_tensor = remove_ctf_batch(batch_tensor, ctf_batch)

                                # Normalize
                                batch_tensor = normalize_batch_gpu(batch_tensor, method=normalize,
                                                                  global_stats=global_stats)

                                # Write directly to mrc.data (no intermediate array)
                                mrc.data[i:end_idx] = batch_tensor.cpu().numpy()

                            # Clean up
                            del batch
                            if 'batch_tensor' in locals():
                                del batch_tensor
                            
                            pbar.update(end_idx - i)
                            
                            # Clear GPU cache periodically
                            if device == 'cuda' and (i // batch_size) % 10 == 0:
                                torch.cuda.empty_cache()
                    
                    print(f"✓ Processing complete")
                    # Update final header stats after processing is done
                    mrc.update_header_stats()
                    print(f"  Output shape: {mrc.data.shape}")
                    
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Processing interrupted by user")
                    return False
                except Exception as e:
                    print(f"\n\n❌ Error during processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                print(f"✓ Processing complete")
                # Update final header stats after processing is done
                mrc.update_header_stats()
                print(f"  Output shape: {mrc.data.shape}")
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Processing interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n❌ Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Clean up the input memmap object now that processing is complete
    del data
    gc.collect()
    
    # Verify output
    print(f"\n🔍 Verifying output...")
    try:
        with mrcfile.open(output_path, mode='r') as mrc:
            print(f"✓ Output file is valid")
            print(f"  Shape: {mrc.data.shape}")
            print(f"  Voxel size: {mrc.voxel_size.x:.3f} Å")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify output: {str(e)}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ SUCCESS: Processing complete")
    print(f"{'=' * 80}\n")
    
    return True
