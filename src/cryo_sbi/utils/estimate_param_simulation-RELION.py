#!/usr/bin/env python3
"""
estimate_param_simulation-RELION.py

GPU-accelerated extraction of realistic simulation parameters from cryo-EM data.
Uses consistent SNR and B-factor estimation methods.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import starfile
import mrcfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from typing import Optional, Tuple, Dict
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*delim_whitespace.*')
warnings.filterwarnings('ignore', message='.*errors.*', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mrcfile')

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
# CTF PARAMETERS EXTRACTION
# ============================================================================

def extract_ctf_parameters(star_file):
    """
    Extract CTF-related parameters from STAR file.
    
    Extracts:
    - rlnVoltage: Acceleration voltage (kV)
    - rlnSphericalAberration: Spherical aberration Cs (mm)
    - rlnCtfBfactor: B-factor for CTF (Å²)
    - rlnCtfScalefactor: Scale factor for CTF
    
    Returns:
    --------
    dict : Dictionary containing CTF parameters
    """
    print("\n" + "="*60)
    print("EXTRACTING CTF PARAMETERS")
    print("="*60)
    
    try:
        data = starfile.read(star_file)
    except Exception as e:
        print(f"\n❌ Error reading STAR file: {e}")
        return None
    
    ctf_params = {
        'voltage': None,
        'spherical_aberration': None,
        'bfactor': None,
        'scalefactor': None,
        'source': None
    }
    
    # Get particles table
    particles = data.get('particles', data)
    
    # ========================================================================
    # EXTRACT VOLTAGE (kV)
    # ========================================================================
    
    # Try optics table first (RELION 3.1+)
    if 'optics' in data and 'rlnVoltage' in data['optics'].columns:
        voltages = data['optics']['rlnVoltage'].dropna().unique()
        
        if len(voltages) > 0:
            ctf_params['voltage'] = float(voltages[0])
            ctf_params['source'] = 'optics table'
            
            if len(voltages) > 1:
                print(f"\n⚠️  Multiple voltages found: {voltages} kV")
                print(f"   Using first optics group: {ctf_params['voltage']:.1f} kV")
            else:
                print(f"\n✓ Voltage: {ctf_params['voltage']:.1f} kV")
    
    # Try particles table (older format)
    elif 'rlnVoltage' in particles.columns:
        voltages = particles['rlnVoltage'].dropna().unique()
        
        if len(voltages) > 0:
            ctf_params['voltage'] = float(voltages[0])
            ctf_params['source'] = 'particles table'
            
            if len(voltages) > 1:
                print(f"\n⚠️  Multiple voltages found in particles")
                print(f"   Using most common: {ctf_params['voltage']:.1f} kV")
            else:
                print(f"\n✓ Voltage: {ctf_params['voltage']:.1f} kV")
    
    if ctf_params['voltage'] is None:
        print(f"\n⚠️  Voltage not found, using default: 300 kV")
        ctf_params['voltage'] = 300.0
    else:
        # Validate voltage (typical cryo-EM: 100, 120, 200, 300 kV)
        typical_voltages = [80, 100, 120, 200, 300]
        if ctf_params['voltage'] not in typical_voltages:
            print(f"   ⚠️  Unusual voltage (typical: {typical_voltages} kV)")
    
    # ========================================================================
    # EXTRACT SPHERICAL ABERRATION Cs (mm)
    # ========================================================================
    
    # Try optics table first
    if 'optics' in data and 'rlnSphericalAberration' in data['optics'].columns:
        cs_values = data['optics']['rlnSphericalAberration'].dropna().unique()
        
        if len(cs_values) > 0:
            ctf_params['spherical_aberration'] = float(cs_values[0])
            
            if len(cs_values) > 1:
                print(f"\n⚠️  Multiple Cs values found: {cs_values} mm")
                print(f"   Using first optics group: {ctf_params['spherical_aberration']:.2f} mm")
            else:
                print(f"\n✓ Spherical aberration (Cs): {ctf_params['spherical_aberration']:.2f} mm")
    
    # Try particles table
    elif 'rlnSphericalAberration' in particles.columns:
        cs_values = particles['rlnSphericalAberration'].dropna().unique()
        
        if len(cs_values) > 0:
            ctf_params['spherical_aberration'] = float(cs_values[0])
            print(f"\n✓ Spherical aberration (Cs): {ctf_params['spherical_aberration']:.2f} mm")
    
    if ctf_params['spherical_aberration'] is None:
        print(f"\n⚠️  Spherical aberration not found, using default: 2.7 mm")
        ctf_params['spherical_aberration'] = 2.7
    else:
        # Validate Cs (typical range: 0.01 for Cs-corrected to 2.7 for uncorrected)
        if ctf_params['spherical_aberration'] < 0.001:
            print(f"   ℹ️  Very low Cs - likely Cs-corrected microscope")
        elif 2.0 <= ctf_params['spherical_aberration'] <= 2.7:
            print(f"   ℹ️  Standard uncorrected microscope Cs")
        elif ctf_params['spherical_aberration'] > 3.0:
            print(f"   ⚠️  Unusually high Cs value")
    
    # ========================================================================
    # EXTRACT CTF B-FACTOR (Å²)
    # ========================================================================
    
    # B-factor is usually per-particle, so get statistics
    if 'rlnCtfBfactor' in particles.columns:
        bfactor_values = particles['rlnCtfBfactor'].dropna()
        
        if len(bfactor_values) > 0:
            ctf_params['bfactor'] = {
                'mean': float(bfactor_values.mean()),
                'median': float(bfactor_values.median()),
                'std': float(bfactor_values.std()),
                'min': float(bfactor_values.min()),
                'max': float(bfactor_values.max()),
                'values': bfactor_values.values
            }
            
            print(f"\n✓ CTF B-factor statistics (Å²):")
            print(f"   Mean:   {ctf_params['bfactor']['mean']:.1f}")
            print(f"   Median: {ctf_params['bfactor']['median']:.1f}")
            print(f"   Range:  [{ctf_params['bfactor']['min']:.1f}, {ctf_params['bfactor']['max']:.1f}]")
            
            # Validate B-factor (typical range: 0-200 Å² for cryo-EM)
            if ctf_params['bfactor']['mean'] < 0:
                print(f"   ⚠️  Negative B-factor detected (sharpening applied)")
            elif ctf_params['bfactor']['mean'] > 200:
                print(f"   ⚠️  Very high B-factor (excessive blurring)")
        else:
            print(f"\n⚠️  CTF B-factor column exists but is empty")
            ctf_params['bfactor'] = None
    else:
        print(f"\n⚠️  CTF B-factor not found in STAR file")
        ctf_params['bfactor'] = None
    
    # ========================================================================
    # EXTRACT CTF SCALE FACTOR
    # ========================================================================
    
    # Scale factor is also usually per-particle
    if 'rlnCtfScalefactor' in particles.columns:
        scale_values = particles['rlnCtfScalefactor'].dropna()
        
        if len(scale_values) > 0:
            ctf_params['scalefactor'] = {
                'mean': float(scale_values.mean()),
                'median': float(scale_values.median()),
                'std': float(scale_values.std()),
                'min': float(scale_values.min()),
                'max': float(scale_values.max()),
                'values': scale_values.values
            }
            
            print(f"\n✓ CTF Scale factor statistics:")
            print(f"   Mean:   {ctf_params['scalefactor']['mean']:.3f}")
            print(f"   Median: {ctf_params['scalefactor']['median']:.3f}")
            print(f"   Range:  [{ctf_params['scalefactor']['min']:.3f}, {ctf_params['scalefactor']['max']:.3f}]")
            
            # Validate scale factor (typically around 1.0)
            if ctf_params['scalefactor']['mean'] < 0.5 or ctf_params['scalefactor']['mean'] > 2.0:
                print(f"   ⚠️  Unusual scale factor (expected ~1.0)")
        else:
            print(f"\n⚠️  CTF scale factor column exists but is empty")
            ctf_params['scalefactor'] = None
    else:
        print(f"\n⚠️  CTF scale factor not found in STAR file")
        ctf_params['scalefactor'] = None
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "-"*60)
    print("CTF PARAMETERS SUMMARY:")
    print("-"*60)
    print(f"  Voltage (kV):           {ctf_params['voltage']:.1f}")
    print(f"  Spherical aberration:   {ctf_params['spherical_aberration']:.2f} mm")
    
    if ctf_params['bfactor'] is not None:
        print(f"  B-factor (mean):        {ctf_params['bfactor']['mean']:.1f} Ų")
    else:
        print(f"  B-factor:               Not available")
    
    if ctf_params['scalefactor'] is not None:
        print(f"  Scale factor (mean):    {ctf_params['scalefactor']['mean']:.3f}")
    else:
        print(f"  Scale factor:           Not available")
    
    if ctf_params['source']:
        print(f"  Source:                 {ctf_params['source']}")
    
    print("="*60)
    
    return ctf_params


# ============================================================================
# PIXEL SIZE AND IMAGE INFORMATION EXTRACTION
# ============================================================================

def extract_pixel_and_image_info(star_file):
    """
    Extract pixel size, image dimensions, and particle count from STAR file.
    
    Handles:
    - Direct pixel size (rlnImagePixelSize)
    - Calculated from detector pixel size and magnification
    - Image/box size
    - Number of particles
    
    Returns:
    --------
    dict : Dictionary containing:
        - pixel_size: Pixel size in Angstroms
        - image_size: Box size in pixels (width, height)
        - num_particles: Total number of particles
        - calculation_method: How pixel size was determined
    """
    print("\n" + "="*60)
    print("EXTRACTING PIXEL SIZE AND IMAGE INFORMATION")
    print("="*60)
    
    try:
        data = starfile.read(star_file)
    except Exception as e:
        print(f"\n❌ Error reading STAR file: {e}")
        return None
    
    info = {
        'pixel_size': None,
        'image_size': None,
        'num_particles': 0,
        'calculation_method': None,
        'mrc_read_method': None
    }
    
    # Get particles table
    particles = data.get('particles', data)
    info['num_particles'] = len(particles)
    
    print(f"\n📊 Number of particles: {info['num_particles']:,}")
    
    # ========================================================================
    # EXTRACT PIXEL SIZE
    # ========================================================================
    
    # Method 1: Direct from optics table (RELION 3.1+)
    if 'optics' in data:
        optics = data['optics']
        
        # Try rlnImagePixelSize (most direct)
        if 'rlnImagePixelSize' in optics.columns:
            pixel_sizes = optics['rlnImagePixelSize'].dropna().unique()
            
            if len(pixel_sizes) > 0:
                info['pixel_size'] = float(pixel_sizes[0])
                info['calculation_method'] = 'Direct (rlnImagePixelSize)'
                
                if len(pixel_sizes) > 1:
                    print(f"\n⚠️  Multiple pixel sizes found: {pixel_sizes}")
                    print(f"   Using first optics group: {info['pixel_size']:.3f} Å/px")
                else:
                    print(f"\n✓ Pixel size: {info['pixel_size']:.3f} Å/px")
        
        # Method 2: Calculate from detector pixel size and magnification
        if info['pixel_size'] is None:
            if 'rlnDetectorPixelSize' in optics.columns and 'rlnMagnification' in optics.columns:
                detector_px = optics['rlnDetectorPixelSize'].values[0]  # micrometers
                magnification = optics['rlnMagnification'].values[0]
                
                # Pixel size (Å) = detector_pixel_size (µm) * 10000 / magnification
                info['pixel_size'] = float(detector_px * 10000.0 / magnification)
                info['calculation_method'] = 'Calculated (detector/magnification)'
                
                print(f"\n✓ Calculated pixel size:")
                print(f"  Detector pixel size: {detector_px:.3f} µm")
                print(f"  Magnification: {magnification:,.0f}x")
                print(f"  → Pixel size: {info['pixel_size']:.3f} Å/px")
    
    # Method 3: From particles table (older RELION format)
    if info['pixel_size'] is None and 'rlnDetectorPixelSize' in particles.columns:
        if 'rlnMagnification' in particles.columns:
            detector_px = particles['rlnDetectorPixelSize'].values[0]
            magnification = particles['rlnMagnification'].values[0]
            
            info['pixel_size'] = float(detector_px * 10000.0 / magnification)
            info['calculation_method'] = 'Calculated (particles table)'
            
            print(f"\n✓ Calculated pixel size (from particles):")
            print(f"  Detector pixel size: {detector_px:.3f} µm")
            print(f"  Magnification: {magnification:,.0f}x")
            print(f"  → Pixel size: {info['pixel_size']:.3f} Å/px")
    
    # Validation
    if info['pixel_size'] is not None:
        if not (0.1 <= info['pixel_size'] <= 10.0):
            print(f"\n⚠️  WARNING: Unusual pixel size: {info['pixel_size']:.3f} Å/px")
            print(f"   Expected range: 0.5-5.0 Å/px for typical cryo-EM")
    else:
        print(f"\n⚠️  Could not determine pixel size from STAR file metadata")
    
    # ========================================================================
    # EXTRACT IMAGE SIZE (BOX SIZE)
    # ========================================================================
    
    # Try optics table first
    if 'optics' in data and 'rlnImageSize' in data['optics'].columns:
        box_size = int(data['optics']['rlnImageSize'].values[0])
        info['image_size'] = (box_size, box_size)
        print(f"\n✓ Image size: {box_size} × {box_size} pixels")
    
    # Try particles table
    elif 'rlnImageSize' in particles.columns:
        box_size = int(particles['rlnImageSize'].values[0])
        info['image_size'] = (box_size, box_size)
        print(f"\n✓ Image size: {box_size} × {box_size} pixels")
    
    # Try to infer from image names using ROBUST MRC reading
    elif 'rlnImageName' in particles.columns:
        print(f"\n🔍 Attempting to read MRC file to determine image size...")
        
        # Image names are like "000001@path/to/stack.mrcs"
        first_image = particles['rlnImageName'].values[0]
        
        if '@' in first_image:
            stack_path = first_image.split('@')[1]
            
            # Check if path is relative to STAR file location
            stack_path_obj = Path(stack_path)
            if not stack_path_obj.is_absolute():
                star_dir = Path(star_file).parent
                stack_path_obj = star_dir / stack_path
            
            if stack_path_obj.exists():
                print(f"  Reading: {stack_path_obj}")
                
                # Use the robust MRC reading function
                mrc_data, success, method = open_mrc_robust(stack_path_obj)
                
                if success and mrc_data is not None:
                    info['mrc_read_method'] = method
                    
                    # Get dimensions
                    if mrc_data.ndim == 3:
                        # Stack format: (n_images, ny, nx)
                        nz, ny, nx = mrc_data.shape
                        info['image_size'] = (nx, ny)
                        print(f"  ✓ MRC read successful ({method})")
                        print(f"  ✓ Stack contains {nz} images")
                        print(f"  ✓ Image size: {nx} × {ny} pixels")
                    elif mrc_data.ndim == 2:
                        # Single image: (ny, nx)
                        ny, nx = mrc_data.shape
                        info['image_size'] = (nx, ny)
                        print(f"  ✓ MRC read successful ({method})")
                        print(f"  ✓ Image size: {nx} × {ny} pixels")
                    else:
                        print(f"  ⚠️  Unexpected MRC dimensions: {mrc_data.ndim}D")
                    
                    # Clean up memmap if needed
                    if isinstance(mrc_data, np.memmap):
                        del mrc_data
                else:
                    print(f"  ❌ Failed to read MRC file: {method}")
                    print(f"     This might indicate a corrupted file or unsupported format")
            else:
                print(f"  ⚠️  MRC file not found: {stack_path_obj}")
                print(f"     Checked absolute and relative to STAR file location")
        else:
            print(f"  ⚠️  Unexpected image name format: {first_image}")
    
    if info['image_size'] is None:
        print(f"\n⚠️  Could not determine image size")
    
    # ========================================================================
    # CALCULATE PHYSICAL IMAGE SIZE
    # ========================================================================
    
    if info['pixel_size'] is not None and info['image_size'] is not None:
        physical_size = info['image_size'][0] * info['pixel_size']
        info['physical_size_angstrom'] = physical_size
        info['physical_size_nm'] = physical_size / 10
        print(f"\n✓ Physical image size: {physical_size:.1f} Å ({physical_size/10:.1f} nm)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "-"*60)
    print("SUMMARY:")
    print("-"*60)
    print(f"  Particles:      {info['num_particles']:,}")
    if info['pixel_size']:
        print(f"  Pixel size:     {info['pixel_size']:.3f} Å/px ({info['calculation_method']})")
    if info['image_size']:
        print(f"  Box size:       {info['image_size'][0]} × {info['image_size'][1]} px")
        if info['mrc_read_method']:
            print(f"  MRC method:     {info['mrc_read_method']}")
        if info['pixel_size']:
            print(f"  Physical size:  {info['physical_size_angstrom']:.1f} Å ({info['physical_size_nm']:.1f} nm)")
    print("="*60)
    
    return info


# ============================================================================
# DEFOCUS EXTRACTION
# ============================================================================

def extract_defocus_statistics(star_file):
    """Extract defocus statistics from STAR file."""
    print("\n" + "="*60)
    print("EXTRACTING DEFOCUS PARAMETERS")
    print("="*60)
    
    data = starfile.read(star_file)
    particles = data['particles'] if 'particles' in data else data
    
    defocus_u = particles['rlnDefocusU'].values / 10000  # Å → µm
    defocus_v = particles['rlnDefocusV'].values / 10000
    defocus_avg = (defocus_u + defocus_v) / 2
    
    stats = {
        'min': float(defocus_avg.min()),
        'max': float(defocus_avg.max()),
        'mean': float(defocus_avg.mean()),
        'median': float(np.median(defocus_avg)),
        'std': float(defocus_avg.std()),
        'p25': float(np.percentile(defocus_avg, 25)),
        'p75': float(np.percentile(defocus_avg, 75)),
    }
    
    print(f"\n✓ Defocus Statistics (µm):")
    print(f"  Range:  {stats['min']:.2f} - {stats['max']:.2f} µm")
    print(f"  Mean:   {stats['mean']:.2f} µm")
    print(f"  Median: {stats['median']:.2f} µm")
    
    recommended_min = max(0.5, stats['p25'] - 0.5)
    recommended_max = min(5.0, stats['p75'] + 0.5)
    
    print(f"✓ Recommended: [{recommended_min:.2f}, {recommended_max:.2f}] µm")
    
    stats['recommended_min'] = recommended_min
    stats['recommended_max'] = recommended_max
    stats['all_values'] = defocus_avg
    
    return stats


def extract_amplitude_contrast(star_file):
    """Extract amplitude contrast from STAR file."""
    print("\n" + "="*60)
    print("EXTRACTING AMPLITUDE CONTRAST")
    print("="*60)
    
    data = starfile.read(star_file)
    
    if 'optics' in data:
        amp = float(data['optics']['rlnAmplitudeContrast'].values[0])
        print(f"\n✓ Amplitude contrast: {amp:.3f}")
        return amp
    
    particles = data['particles'] if 'particles' in data else data
    if 'rlnAmplitudeContrast' in particles.columns:
        amp = float(particles['rlnAmplitudeContrast'].values[0])
        print(f"\n✓ Amplitude contrast: {amp:.3f}")
        return amp
    
    print("\n⚠️  Amplitude contrast not found, using default: 0.1")
    return 0.1


# ============================================================================
# GAUSSIAN SIGMA ESTIMATION
# ============================================================================

def estimate_gaussian_sigma(star_file, model_type='1bead', reported_resolution=None):
    """Estimate Gaussian sigma based on model type and resolution."""
    print("\n" + "="*60)
    print("ESTIMATING GAUSSIAN SIGMA (MODEL-AWARE)")
    print("="*60)
    
    data = starfile.read(star_file)
    particles = data['particles'] if 'particles' in data else data
    
    # Get resolution
    if 'rlnCtfMaxResolution' in particles.columns:
        resolutions = particles['rlnCtfMaxResolution'].values
        median_resolution = float(np.median(resolutions))
        print(f"\nMedian CTF resolution: {median_resolution:.1f} Å")
    else:
        median_resolution = None
    
    if reported_resolution is not None:
        resolution = reported_resolution
        print(f"Using reported resolution: {resolution:.1f} Å")
    elif median_resolution is not None:
        resolution = median_resolution
    else:
        resolution = 10.0
        print(f"⚠️  No resolution info, assuming: {resolution:.1f} Å")
    
    model_type_lower = model_type.lower()
    
    print(f"\nModel type: {model_type}")
    
    if model_type_lower in ['allatom', 'all-atom', 'atomic']:
        sigma_recommended = 1.5
        sigma_min_raw = 0.5
        sigma_max_raw = 2.5
        
        print("  ═══════════════════════════════════════")
        print("  Type: ALL-ATOM")
        print("  ═══════════════════════════════════════")
        print("  • Every atom explicitly represented")
        print("  • Typical atomic radii: 1-2 Å")
        
    elif model_type_lower in ['1bead', 'one-bead', '1-bead', 'ca', 'c-alpha']:
        sigma_recommended = min(3.0, resolution / 3)
        sigma_min_raw = max(1.5, resolution / 5)
        sigma_max_raw = min(4.5, resolution / 2)
        
        print("  ═══════════════════════════════════════")
        print("  Type: ONE BEAD PER RESIDUE")
        print("  ═══════════════════════════════════════")
        print("  • One bead per amino acid (e.g., C-α)")
        print("  • Bead spacing: ~3.8 Å")
        
    elif model_type_lower in ['2bead', 'two-bead', '2-bead', 'bb-sc']:
        sigma_recommended = min(2.5, resolution / 3.5)
        sigma_min_raw = max(1.0, resolution / 6)
        sigma_max_raw = min(4.0, resolution / 2.5)
        
        print("  ═══════════════════════════════════════")
        print("  Type: TWO BEADS PER RESIDUE")
        print("  ═══════════════════════════════════════")
        print("  • Backbone bead + Sidechain bead")
        
    else:
        print(f"  ⚠️  Unknown model type, using 1bead defaults")
        sigma_recommended = min(3.0, resolution / 3)
        sigma_min_raw = max(1.5, resolution / 5)
        sigma_max_raw = min(4.5, resolution / 2)
    
    # First, ensure min < max
    if sigma_min_raw >= sigma_max_raw:
        # If they're equal or inverted, create a valid range
        sigma_center = (sigma_min_raw + sigma_max_raw) / 2
        sigma_min = sigma_center * 0.7
        sigma_max = sigma_center * 1.3
        print(f"\n  ⚠️  Adjusted sigma range (min >= max detected)")
    else:
        sigma_min = sigma_min_raw
        sigma_max = sigma_max_raw
    
    # Ensure recommended is within range
    if sigma_recommended < sigma_min:
        sigma_recommended = sigma_min
    elif sigma_recommended > sigma_max:
        sigma_recommended = sigma_max
    
    # Final sanity check: ensure we have a valid range with some spread
    if sigma_max - sigma_min < 0.5:
        # Too narrow, expand around recommended
        sigma_min = max(0.5, sigma_recommended - 0.5)
        sigma_max = sigma_recommended + 0.5
        print(f"  ⚠️  Expanded narrow sigma range")
    
    # Final validation
    assert sigma_min < sigma_max, f"Invalid sigma range: [{sigma_min}, {sigma_max}]"
    assert sigma_min <= sigma_recommended <= sigma_max, \
        f"Sigma recommended {sigma_recommended} not in range [{sigma_min}, {sigma_max}]"
    
    print(f"\n  Resolution: {resolution:.1f} Å")
    print(f"  → Recommended σ: {sigma_recommended:.2f} Å")
    print(f"  → Suggested range: [{sigma_min:.2f}, {sigma_max:.2f}] Å")
    
    # Verify the range
    print(f"  ✓ Validated: {sigma_min:.2f} < {sigma_recommended:.2f} < {sigma_max:.2f}")
    
    return {
        'resolution': resolution,
        'model_type': model_type,
        'recommended': float(sigma_recommended),
        'min': float(sigma_min),
        'max': float(sigma_max),
    }

# ============================================================================
# CONFIGURATION GENERATION
# ============================================================================

def generate_config(defocus_stats, amp, sigma_stats, pixel_info, ctf_params):
    """Generate complete configuration dictionary."""
    
    config = {
        "SIGMA": [sigma_stats['min'], sigma_stats['max']],
        "DEFOCUS": [defocus_stats['recommended_min'], defocus_stats['recommended_max']],
        "AMP": amp
    }
    
    # Add CTF parameters
    if ctf_params:
        config['VOLTAGE'] = ctf_params['voltage']
        config['CS'] = ctf_params['spherical_aberration']
        
        if ctf_params['bfactor'] is not None:
            config['BFACTOR_MEAN'] = ctf_params['bfactor']['mean']
            config['BFACTOR_RANGE'] = [ctf_params['bfactor']['min'], ctf_params['bfactor']['max']]
        
        if ctf_params['scalefactor'] is not None:
            config['SCALEFACTOR_MEAN'] = ctf_params['scalefactor']['mean']
            config['SCALEFACTOR_RANGE'] = [ctf_params['scalefactor']['min'], ctf_params['scalefactor']['max']]
    
    # Add pixel and image information
    if pixel_info:
        if pixel_info['pixel_size']:
            config['PIXEL_SIZE'] = pixel_info['pixel_size']
        if pixel_info['image_size']:
            config['BOX_SIZE'] = pixel_info['image_size'][0]
        config['NUM_PARTICLES'] = pixel_info['num_particles']
        
        if 'physical_size_angstrom' in pixel_info:
            config['PHYSICAL_SIZE_ANGSTROM'] = pixel_info['physical_size_angstrom']
            config['PHYSICAL_SIZE_NM'] = pixel_info['physical_size_nm']
    
    print("\n📝 FINAL CONFIGURATION SUMMARY:")
    print("="*60)
    print(f"  Model Type:     {sigma_stats['model_type']}")
    print(f"  Resolution:     {sigma_stats['resolution']:.1f} Å")
    print("-"*60)
    print("  SIMULATION PARAMETERS:")
    print(f"    Sigma (σ):      [{config['SIGMA'][0]:.2f}, {config['SIGMA'][1]:.2f}] Å")
    print(f"    Defocus:        [{config['DEFOCUS'][0]:.2f}, {config['DEFOCUS'][1]:.2f}] µm")
    print(f"    Amplitude (A):  {config['AMP']:.3f}")
    print("-"*60)
    print("  CTF PARAMETERS:")
    if 'VOLTAGE' in config:
        print(f"    Voltage:        {config['VOLTAGE']:.1f} kV")
    if 'CS' in config:
        print(f"    Cs:             {config['CS']:.2f} mm")
    if 'BFACTOR_MEAN' in config:
        print(f"    B-factor:       {config['BFACTOR_MEAN']:.1f} Ų (mean)")
        print(f"                    [{config['BFACTOR_RANGE'][0]:.1f}, {config['BFACTOR_RANGE'][1]:.1f}] Ų (range)")
    if 'SCALEFACTOR_MEAN' in config:
        print(f"    Scale factor:   {config['SCALEFACTOR_MEAN']:.3f} (mean)")
    print("-"*60)
    print("  IMAGE PARAMETERS:")
    if 'PIXEL_SIZE' in config:
        print(f"    Pixel size:     {config['PIXEL_SIZE']:.3f} Å/px")
    if 'BOX_SIZE' in config:
        print(f"    Box size:       {config['BOX_SIZE']} px")
    if 'PHYSICAL_SIZE_ANGSTROM' in config:
        print(f"    Physical size:  {config['PHYSICAL_SIZE_ANGSTROM']:.1f} Å ({config['PHYSICAL_SIZE_NM']:.1f} nm)")
    if 'NUM_PARTICLES' in config:
        print(f"    Particles:      {config['NUM_PARTICLES']:,}")
    print("="*60)
    
    return config


def save_config_to_json(config, output_path, star_file, args):
    """Save configuration to JSON file with metadata."""
    
    output_data = {
        "metadata": {
            "source_star_file": str(star_file),
            "model_type": args.model_type,
            "reported_resolution": args.resolution,
            "extraction_date": pd.Timestamp.now().isoformat()
        },
        "simulation_parameters": {
            "sigma_range": config['SIGMA'],
            "defocus_range": config['DEFOCUS'],
            "amplitude_contrast": config['AMP']
        },
        "ctf_parameters": {
            "voltage_kv": config.get('VOLTAGE'),
            "spherical_aberration_mm": config.get('CS'),
            "bfactor_mean": config.get('BFACTOR_MEAN'),
            "bfactor_range": config.get('BFACTOR_RANGE'),
            "scalefactor_mean": config.get('SCALEFACTOR_MEAN'),
            "scalefactor_range": config.get('SCALEFACTOR_RANGE')
        },
        "image_parameters": {
            "pixel_size_angstrom": config.get('PIXEL_SIZE'),
            "box_size_pixels": config.get('BOX_SIZE'),
            "physical_size_angstrom": config.get('PHYSICAL_SIZE_ANGSTROM'),
            "physical_size_nm": config.get('PHYSICAL_SIZE_NM'),
            "num_particles": config.get('NUM_PARTICLES')
        }
    }
    
    # Remove None values
    def remove_none(d):
        if isinstance(d, dict):
            return {k: remove_none(v) for k, v in d.items() if v is not None}
        return d
    
    output_data = remove_none(output_data)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Configuration saved to: {output_path}")


def print_usage_examples(config):
    """Print usage examples for simulation."""
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  USAGE EXAMPLES".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\n📋 Python Code Example:")
    print("-"*60)
    print("import numpy as np")
    print("")
    print("# Simulation parameters")
    print(f"sigma = np.random.uniform({config['SIGMA'][0]:.2f}, {config['SIGMA'][1]:.2f})  # Gaussian sigma (Å)")
    print(f"defocus = np.random.uniform({config['DEFOCUS'][0]:.2f}, {config['DEFOCUS'][1]:.2f})  # Defocus (µm)")
    print(f"amp_contrast = {config['AMP']:.3f}  # Amplitude contrast")
    
    if 'VOLTAGE' in config:
        print(f"voltage = {config['VOLTAGE']:.1f}  # kV")
    if 'CS' in config:
        print(f"cs = {config['CS']:.2f}  # mm")
    if 'PIXEL_SIZE' in config:
        print(f"pixel_size = {config['PIXEL_SIZE']:.3f}  # Å/px")
    if 'BOX_SIZE' in config:
        print(f"box_size = {config['BOX_SIZE']}  # pixels")
    
    print("-"*60)
    
    print("\n📋 Command Line Example (if applicable):")
    print("-"*60)
    print("python simulate_projections.py \\")
    print(f"    --sigma {config['SIGMA'][0]:.2f} {config['SIGMA'][1]:.2f} \\")
    print(f"    --defocus {config['DEFOCUS'][0]:.2f} {config['DEFOCUS'][1]:.2f} \\")
    print(f"    --amp {config['AMP']:.3f} \\")
    
    if 'VOLTAGE' in config:
        print(f"    --voltage {config['VOLTAGE']:.1f} \\")
    if 'CS' in config:
        print(f"    --cs {config['CS']:.2f} \\")
    if 'PIXEL_SIZE' in config:
        print(f"    --pixel-size {config['PIXEL_SIZE']:.3f} \\")
    if 'BOX_SIZE' in config:
        print(f"    --box-size {config['BOX_SIZE']} \\")
    
    print("    --output simulated_particles.mrcs")
    print("-"*60)


# ========================================================================
# MAIN 
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extraction of simulation parameters from cryo-EM data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types:
  allatom  - All-atom representation
  1bead    - One bead per residue (C-alpha)
  2bead    - Two beads per residue (backbone + sidechain)

Examples:
  # Basic usage
  python estimate_param_simulation-RELION.py particles.star \\
      --model-type 1bead --resolution 6.5

  # With JSON output
  python estimate_param_simulation-RELION.py particles.star \\
      --model-type 1bead --resolution 6.5 \\
      --output simulation_params.json

  # All-atom model
  python estimate_param_simulation-RELION.py particles.star \\
      --model-type allatom --resolution 3.5 \\
      --output params.json
        """
    )
    
    parser.add_argument('star_file', type=str,
                        help='Input STAR file from RELION')
    parser.add_argument('--model-type', type=str, default='1bead',
                        choices=['allatom', '1bead', '2bead'],
                        help='Type of coarse-graining (default: 1bead)')
    parser.add_argument('--resolution', type=float, default=None,
                        help='Reported resolution in Angstroms')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for parameters (optional)')
    parser.add_argument('--print-examples', action='store_true',
                        help='Print usage examples at the end')
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  CRYO-EM SIMULATION PARAMETER EXTRACTION".center(58) + "║")
    print("║" + "  FROM RELION STAR FILES".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print(f"\n📁 Input: {args.star_file}")
    print(f"🧬 Model type: {args.model_type}")
    if args.resolution:
        print(f"🔬 Resolution: {args.resolution:.1f} Å")
    if args.output:
        print(f"💾 Output: {args.output}")
    
    # Check file exists
    if not Path(args.star_file).exists():
        print(f"\n❌ Error: File not found: {args.star_file}")
        return 1
    
    # Extract parameters
    print("\n" + "█"*60)
    print("  STEP 1/5: EXTRACTING PIXEL SIZE AND IMAGE INFO")
    print("█"*60)
    pixel_info = extract_pixel_and_image_info(args.star_file)
    
    print("\n" + "█"*60)
    print("  STEP 2/5: EXTRACTING CTF PARAMETERS")
    print("█"*60)
    ctf_params = extract_ctf_parameters(args.star_file)
    
    print("\n" + "█"*60)
    print("  STEP 3/5: EXTRACTING DEFOCUS")
    print("█"*60)
    defocus_stats = extract_defocus_statistics(args.star_file)
    
    print("\n" + "█"*60)
    print("  STEP 4/5: EXTRACTING AMPLITUDE CONTRAST")
    print("█"*60)
    amp = extract_amplitude_contrast(args.star_file)
    
    print("\n" + "█"*60)
    print("  STEP 5/5: ESTIMATING GAUSSIAN SIGMA")
    print("█"*60)
    sigma_stats = estimate_gaussian_sigma(
        args.star_file,
        model_type=args.model_type,
        reported_resolution=args.resolution
    )
    
    # Generate configuration
    config = generate_config(defocus_stats, amp, sigma_stats, pixel_info, ctf_params)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        save_config_to_json(config, output_path, args.star_file, args)
    
    # Print usage examples if requested
    if args.print_examples:
        print_usage_examples(config)
    
    # Success message
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "✓ EXTRACTION COMPLETE!".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\n💡 Tips:")
    print("  • Use --output to save parameters to JSON")
    print("  • Use --print-examples to see usage examples")
    print("  • Adjust sigma range based on your model resolution")
    print("  • Consider multiple defocus values for realistic simulation")
    
    if not args.output:
        print("\n💡 Hint: Add --output params.json to save these parameters!")
    
    if not args.print_examples:
        print("💡 Hint: Add --print-examples to see simulation code examples!")
    
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
