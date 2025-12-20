#!/usr/bin/env python3
"""
estimate_param_simulation.py
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm
from scipy.optimize import minimize

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

def extract_defocus_statistics(star_file: str, output_plot_path: Optional[str] = None):
    """
    Extracts defocus statistics, fits a truncated Gaussian, and plots the distribution.

    Args:
    - star_file (str): Path to the input STAR file.
    - output_plot_path (Optional[str]): If provided, saves a plot of the defocus
      distribution to this path (e.g., "defocus_distribution.png").

    Returns:
    - dict: A dictionary containing defocus statistics and fitted parameters.
    """
    print("\n" + "="*60)
    print("EXTRACTING AND ANALYZING DEFOCUS PARAMETERS")
    print("="*60)
    
    try:
        data = starfile.read(star_file)
        particles = data['particles'] if 'particles' in data else data
    except Exception as e:
        print(f"❌ Error reading STAR file: {e}")
        return {}

    if not all(k in particles.columns for k in ['rlnDefocusU', 'rlnDefocusV']):
        print("❌ 'rlnDefocusU' or 'rlnDefocusV' columns not found.")
        return {}

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
    
    print(f"\n✓ Basic Defocus Statistics (µm):")
    print(f"  Range:  {stats['min']:.2f} - {stats['max']:.2f} µm")
    print(f"  Mean:   {stats['mean']:.2f} µm")
    print(f"  Median: {stats['median']:.2f} µm")
    print(f"  StdDev: {stats['std']:.2f} µm")
    
    # --- NEW: Truncated Gaussian Fitting ---
    print("\n" + "-"*60)
    print("FITTING TRUNCATED GAUSSIAN DISTRIBUTION")
    print("-"*60)
    
    # Define bounds for the fit
    lower_bound, upper_bound = stats['min'], stats['max']

    # The negative log-likelihood function to minimize
    def neg_log_likelihood(params, data):
        loc, scale = params
        if scale <= 0: # Scale must be positive
            return np.inf
        # Calculate a and b parameters for truncnorm in standard units
        a = (lower_bound - loc) / scale
        b = (upper_bound - loc) / scale
        # Calculate log-likelihood and return its negative
        log_likelihood = np.sum(truncnorm.logpdf(data, a=a, b=b, loc=loc, scale=scale))
        return -log_likelihood

    # Initial guess and optimization
    if stats['std'] > 0:
        initial_guess = [stats['mean'], stats['std']]
        result = minimize(
            neg_log_likelihood,
            initial_guess,
            args=(defocus_avg,),
            method='Nelder-Mead'
        )
        if result.success:
            fit_loc, fit_scale = result.x
            stats['fit_loc'] = float(fit_loc)
            stats['fit_scale'] = float(fit_scale)
            print("✓ Fit successful.")
            print(f"  Fitted Location (µ): {fit_loc:.2f}")
            print(f"  Fitted Scale (σ):    {fit_scale:.2f}")
        else:
            print("❌ Fitting failed. Using sample mean/std as fallback.")
            stats['fit_loc'] = stats['mean']
            stats['fit_scale'] = stats['std']
    else:
        print("⚠️ Data has zero variance. Skipping fit.")
        stats['fit_loc'] = stats['mean']
        stats['fit_scale'] = 0.0

    # --- NEW: Plotting ---
    if output_plot_path:
        print("\n" + "-"*60)
        print(f"GENERATING PLOT: {output_plot_path}")
        print("-"*60)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram and KDE
        sns.histplot(defocus_avg, bins='auto', stat='density', kde=True, ax=ax,
                     label='Data (Histogram + KDE)', color='skyblue', alpha=0.7)

        # Plot the fitted Truncated Gaussian PDF
        x_fit = np.linspace(lower_bound, upper_bound, 400)
        fit_loc, fit_scale = stats['fit_loc'], stats['fit_scale']
        a_fit = (lower_bound - fit_loc) / fit_scale
        b_fit = (upper_bound - fit_loc) / fit_scale
        y_fit = truncnorm.pdf(x_fit, a=a_fit, b=b_fit, loc=fit_loc, scale=fit_scale)
        ax.plot(x_fit, y_fit, 'r-', lw=2.5, label=f'Truncated Gaussian Fit\n(loc={fit_loc:.2f}, scale={fit_scale:.2f})')

        ax.set_title('Defocus Distribution Analysis', fontsize=16)
        ax.set_xlabel('Average Defocus (µm)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        try:
            plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot successfully saved to {output_plot_path}")
        except Exception as e:
            print(f"❌ Failed to save plot: {e}")
        plt.close(fig)

    recommended_min = max(0.5, stats['p25'] - 0.5)
    recommended_max = min(5.0, stats['p75'] + 0.5)
    
    print("\n" + "-"*60)
    print("RECOMMENDATIONS")
    print("-"*60)
    print(f"✓ Recommended defocus range for simulation: [{recommended_min:.2f}, {recommended_max:.2f}] µm")
    
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
# CONFIGURATION GENERATION
# ============================================================================

def generate_config(defocus_stats, amp, pixel_info, ctf_params):
    """Generate complete configuration dictionary."""
    
    config = {
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
    print("-"*60)
    print("  SIMULATION PARAMETERS:")
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

def estimate_param_simulation_RELION(star_file):
    # Extract parameters
    print("\n" + "█"*60)
    print("  STEP 1/4: EXTRACTING PIXEL SIZE AND IMAGE INFO")
    print("█"*60)
    pixel_info = extract_pixel_and_image_info(star_file)
    
    print("\n" + "█"*60)
    print("  STEP 2/4: EXTRACTING CTF PARAMETERS")
    print("█"*60)
    ctf_params = extract_ctf_parameters(star_file)
    
    print("\n" + "█"*60)
    print("  STEP 3/4: EXTRACTING DEFOCUS")
    print("█"*60)
    defocus_stats = extract_defocus_statistics(star_file, "defocus_distribution.png")
    
    print("\n" + "█"*60)
    print("  STEP 4/4: EXTRACTING AMPLITUDE CONTRAST")
    print("█"*60)
    amp = extract_amplitude_contrast(star_file)

    # generate final configuration config
    config = generate_config(defocus_stats, amp, pixel_info, ctf_params)
