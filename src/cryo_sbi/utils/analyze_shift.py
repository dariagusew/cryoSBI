#!/usr/bin/env python3
"""
Cryo-EM Particle Shift Distribution Analyzer (Simplified)

Estimates translational shifts in cryo-EM particle stacks using center of mass
method with different filtering aggressiveness levels.

Author: Cryo-EM Analysis Tools
Version: 2.0.0
"""

import argparse
import sys
import os
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, fftfreq
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

__version__ = "2.0.0"


def low_pass_filter(image, cutoff, order=2):
    """
    Apply Butterworth low-pass filter
    
    Parameters:
    -----------
    image : 2D array
    cutoff : float
        Cutoff frequency (0-0.5)
    order : int
        Filter order (higher = sharper cutoff)
    """
    rows, cols = image.shape
    
    # Compute 2D frequency coordinates
    freq_y = fftfreq(rows)
    freq_x = fftfreq(cols)
    freq_y, freq_x = np.meshgrid(freq_y, freq_x, indexing='ij')
    freq_radius = np.sqrt(freq_y**2 + freq_x**2)
    
    # Butterworth filter
    filter_mask = 1.0 / (1.0 + (freq_radius / cutoff)**(2 * order))
    
    # Apply filter
    f_image = fftshift(fft2(image))
    f_filtered = f_image * filter_mask
    filtered = np.real(ifft2(fftshift(f_filtered)))
    
    return filtered


def calculate_confidence(thresholded_image, shift, mask_radius):
    """Estimate confidence in the shift measurement"""
    if np.sum(thresholded_image) == 0:
        return 0.0
    
    rows, cols = thresholded_image.shape
    center = np.array([rows / 2.0, cols / 2.0])
    estimated_center = center - shift
    
    y_grid, x_grid = np.ogrid[:rows, :cols]
    distances = np.sqrt((y_grid - estimated_center[0])**2 + 
                       (x_grid - estimated_center[1])**2)
    
    weighted_dist = np.sum(distances * thresholded_image) / np.sum(thresholded_image)
    confidence = np.exp(-weighted_dist / (mask_radius * 0.5))
    
    # Penalize unreasonably large shifts
    shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)
    max_reasonable_shift = min(rows, cols) * 0.35
    
    if shift_magnitude > max_reasonable_shift:
        confidence *= 0.5
    
    return confidence


def center_of_mass_shift(image, cutoff, percentile=65, n_iterations=3):
    """
    Estimate shift using center of mass with iterative refinement
    
    Parameters:
    -----------
    image : 2D array
    cutoff : float
        Low-pass filter cutoff frequency
    percentile : float
        Percentile for thresholding
    n_iterations : int
        Number of refinement iterations
    """
    rows, cols = image.shape
    center = np.array([rows / 2.0, cols / 2.0])
    
    # Filter
    filtered = low_pass_filter(image, cutoff)
    
    # Normalize
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
    
    # Iterative refinement
    current_shift = np.array([0.0, 0.0])
    
    for iteration in range(n_iterations):
        # Shrinking circular mask
        mask_radius = min(rows, cols) * (0.4 - 0.1 * iteration / n_iterations)
        
        y_grid, x_grid = np.ogrid[:rows, :cols]
        current_center = center - current_shift
        
        circular_mask = ((y_grid - current_center[0])**2 + 
                        (x_grid - current_center[1])**2) <= mask_radius**2
        
        masked = filtered * circular_mask
        
        # Threshold
        threshold = np.percentile(masked[circular_mask], percentile)
        thresholded = np.maximum(masked - threshold, 0)
        
        # Calculate center of mass
        if np.sum(thresholded) > 0:
            com_y = np.sum(y_grid * thresholded) / np.sum(thresholded)
            com_x = np.sum(x_grid * thresholded) / np.sum(thresholded)
            
            # Damped update
            new_shift = center - np.array([com_y, com_x])
            damping = 0.7
            current_shift = damping * new_shift + (1 - damping) * current_shift
    
    # Calculate confidence
    confidence = calculate_confidence(thresholded, current_shift, mask_radius)
    
    return current_shift[0], current_shift[1], confidence


def analyze_with_cutoff(particles, cutoff, cutoff_name, verbose=True):
    """
    Analyze all particles with a specific cutoff
    """
    n_particles = particles.shape[0]
    
    if verbose:
        print(f"\nProcessing with {cutoff_name} filter (cutoff={cutoff:.3f})...")
    
    shifts = np.zeros((n_particles, 2))
    confidences = np.zeros(n_particles)
    
    for i in range(n_particles):
        shift_y, shift_x, confidence = center_of_mass_shift(
            particles[i], cutoff=cutoff
        )
        shifts[i] = [shift_y, shift_x]
        confidences[i] = confidence
        
        if verbose and (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n_particles} particles")
    
    if verbose:
        print(f"  {n_particles}/{n_particles} particles")
    
    shift_magnitudes = np.sqrt(shifts[:, 0]**2 + shifts[:, 1]**2)
    
    return {
        'cutoff_name': cutoff_name,
        'cutoff_value': cutoff,
        'shifts': shifts,
        'confidences': confidences,
        'shift_magnitudes': shift_magnitudes,
        'mean_confidence': np.mean(confidences)
    }


def compute_compact_stats(shifts, magnitudes):
    """Compute compact statistics"""
    return {
        'shift_y_range': (np.min(shifts[:, 0]), np.max(shifts[:, 0])),
        'shift_x_range': (np.min(shifts[:, 1]), np.max(shifts[:, 1])),
        'shift_y_median': np.median(shifts[:, 0]),
        'shift_x_median': np.median(shifts[:, 1]),
        'shift_y_95': np.percentile(np.abs(shifts[:, 0]), 95),
        'shift_x_95': np.percentile(np.abs(shifts[:, 1]), 95),
        'magnitude_median': np.median(magnitudes),
        'magnitude_95': np.percentile(magnitudes, 95),
        'magnitude_max': np.max(magnitudes)
    }


def suggest_shift_range(all_results, percentile=95):
    """
    Suggest representative shift range based on all analyses
    
    Uses the analysis with highest mean confidence
    """
    # Find best result (highest mean confidence)
    best_result = max(all_results, key=lambda x: x['mean_confidence'])
    
    shifts = best_result['shifts']
    
    # Use percentile to avoid outliers
    shift_y_abs = np.abs(shifts[:, 0])
    shift_x_abs = np.abs(shifts[:, 1])
    
    range_y = np.percentile(shift_y_abs, percentile)
    range_x = np.percentile(shift_x_abs, percentile)
    
    # Round up to nearest integer
    range_y_px = int(np.ceil(range_y))
    range_x_px = int(np.ceil(range_x))
    
    return {
        'best_cutoff': best_result['cutoff_name'],
        'range_y_px': range_y_px,
        'range_x_px': range_x_px,
        'range_y_exact': range_y,
        'range_x_exact': range_x,
        'percentile': percentile
    }


def print_compact_report(all_results, suggestion, pixel_size):
    """Print compact comparison report"""
    print(f"\n{'='*90}")
    print("SHIFT ANALYSIS SUMMARY")
    print(f"{'='*90}")
    print(f"Pixel size: {pixel_size:.3f} Å/pixel")
    print()
    
    # Header
    header = f"{'Filter':<15} {'Cutoff':<8} {'Conf':<6} {'Y_med':<12} {'X_med':<12} {'Y_95':<12} {'X_95':<12} {'Mag_95':<12}"
    print(header)
    print("-"*90)
    
    # Results for each cutoff
    for result in all_results:
        stats = compute_compact_stats(result['shifts'], result['shift_magnitudes'])
        
        y_med_px = stats['shift_y_median']
        x_med_px = stats['shift_x_median']
        y_95_px = stats['shift_y_95']
        x_95_px = stats['shift_x_95']
        mag_95_px = stats['magnitude_95']
        
        y_med_a = y_med_px * pixel_size
        x_med_a = x_med_px * pixel_size
        y_95_a = y_95_px * pixel_size
        x_95_a = x_95_px * pixel_size
        mag_95_a = mag_95_px * pixel_size
        
        print(f"{result['cutoff_name']:<15} "
              f"{result['cutoff_value']:<8.3f} "
              f"{result['mean_confidence']:<6.3f} "
              f"{y_med_px:>5.2f}({y_med_a:>4.1f}) "
              f"{x_med_px:>5.2f}({x_med_a:>4.1f}) "
              f"{y_95_px:>5.2f}({y_95_a:>4.1f}) "
              f"{x_95_px:>5.2f}({x_95_a:>4.1f}) "
              f"{mag_95_px:>5.2f}({mag_95_a:>4.1f})")
    
    print("-"*90)
    print("Values in pixels(Angstroms) | Conf: Mean confidence")
    print("Y_med/X_med: Median shifts | Y_95/X_95: 95th percentile | Mag_95: 95th percentile magnitude")
    
    # Detailed statistics
    print(f"\n{'='*90}")
    print("DETAILED STATISTICS")
    print(f"{'='*90}")
    
    for result in all_results:
        stats = compute_compact_stats(result['shifts'], result['shift_magnitudes'])
        
        print(f"\n{result['cutoff_name']} (cutoff={result['cutoff_value']:.3f}):")
        
        # Shift Y
        y_min_px, y_max_px = stats['shift_y_range']
        y_med_px = stats['shift_y_median']
        y_95_px = stats['shift_y_95']
        print(f"  Shift Y: range=[{y_min_px:>6.2f}, {y_max_px:>6.2f}] px = [{y_min_px*pixel_size:>6.1f}, {y_max_px*pixel_size:>6.1f}] Å")
        print(f"           median={y_med_px:>6.2f} px = {y_med_px*pixel_size:>6.1f} Å, "
              f"95%ile={y_95_px:>6.2f} px = {y_95_px*pixel_size:>6.1f} Å")
        
        # Shift X
        x_min_px, x_max_px = stats['shift_x_range']
        x_med_px = stats['shift_x_median']
        x_95_px = stats['shift_x_95']
        print(f"  Shift X: range=[{x_min_px:>6.2f}, {x_max_px:>6.2f}] px = [{x_min_px*pixel_size:>6.1f}, {x_max_px*pixel_size:>6.1f}] Å")
        print(f"           median={x_med_px:>6.2f} px = {x_med_px*pixel_size:>6.1f} Å, "
              f"95%ile={x_95_px:>6.2f} px = {x_95_px*pixel_size:>6.1f} Å")
        
        # Magnitude
        mag_med_px = stats['magnitude_median']
        mag_95_px = stats['magnitude_95']
        mag_max_px = stats['magnitude_max']
        print(f"  Magnitude: median={mag_med_px:>6.2f} px = {mag_med_px*pixel_size:>6.1f} Å")
        print(f"             95%ile={mag_95_px:>6.2f} px = {mag_95_px*pixel_size:>6.1f} Å, "
              f"max={mag_max_px:>6.2f} px = {mag_max_px*pixel_size:>6.1f} Å")
        print(f"  Mean confidence: {result['mean_confidence']:.3f}")
    
    # Suggestion
    print(f"\n{'='*90}")
    print("RECOMMENDED SHIFT RANGE")
    print(f"{'='*90}")
    print(f"Based on '{suggestion['best_cutoff']}' analysis (highest confidence)")
    print(f"Using {suggestion['percentile']}th percentile to exclude outliers:")
    
    range_y_px = suggestion['range_y_px']
    range_x_px = suggestion['range_x_px']
    range_y_a = range_y_px * pixel_size
    range_x_a = range_x_px * pixel_size
    
    print(f"\n  Shift range Y: ±{range_y_px} pixels = ±{range_y_a:.1f} Å")
    print(f"  Shift range X: ±{range_x_px} pixels = ±{range_x_a:.1f} Å")
    print(f"\nThis range covers {suggestion['percentile']}% of particles in the stack.")


def save_compact_results(all_results, suggestion, pixel_size, output_prefix):
    """Save compact results to file"""
    
    output_file = f"{output_prefix}_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*90 + "\n")
        f.write("CRYO-EM SHIFT ANALYSIS SUMMARY\n")
        f.write("="*90 + "\n\n")
        f.write(f"Pixel size: {pixel_size:.3f} Å/pixel\n\n")
        
        # Comparison table
        f.write("COMPARISON TABLE\n")
        f.write("-"*90 + "\n")
        f.write(f"{'Filter':<15} {'Cutoff':<8} {'Conf':<6} {'Y_med':<12} {'X_med':<12} {'Y_95':<12} {'X_95':<12} {'Mag_95':<12}\n")
        f.write("-"*90 + "\n")
        
        for result in all_results:
            stats = compute_compact_stats(result['shifts'], result['shift_magnitudes'])
            
            y_med_px = stats['shift_y_median']
            x_med_px = stats['shift_x_median']
            y_95_px = stats['shift_y_95']
            x_95_px = stats['shift_x_95']
            mag_95_px = stats['magnitude_95']
            
            y_med_a = y_med_px * pixel_size
            x_med_a = x_med_px * pixel_size
            y_95_a = y_95_px * pixel_size
            x_95_a = x_95_px * pixel_size
            mag_95_a = mag_95_px * pixel_size
            
            f.write(f"{result['cutoff_name']:<15} "
                   f"{result['cutoff_value']:<8.3f} "
                   f"{result['mean_confidence']:<6.3f} "
                   f"{y_med_px:>5.2f}({y_med_a:>4.1f}) "
                   f"{x_med_px:>5.2f}({x_med_a:>4.1f}) "
                   f"{y_95_px:>5.2f}({y_95_a:>4.1f}) "
                   f"{x_95_px:>5.2f}({x_95_a:>4.1f}) "
                   f"{mag_95_px:>5.2f}({mag_95_a:>4.1f})\n")
        
        f.write("\nValues in pixels(Angstroms)\n\n")
        
        # Detailed statistics
        f.write("="*90 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*90 + "\n\n")
        
        for result in all_results:
            stats = compute_compact_stats(result['shifts'], result['shift_magnitudes'])
            
            f.write(f"{result['cutoff_name']} (cutoff={result['cutoff_value']:.3f}):\n")
            
            # Shift Y
            y_min_px, y_max_px = stats['shift_y_range']
            y_med_px = stats['shift_y_median']
            y_95_px = stats['shift_y_95']
            f.write(f"  Shift Y: range=[{y_min_px:>6.2f}, {y_max_px:>6.2f}] px = [{y_min_px*pixel_size:>6.1f}, {y_max_px*pixel_size:>6.1f}] Å\n")
            f.write(f"           median={y_med_px:>6.2f} px = {y_med_px*pixel_size:>6.1f} Å, "
                   f"95%ile={y_95_px:>6.2f} px = {y_95_px*pixel_size:>6.1f} Å\n")
            
            # Shift X
            x_min_px, x_max_px = stats['shift_x_range']
            x_med_px = stats['shift_x_median']
            x_95_px = stats['shift_x_95']
            f.write(f"  Shift X: range=[{x_min_px:>6.2f}, {x_max_px:>6.2f}] px = [{x_min_px*pixel_size:>6.1f}, {x_max_px*pixel_size:>6.1f}] Å\n")
            f.write(f"           median={x_med_px:>6.2f} px = {x_med_px*pixel_size:>6.1f} Å, "
                   f"95%ile={x_95_px:>6.2f} px = {x_95_px*pixel_size:>6.1f} Å\n")
            
            # Magnitude
            mag_med_px = stats['magnitude_median']
            mag_95_px = stats['magnitude_95']
            mag_max_px = stats['magnitude_max']
            f.write(f"  Magnitude: median={mag_med_px:>6.2f} px = {mag_med_px*pixel_size:>6.1f} Å\n")
            f.write(f"             95%ile={mag_95_px:>6.2f} px = {mag_95_px*pixel_size:>6.1f} Å, "
                   f"max={mag_max_px:>6.2f} px = {mag_max_px*pixel_size:>6.1f} Å\n")
            f.write(f"  Mean confidence: {result['mean_confidence']:.3f}\n\n")
        
        # Recommendation
        f.write("="*90 + "\n")
        f.write("RECOMMENDED SHIFT RANGE\n")
        f.write("="*90 + "\n")
        f.write(f"Based on '{suggestion['best_cutoff']}' analysis (highest confidence)\n")
        f.write(f"Using {suggestion['percentile']}th percentile to exclude outliers:\n\n")
        
        range_y_px = suggestion['range_y_px']
        range_x_px = suggestion['range_x_px']
        range_y_a = range_y_px * pixel_size
        range_x_a = range_x_px * pixel_size
        
        f.write(f"  Shift range Y: ±{range_y_px} pixels = ±{range_y_a:.1f} Å\n")
        f.write(f"  Shift range X: ±{range_x_px} pixels = ±{range_x_a:.1f} Å\n\n")
        f.write(f"This range covers {suggestion['percentile']}% of particles in the stack.\n")
    
    print(f"\nSummary saved to '{output_file}'")
    
    # Save detailed data for best cutoff
    best_result = max(all_results, key=lambda x: x['mean_confidence'])
    data_file = f"{output_prefix}_shifts.txt"
    
    # Convert to Angstroms
    shifts_angstrom = best_result['shifts'] * pixel_size
    magnitudes_angstrom = best_result['shift_magnitudes'] * pixel_size
    
    np.savetxt(data_file,
               np.column_stack([best_result['shifts'], 
                               shifts_angstrom,
                               best_result['shift_magnitudes'],
                               magnitudes_angstrom,
                               best_result['confidences']]),
               header='shift_y_px shift_x_px shift_y_A shift_x_A magnitude_px magnitude_A confidence',
               fmt='%.6f')
    print(f"Shift data (best cutoff) saved to '{data_file}'")


def visualize_comparison(all_results, suggestion, pixel_size, save_path):
    """Create comparison visualization"""
    
    n_cutoffs = len(all_results)
    fig = plt.figure(figsize=(16, 4*n_cutoffs))
    
    for idx, result in enumerate(all_results):
        shifts = result['shifts']
        confidences = result['confidences']
        shift_magnitudes = result['shift_magnitudes']
        
        # Convert to Angstroms for plotting
        shifts_a = shifts * pixel_size
        shift_magnitudes_a = shift_magnitudes * pixel_size
        
        # 2D scatter
        ax1 = plt.subplot(n_cutoffs, 4, idx*4 + 1)
        scatter = ax1.scatter(shifts_a[:, 1], shifts_a[:, 0], 
                             c=confidences, cmap='viridis', 
                             s=5, alpha=0.6)
        ax1.set_xlabel('Shift X (Å)')
        ax1.set_ylabel('Shift Y (Å)')
        ax1.set_title(f'{result["cutoff_name"]} (cutoff={result["cutoff_value"]:.3f})')
        ax1.axhline(0, color='r', linestyle='--', alpha=0.3)
        ax1.axvline(0, color='r', linestyle='--', alpha=0.3)
        ax1.set_aspect('equal')
        plt.colorbar(scatter, ax=ax1, label='Confidence')
        
        # Shift Y histogram
        ax2 = plt.subplot(n_cutoffs, 4, idx*4 + 2)
        ax2.hist(shifts_a[:, 0], bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Shift Y (Å)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Mean conf: {result["mean_confidence"]:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Shift X histogram
        ax3 = plt.subplot(n_cutoffs, 4, idx*4 + 3)
        ax3.hist(shifts_a[:, 1], bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Shift X (Å)')
        ax3.set_ylabel('Count')
        stats = compute_compact_stats(shifts, shift_magnitudes)
        ax3.set_title(f'Mag 95%: {stats["magnitude_95"]*pixel_size:.1f} Å')
        ax3.grid(True, alpha=0.3)
        
        # Magnitude histogram
        ax4 = plt.subplot(n_cutoffs, 4, idx*4 + 4)
        ax4.hist(shift_magnitudes_a, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Shift Magnitude (Å)')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Max: {np.max(shift_magnitudes)*pixel_size:.1f} Å')
        ax4.grid(True, alpha=0.3)
    
    # Add recommendation text
    range_y_a = suggestion['range_y_px'] * pixel_size
    range_x_a = suggestion['range_x_px'] * pixel_size
    
    fig.text(0.5, 0.02, 
             f"RECOMMENDED: {suggestion['best_cutoff']} → "
             f"Shift range Y: ±{suggestion['range_y_px']} px (±{range_y_a:.1f} Å), "
             f"X: ±{suggestion['range_x_px']} px (±{range_x_a:.1f} Å) "
             f"({suggestion['percentile']}th percentile)",
             ha='center', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to '{save_path}'")


def read_pixel_size_from_mrc(mrc_path):
    """Read pixel size from MRC header"""
    try:
        import mrcfile
    except ImportError:
        print("ERROR: mrcfile package not found. Install with: pip install mrcfile")
        sys.exit(1)
    
    try:
        with mrcfile.open(mrc_path, mode='r') as mrc:
            # Get voxel size (returns x, y, z in Angstroms)
            voxel_size = mrc.voxel_size
            # Use x dimension (assuming square pixels)
            pixel_size = float(voxel_size.x)
            
            if pixel_size <= 0 or pixel_size > 100:
                print(f"WARNING: Unusual pixel size in MRC header: {pixel_size:.3f} Å")
                print("         Please verify or use --pixel-size to override")
            
            return pixel_size
    except Exception as e:
        print(f"ERROR: Failed to read pixel size from MRC header: {e}")
        return None


def analyze_shifts(mrc_path, pixel_size_override=None, verbose=True):
    """
    Main analysis function
    """
    # Import mrcfile
    try:
        import mrcfile
    except ImportError:
        print("ERROR: mrcfile package not found. Install with: pip install mrcfile")
        sys.exit(1)
    
    # Read MRC file
    if verbose:
        print(f"Reading {mrc_path}...")
    
    try:
        with mrcfile.open(mrc_path, mode='r') as mrc:
            particles = mrc.data
            
            # Get pixel size
            if pixel_size_override is not None:
                pixel_size = pixel_size_override
                if verbose:
                    print(f"Using provided pixel size: {pixel_size:.3f} Å/pixel")
            else:
                voxel_size = mrc.voxel_size
                pixel_size = float(voxel_size.x)
                if verbose:
                    print(f"Pixel size from MRC header: {pixel_size:.3f} Å/pixel")
                
                if pixel_size <= 0 or pixel_size > 100:
                    print(f"ERROR: Invalid pixel size: {pixel_size:.3f} Å/pixel")
                    print("       Please provide pixel size with --pixel-size option")
                    sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Failed to read MRC file: {e}")
        sys.exit(1)
    
    n_particles = particles.shape[0]
    img_shape = particles.shape[1:]
    
    if verbose:
        print(f"Loaded {n_particles} particles of shape {img_shape}")
    
    # Define three cutoff levels
    cutoffs = [
        (0.12, "Very Aggressive"),
        (0.20, "Aggressive"),
        (0.30, "Mild")
    ]
    
    # Analyze with each cutoff
    all_results = []
    for cutoff_value, cutoff_name in cutoffs:
        result = analyze_with_cutoff(particles, cutoff_value, cutoff_name, verbose)
        all_results.append(result)
    
    # Get suggestion
    suggestion = suggest_shift_range(all_results, percentile=95)
    
    return all_results, suggestion, pixel_size

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze translational shift distribution in cryo-EM particle stacks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (reads pixel size from MRC header)
  %(prog)s particles.mrc
  
  # Specify pixel size manually
  %(prog)s particles.mrc --pixel-size 1.5
  
  # Specify output prefix
  %(prog)s particles.mrc --output myanalysis
  
  # No plotting, quiet mode
  %(prog)s particles.mrc --no-plot --quiet

The script tests three filtering levels:
  - Very Aggressive (cutoff=0.12): Maximum noise reduction
  - Aggressive (cutoff=0.20): Balanced filtering
  - Mild (cutoff=0.30): Minimal filtering

Output files:
  <output>_summary.txt : Compact statistical summary and recommendations
  <output>_shifts.txt  : Per-particle shift data (best cutoff)
  <output>_plot.png    : Comparison visualization

All results are reported in both pixels and Angstroms.
        """
    )
    
    # Required arguments
    parser.add_argument('input', 
                       type=str,
                       help='Input MRC file containing particle stack')
    
    # Pixel size
    parser.add_argument('--pixel-size',
                       type=float,
                       default=None,
                       dest='pixel_size',
                       help='Pixel size in Angstroms (default: read from MRC header)')
    
    # Output parameters
    parser.add_argument('-o', '--output',
                       type=str,
                       default='shift_analysis',
                       help='Output prefix for result files (default: shift_analysis)')
    
    parser.add_argument('--no-plot',
                       action='store_true',
                       help='Do not generate visualization plots')
    
    # Verbosity
    parser.add_argument('-q', '--quiet',
                       action='store_true',
                       help='Suppress progress messages')
    
    parser.add_argument('-v', '--version',
                       action='version',
                       version=f'%(prog)s {__version__}')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Check input file exists
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Validate pixel size if provided
    if args.pixel_size is not None:
        if args.pixel_size <= 0:
            print(f"ERROR: Pixel size must be positive, got: {args.pixel_size}")
            sys.exit(1)
        if args.pixel_size > 100:
            print(f"WARNING: Unusually large pixel size: {args.pixel_size} Å")
    
    # Print header
    if not args.quiet:
        print("="*90)
        print("Cryo-EM Particle Shift Analyzer (Simplified)")
        print(f"Version {__version__}")
        print("="*90)
        print()
    
    try:
        # Analyze
        all_results, suggestion, pixel_size = analyze_shifts(
            args.input, 
            pixel_size_override=args.pixel_size,
            verbose=not args.quiet
        )
        
        # Print report
        if not args.quiet:
            print_compact_report(all_results, suggestion, pixel_size)
        
        # Save results
        if not args.quiet:
            print(f"\n{'='*90}")
            print("SAVING RESULTS")
            print(f"{'='*90}")
        
        save_compact_results(all_results, suggestion, pixel_size, args.output)
        
        # Generate visualization
        if not args.no_plot:
            if not args.quiet:
                print("\nGenerating visualization...")
            
            plot_file = f"{args.output}_plot.png"
            visualize_comparison(all_results, suggestion, pixel_size, plot_file)
        
        # Final message
        if not args.quiet:
            print(f"\n{'='*90}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*90}")
            
            # Print quick summary
            range_y_px = suggestion['range_y_px']
            range_x_px = suggestion['range_x_px']
            range_y_a = range_y_px * pixel_size
            range_x_a = range_x_px * pixel_size
            
            print(f"\nQuick Summary:")
            print(f"  Best filter: {suggestion['best_cutoff']}")
            print(f"  Recommended range (95th percentile):")
            print(f"    Y: ±{range_y_px} px (±{range_y_a:.1f} Å)")
            print(f"    X: ±{range_x_px} px (±{range_x_a:.1f} Å)")
            print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"\n\nERROR: Analysis failed: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
