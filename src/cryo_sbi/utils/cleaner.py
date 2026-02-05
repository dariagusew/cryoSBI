#!/usr/bin/env python3
"""
Detect and visualize particles with edge artifacts (vertical/horizontal bars) in MRC files.
"""

import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


def detect_edge_artifacts(particle, threshold=0.01):
    """
    Detect if a particle has vertical or horizontal bars on its edges.
    First checks if stripe exists, then calculates its thickness.
    
    Parameters:
    -----------
    particle : numpy.ndarray
        2D array representing the particle image
    threshold : float
        Threshold for pixel difference (lower = more strict)
    
    Returns:
    --------
    dict : {
        'is_broken': bool,
        'edges': list of str (e.g., ['top', 'left']),
        'max_offset': int,
        'edge_details': dict with offset for each edge
    }
    """
    result = {
        'is_broken': False,
        'edges': [],
        'max_offset': 0,
        'edge_details': {}
    }
    
    if particle.ndim != 2:
        return result
    
    h, w = particle.shape
    min_offset = 4
    
    # Need minimum size to perform check
    if h <= min_offset * 2 or w <= min_offset * 2:
        return result
    
    # Sample every 50 pixels for initial detection
    sample_step = min(50, max(h, w) // 4)
    
    # Define edges: (get_coords_func, sample_range, test_points_for_thickness)
    edges_to_test = {
        'top': {
            'coords': lambda offset, pos: (0, pos, offset, pos),
            'range': range(0, w, sample_step),
            'test_points': [w // 4, w // 2, 3 * w // 4],
            'thickness_coords': lambda offset, point: (offset, point),
            'max_thickness': h // 2
        },
        'bottom': {
            'coords': lambda offset, pos: (-1, pos, -(offset+1), pos),
            'range': range(0, w, sample_step),
            'test_points': [w // 4, w // 2, 3 * w // 4],
            'thickness_coords': lambda offset, point: (-(offset+1), point),
            'max_thickness': h // 2
        },
        'left': {
            'coords': lambda offset, pos: (pos, 0, pos, offset),
            'range': range(0, h, sample_step),
            'test_points': [h // 4, h // 2, 3 * h // 4],
            'thickness_coords': lambda offset, point: (point, offset),
            'max_thickness': w // 2
        },
        'right': {
            'coords': lambda offset, pos: (pos, -1, pos, -(offset+1)),
            'range': range(0, h, sample_step),
            'test_points': [h // 4, h // 2, 3 * h // 4],
            'thickness_coords': lambda offset, point: (point, -(offset+1)),
            'max_thickness': w // 2
        }
    }
    
    for edge_name, edge_config in edges_to_test.items():
        # Step 1: Quick check if stripe exists at min_offset
        has_stripe = True
        for pos in edge_config['range']:
            try:
                r1, c1, r2, c2 = edge_config['coords'](min_offset, pos)
                if abs(float(particle[r1, c1]) - float(particle[r2, c2])) > threshold:
                    has_stripe = False
                    break
            except IndexError:
                has_stripe = False
                break
        
        if not has_stripe:
            continue
        
        # Step 2: Measure thickness on a few test points
        max_depth = 0
        
        for test_point in edge_config['test_points']:
            try:
                # Get reference value at edge
                r_ref, c_ref = edge_config['thickness_coords'](0, test_point)
                reference_val = float(particle[r_ref, c_ref])
                
                # Increment offset until pixels are different
                depth = 0
                for offset in range(1, min(edge_config['max_thickness'], 50)):
                    r, c = edge_config['thickness_coords'](offset, test_point)
                    if abs(float(particle[r, c]) - reference_val) <= threshold:
                        depth = offset
                    else:
                        break
                
                max_depth = max(max_depth, depth)
            except IndexError:
                continue
        
        # If thickness is significant, mark as artifact
        if max_depth >= min_offset:
            result['is_broken'] = True
            result['edges'].append(edge_name)
            result['edge_details'][edge_name] = max_depth
            result['max_offset'] = max(result['max_offset'], max_depth)
    
    return result


def read_mrc_particles(mrcfile_path):
    """
    Read particles from MRC file (memory-mapped, fastest).
    
    Parameters:
    -----------
    mrcfile_path : str
        Path to the MRC file
    
    Returns:
    --------
    numpy.ndarray : Memory-mapped array of particles
    """
    # Open with memory mapping - no data loaded into RAM
    mrc = mrcfile.mmap(mrcfile_path, mode='r', permissive=True)
    return mrc.data


def read_star_file(star_path):
    """
    Read RELION/cryoSPARC STAR file.
    
    Returns:
    --------
    dict : {
        'headers': list of column names,
        'data': list of lists (rows),
        'pre_header': list of lines before data block,
        'image_name_col': int (index of _rlnImageName column, -1 if not found)
    }
    """
    with open(star_path, 'r') as f:
        lines = f.readlines()
    
    headers = []
    data = []
    pre_header = []
    in_data_block = False
    in_loop = False
    image_name_col = -1
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines before data
        if not stripped and not in_data_block:
            pre_header.append(line)
            continue
        
        # Check for data_ block
        if stripped.startswith('data_'):
            pre_header.append(line)
            in_data_block = True
            continue
        
        # Check for loop_
        if stripped.startswith('loop_'):
            pre_header.append(line)
            in_loop = True
            continue
        
        # Read column headers
        if in_loop and stripped.startswith('_'):
            headers.append(stripped.split()[0])
            pre_header.append(line)
            
            # Check if this is the image name column
            if '_rlnImageName' in stripped or '_rlnImageName' == stripped.split()[0]:
                image_name_col = len(headers) - 1
            
            continue
        
        # Read data rows
        if in_loop and headers and not stripped.startswith('_') and not stripped.startswith('loop_'):
            if stripped:  # Non-empty line
                data.append(stripped.split())
        elif not in_loop and not stripped.startswith('_'):
            pre_header.append(line)
    
    return {
        'headers': headers,
        'data': data,
        'pre_header': pre_header,
        'image_name_col': image_name_col
    }


def write_star_file(star_path, star_dict):
    """
    Write STAR file.
    """
    with open(star_path, 'w') as f:
        # Write pre-header
        for line in star_dict['pre_header']:
            f.write(line)
        
        # Write data
        for row in star_dict['data']:
            f.write(' '.join(row) + '\n')

def create_clean_files(particles_data, star_dict, clean_indices, output_mrc, output_star, original_mrc_name):
    """
    Create cleaned MRC and STAR files - direct file writing (no concatenation).
    """
    print(f"\nCreating cleaned files...")
    print(f"  Clean particles: {len(clean_indices)}/{particles_data.shape[0]}")
    
    if particles_data.ndim != 3:
        raise ValueError("Expected 3D particle stack")
    
    particle_shape = particles_data[0].shape
    n_total = len(clean_indices)
    
    print(f"  Writing clean particles to {output_mrc}...")
    
    # Create a temporary MRC to get proper header
    temp_particle = np.array(particles_data[clean_indices[0]], dtype=np.float32)[np.newaxis, ...]
    temp_file = '_temp_header.mrc'
    with mrcfile.new(temp_file, overwrite=True) as mrc:
        mrc.set_data(temp_particle)
    
    # Read the header
    with mrcfile.open(temp_file, mode='r') as mrc:
        header = mrc.header.copy()
    
    import os
    os.remove(temp_file)
    del temp_particle
    
    # Update header for full stack
    header['nz'] = n_total
    header['mz'] = n_total
    
    # Write MRC file manually
    with open(output_mrc, 'wb') as f:
        # Write header (1024 bytes)
        f.write(header.tobytes())
        
        # Write particles one by one
        for new_idx, old_idx in enumerate(clean_indices):
            particle = np.array(particles_data[old_idx], dtype=np.float32)
            f.write(particle.tobytes())
            
            if (new_idx + 1) % 1000 == 0:
                print(f"    Written {new_idx + 1}/{n_total} particles...")
        
        # Write extended header if needed (usually empty for stacks)
        # and update with proper values
    
    print(f"  Saved: {output_mrc}")
    
    # Create new STAR file
    if star_dict and star_dict['image_name_col'] >= 0:
        print(f"  Creating {output_star}...")
        new_star_dict = {
            'headers': star_dict['headers'],
            'data': [],
            'pre_header': star_dict['pre_header'],
            'image_name_col': star_dict['image_name_col']
        }
        
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(clean_indices)}
        output_mrc_basename = os.path.basename(output_mrc)
        
        for old_idx in clean_indices:
            if old_idx < len(star_dict['data']):
                row = star_dict['data'][old_idx].copy()
                img_col = star_dict['image_name_col']
                old_ref = row[img_col]
                
                if '@' in old_ref:
                    new_idx = old_to_new_idx[old_idx]
                    new_ref = f"{new_idx + 1:06d}@{output_mrc_basename}"
                    row[img_col] = new_ref
                
                new_star_dict['data'].append(row)
        
        write_star_file(output_star, new_star_dict)
        print(f"  Saved: {output_star}")

def create_pdf_report(broken_particles, broken_indices, broken_info, output_pdf='broken_particles.pdf'):
    """
    Create a PDF report - PIL optimized version (simple).
    """
    from PIL import Image, ImageDraw, ImageFont
    import gc
    
    particles_per_page = 20
    n_rows, n_cols = 5, 4
    cell_size = 180
    margin = 10
    text_height = 35
    
    page_width = n_cols * (cell_size + margin) + margin
    page_height = n_rows * (cell_size + text_height + margin) + margin + 50
    
    n_particles = len(broken_particles)
    n_pages = (n_particles + particles_per_page - 1) // particles_per_page
    
    pdf_pages = []
    
    try:
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font_text = ImageFont.load_default()
    
    for page in range(n_pages):
        if page % 10 == 0:
            print(f"  Creating page {page + 1}/{n_pages}...")
        
        page_img = Image.new('L', (page_width, page_height), 255)  # Grayscale
        draw = ImageDraw.Draw(page_img)
        
        title = f'Page {page + 1}/{n_pages}'
        draw.text((page_width // 2, 25), title, fill=0, font=font_text, anchor='mm')
        
        start_idx = page * particles_per_page
        end_idx = min(start_idx + particles_per_page, n_particles)
        
        for i in range(end_idx - start_idx):
            particle_idx = start_idx + i
            row, col = i // n_cols, i % n_cols
            x = margin + col * (cell_size + margin)
            y = 50 + margin + row * (cell_size + text_height + margin)
            
            particle = broken_particles[particle_idx]
            info = broken_info[particle_idx]
            
            # Normalize
            p_min, p_max = np.percentile(particle, [2, 98])
            particle_norm = np.clip((particle - p_min) / (p_max - p_min + 1e-10), 0, 1)
            particle_8bit = (particle_norm * 255).astype(np.uint8)
            
            # Resize and paste
            p_img = Image.fromarray(particle_8bit, mode='L')
            p_img = p_img.resize((cell_size, cell_size), Image.BILINEAR)
            page_img.paste(p_img, (x, y))
            
            # Text
            text_y = y + cell_size + 3
            text = f"#{broken_indices[particle_idx]} {','.join(info['edges'][:2])} {info['max_offset']} px"
            draw.text((x + cell_size // 2, text_y), text, fill=0, font=font_text, anchor='mt')
        
        pdf_pages.append(page_img)
        
        # Free memory every 50 pages
        if (page + 1) % 50 == 0:
            gc.collect()
    
    # Save all pages
    print("  Saving PDF...")
    pdf_pages[0].save(output_pdf, save_all=True, append_images=pdf_pages[1:], 
                      resolution=100.0, quality=90, optimize=True)
    
    print(f"PDF report saved to: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect particles with edge artifacts (vertical/horizontal bars) in MRC files.'
    )
    parser.add_argument('--mrcfile', required=True, help='Path to input MRC file')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Detection threshold (default: 0.01, lower = more sensitive)')
    parser.add_argument('--plots', action='store_true',
                       help='Create PDF plots of broken particles')
    parser.add_argument('--output', default='broken_particles.pdf',
                       help='Output PDF filename (default: broken_particles.pdf)')
    parser.add_argument('--starfile', default=None,
                       help='Optional STAR file - creates fixed.mrc and fixed.star with clean particles only')
    
    args = parser.parse_args()
    
    print(f"Reading MRC file: {args.mrcfile}")
    particles_data = read_mrc_particles(args.mrcfile)
    
    # Handle both 2D (single particle) and 3D (particle stack) data
    if particles_data.ndim == 2:
        particles = [particles_data]
    elif particles_data.ndim == 3:
        particles = [particles_data[i] for i in range(particles_data.shape[0])]
    else:
        raise ValueError(f"Unexpected data dimensions: {particles_data.ndim}")
    
    # Read STAR file if provided
    star_dict = None
    if args.starfile:
        print(f"Reading STAR file: {args.starfile}")
        star_dict = read_star_file(args.starfile)
        print(f"  Found {len(star_dict['data'])} entries")
        if star_dict['image_name_col'] >= 0:
            print(f"  Image name column: {star_dict['headers'][star_dict['image_name_col']]}")
    
    print(f"Total particles: {len(particles)}")
    print(f"Detection threshold: {args.threshold}")
    print("Analyzing particles...")
    
    broken_particles = []
    broken_indices = []
    broken_info = []
    clean_indices = []
    
    for idx, particle in enumerate(particles):
        detection_result = detect_edge_artifacts(particle, threshold=args.threshold)
        
        if detection_result['is_broken']:
            broken_particles.append(particle)
            broken_indices.append(idx)
            broken_info.append(detection_result)
        else:
            clean_indices.append(idx)
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(particles)} particles...")
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Number of broken particles detected: {len(broken_particles)}")
    print(f"Number of clean particles: {len(clean_indices)}")
    print(f"Broken percentage: {100 * len(broken_particles) / len(particles):.2f}%")
    
    # Print statistics about edges
    if len(broken_info) > 0:
        edge_counts = {}
        max_offsets = []
        
        for info in broken_info:
            for edge in info['edges']:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
            max_offsets.append(info['max_offset'])
        
        print(f"\nEdge statistics:")
        for edge, count in sorted(edge_counts.items()):
            print(f"  {edge}: {count} ({100*count/len(broken_info):.1f}%)")
        print(f"\nOffset statistics:")
        print(f"  Average max offset: {np.mean(max_offsets):.1f}")
        print(f"  Max offset found: {np.max(max_offsets)}")
    
    print(f"{'='*60}\n")
    
    # Create cleaned files if STAR file provided
    if args.starfile and len(clean_indices) > 0:
        output_mrc = 'fixed.mrcs'
        output_star = 'fixed.star'
        create_clean_files(particles_data, star_dict, clean_indices, 
                          output_mrc, output_star, args.mrcfile)
    
    # Create PDF report if requested
    if args.plots and len(broken_particles) > 0:
        print("\nCreating PDF report...")
        create_pdf_report(broken_particles, broken_indices, broken_info, args.output)
    elif args.plots and len(broken_particles) == 0:
        print("No broken particles found, skipping PDF creation.")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
