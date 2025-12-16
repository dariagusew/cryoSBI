#!/usr/bin/env python3
"""
Extract CA atoms from multiple PDBs and save as PyTorch tensor.
Centers each structure at the origin.
"""

import MDAnalysis as mda
import torch
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')


def center_coordinates(coords):
    """
    Center coordinates at origin (subtract centroid).
    
    Parameters
    ----------
    coords : numpy array
        Coordinates of shape (Natoms, 3)
    
    Returns
    -------
    centered_coords : numpy array
        Centered coordinates
    """
    centroid = coords.mean(axis=0)
    return coords - centroid


def extract_ca_coordinates(pdb_files, output_file="ca_coords.pt"):
    """
    Extract CA atom coordinates from multiple PDB files.
    
    Parameters
    ----------
    pdb_files : list of str
        Input PDB filenames
    output_file : str
        Output PyTorch tensor file (.pt)
    
    Returns
    -------
    coords_tensor : torch.Tensor
        Tensor of shape [NPDBs, 3, Natoms] with centered coordinates
    """
    
    print("="*70)
    print("EXTRACT CA COORDINATES FROM MULTIPLE PDBs")
    print("="*70)
    
    n_pdbs = len(pdb_files)
    
    # First pass: check all structures have same number of CA atoms
    print("\nChecking structures...")
    print("-"*70)
    
    ca_counts = []
    for i, pdb_file in enumerate(pdb_files):
        u = mda.Universe(pdb_file)
        ca_atoms = u.select_atoms("protein and name CA")
        ca_counts.append(len(ca_atoms))
        print(f"  [{i+1}/{n_pdbs}] {pdb_file}: {len(ca_atoms)} CA atoms")
    
    if len(set(ca_counts)) != 1:
        raise ValueError(f"All structures must have the same number of CA atoms! Found: {ca_counts}")
    
    n_ca = ca_counts[0]
    
    print(f"\n{'='*70}")
    print(f"All structures have {n_ca} CA atoms ✓")
    print(f"{'='*70}")
    
    # Second pass: extract and center coordinates
    print("\nExtracting and centering coordinates...")
    print("-"*70)
    
    # Initialize array to hold all coordinates
    # Shape: [NPDBs, Natoms, 3]
    all_coords = np.zeros((n_pdbs, n_ca, 3))
    
    for i, pdb_file in enumerate(pdb_files):
        u = mda.Universe(pdb_file)
        ca_atoms = u.select_atoms("protein and name CA")
        
        # Get coordinates
        coords = ca_atoms.positions  # Shape: (Natoms, 3)
        
        # Center at origin
        centered_coords = center_coordinates(coords)
        
        all_coords[i] = centered_coords
        
        centroid_before = coords.mean(axis=0)
        centroid_after = centered_coords.mean(axis=0)
        
        print(f"  [{i+1}/{n_pdbs}] {pdb_file}")
        print(f"       Original centroid: [{centroid_before[0]:.2f}, {centroid_before[1]:.2f}, {centroid_before[2]:.2f}]")
        print(f"       After centering:   [{centroid_after[0]:.2e}, {centroid_after[1]:.2e}, {centroid_after[2]:.2e}]")
    
    # Convert to torch tensor and transpose to requested shape [NPDBs, 3, Natoms]
    coords_tensor = torch.from_numpy(all_coords).float()  # [NPDBs, Natoms, 3]
    coords_tensor = coords_tensor.permute(0, 2, 1)  # [NPDBs, 3, Natoms]
    
    print(f"\n{'='*70}")
    print(f"Tensor shape: {list(coords_tensor.shape)} [NPDBs, 3, Natoms]")
    print(f"  NPDBs:  {coords_tensor.shape[0]}")
    print(f"  Coords: {coords_tensor.shape[1]} (x, y, z)")
    print(f"  Natoms: {coords_tensor.shape[2]}")
    print(f"{'='*70}")
    
    # Save to file
    print(f"\nSaving to {output_file}...")
    torch.save(coords_tensor, output_file)
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
    print(f"Saved: {output_file}")
    print(f"Shape: {list(coords_tensor.shape)}")
    print(f"\nTo load in PyTorch:")
    print(f"  coords = torch.load('{output_file}')")
    print("="*70)
    
    return coords_tensor


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("="*70)
        print("Extract CA Coordinates to PyTorch Tensor")
        print("="*70)
        print("\nUsage:")
        print("  python extract_ca_coords.py <pdb1> <pdb2> [pdb3 ...] [options]")
        print("\nArguments:")
        print("  pdb1, pdb2, ... : Input PDB files (minimum 2)")
        print("\nOptions:")
        print("  --output, -o : Output .pt file (default: ca_coords.pt)")
        print("\nFeatures:")
        print("  • Extracts CA atoms only")
        print("  • Centers each structure at origin")
        print("  • Saves as PyTorch tensor of shape [NPDBs, 3, Natoms]")
        print("  • All structures must have same number of CA atoms")
        print("\nExamples:")
        print("  python extract_ca_coords.py struct1.pdb struct2.pdb")
        print("  python extract_ca_coords.py *.pdb -o coords.pt")
        print("  python extract_ca_coords.py conf1.pdb conf2.pdb conf3.pdb")
        print("="*70)
        sys.exit(1)
    
    # Parse arguments
    pdb_files = []
    output_file = "ca_coords.pt"
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ['--output', '-o']:
            if i + 1 >= len(sys.argv):
                print("Error: --output requires a value")
                sys.exit(1)
            output_file = sys.argv[i + 1]
            i += 2
        else:
            # Assume it's a PDB file
            pdb_files.append(arg)
            i += 1
    
    if len(pdb_files) < 1:
        print("Error: At least 1 PDB file is required")
        sys.exit(1)
    
    print(f"\nInput files: {len(pdb_files)}")
    for i, pdb in enumerate(pdb_files):
        print(f"  {i+1}. {pdb}")
    print(f"\nOutput file: {output_file}")
    
    try:
        extract_ca_coordinates(pdb_files, output_file)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
