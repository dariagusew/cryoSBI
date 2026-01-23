#!/usr/bin/env python3
"""
Find common residues across multiple PDBs via sequence alignment.
Write out PDBs with only common residues (no structural alignment).
Handles alternative locations by keeping only altloc A (or first altloc).
"""

import MDAnalysis as mda
from Bio import pairwise2
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# Three-letter to one-letter amino acid code mapping
AA_CODE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M',  # Selenomethionine
    'UNK': 'X'   # Unknown
}

def three_to_one(resname):
    """Convert three-letter amino acid code to one-letter."""
    return AA_CODE.get(resname.upper(), 'X')

def filter_altlocs(atoms):
    """
    Filter atoms to keep only altloc A or blank.
    If altloc A doesn't exist, keep blank.
    If neither exist, keep first altloc alphabetically.
    """
    if not hasattr(atoms, 'altLocs'):
        # No altloc information, return all atoms
        return atoms
    
    # Check what altlocs are present
    altlocs = set(atoms.altLocs)
    
    # If no alternatives or all blank, return all
    if len(altlocs) == 0 or (len(altlocs) == 1 and '' in altlocs):
        return atoms
    
    # Determine which altloc to keep
    altlocs_non_blank = altlocs - {''}
    
    if not altlocs_non_blank:
        # Only blank altlocs
        return atoms
    
    if 'A' in altlocs:
        selected_altloc = 'A'
    else:
        # Take first alphabetically
        selected_altloc = sorted(altlocs_non_blank)[0]
    
    # Filter atoms: keep blank OR selected altloc
    indices = [i for i, atom in enumerate(atoms) 
               if atom.altLoc == '' or atom.altLoc == selected_altloc]
    
    if len(indices) == 0:
        return atoms  # Fallback: return all if filtering failed
    
    return atoms[indices]

def get_chain_info(universe):
    """
    Extract chain information from universe.
    Returns dict: {chain_id: {'residues': [...], 'sequence': str}}
    """
    protein = universe.select_atoms("protein")
    chains = {}
    
    for chain_id in np.unique(protein.segments.segids):
        chain_atoms = universe.select_atoms(f"protein and segid {chain_id}")
        residues = []
        sequence = []
        seen_resids = set()
        
        for residue in chain_atoms.residues:
            # Use resid as unique identifier (handle altlocs/duplicates)
            if residue.resid in seen_resids:
                continue
            seen_resids.add(residue.resid)
            
            residues.append({
                'resid': residue.resid,
                'resnum': residue.resnum,
                'resname': residue.resname,
                'segid': chain_id,
                'n_atoms': len(residue.atoms)
            })
            sequence.append(three_to_one(residue.resname))
        
        chains[chain_id] = {
            'residues': residues,
            'sequence': ''.join(sequence),
            'n_residues': len(residues)
        }
    
    return chains

def match_chains(reference_chains, target_chains, identity_threshold=0.5):
    """
    Match chains from target structure to reference structure based on sequence similarity.
    Returns dict: {ref_chain_id: target_chain_id}
    """
    matches = {}
    used_target_chains = set()
    
    for ref_cid, ref_cdata in reference_chains.items():
        seq_ref = ref_cdata['sequence']
        best_match = None
        best_score = 0
        
        for tgt_cid, tgt_cdata in target_chains.items():
            if tgt_cid in used_target_chains:
                continue
                
            seq_tgt = tgt_cdata['sequence']
            
            # Alignment to find best match
            alignments = pairwise2.align.globalxx(seq_ref, seq_tgt)
            if alignments:
                aln = alignments[0]
                # Calculate identity
                matches_count = sum(1 for a, b in zip(aln.seqA, aln.seqB) if a == b and a != '-')
                identity = matches_count / max(len(seq_ref), len(seq_tgt))
                
                if identity > best_score:
                    best_score = identity
                    best_match = tgt_cid
        
        if best_match and best_score >= identity_threshold:
            matches[ref_cid] = best_match
            used_target_chains.add(best_match)
            print(f"    Reference chain {ref_cid} -> Target chain {best_match} (identity: {best_score:.2%})")
        else:
            print(f"    Reference chain {ref_cid} -> No match (best score: {best_score:.2%})")
    
    return matches

def align_chain_sequences_with_indices(chain1_data, chain2_data):
    """
    Align two chain sequences and return mapping of common residues with indices.
    Returns: (common_residues1, common_residues2, indices1, indices2)
    where indices are the positions in the original residue lists.
    """
    seq1 = chain1_data['sequence']
    seq2 = chain2_data['sequence']
    residues1 = chain1_data['residues']
    residues2 = chain2_data['residues']
    
    # Perform global alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)
    
    if not alignments:
        return [], [], [], []
    
    aln = alignments[0]
    aln_seq1 = aln.seqA
    aln_seq2 = aln.seqB
    
    # Map aligned positions to original residue indices
    common_res1 = []
    common_res2 = []
    indices1 = []
    indices2 = []
    
    idx1 = 0
    idx2 = 0
    
    for pos in range(len(aln_seq1)):
        aa1 = aln_seq1[pos]
        aa2 = aln_seq2[pos]
        
        if aa1 != '-' and aa2 != '-' and aa1 == aa2:
            common_res1.append(residues1[idx1])
            common_res2.append(residues2[idx2])
            indices1.append(idx1)
            indices2.append(idx2)
        
        if aa1 != '-':
            idx1 += 1
        if aa2 != '-':
            idx2 += 1
    
    return common_res1, common_res2, indices1, indices2

def build_selection_from_residues(common_residues_dict):
    """
    Build MDAnalysis selection string from dictionary of common residues.
    """
    selections = []
    
    for chain_id, residues in common_residues_dict.items():
        if not residues:
            continue
        
        resids = [res['resid'] for res in residues]
        
        if resids:
            resid_str = ' '.join(map(str, resids))
            selections.append(f"(segid {chain_id} and resid {resid_str})")
    
    if not selections:
        raise ValueError("No common residues found!")
    
    return "protein and (" + " or ".join(selections) + ")"

def generate_output_filename(input_filename, suffix="_common"):
    """
    Generate output filename by adding suffix before extension.
    Example: input.pdb -> input_common.pdb
    """
    base, ext = os.path.splitext(input_filename)
    return f"{base}{suffix}{ext}"

def extract_common_residues_multi(pdb_files, output_files=None, identity_threshold=0.5):
    """
    Extract common residues from multiple PDBs based on sequence alignment.
    Uses first PDB as reference structure.
    NO structural alignment - just write out common residues.
    
    Parameters
    ----------
    pdb_files : list of str
        Input PDB filenames
    output_files : list of str, optional
        Output PDB filenames (default: input_name_common.pdb)
    identity_threshold : float
        Minimum sequence identity for chain matching (0-1)
    """
    
    n_pdbs = len(pdb_files)
    
    if output_files is None:
        output_files = [generate_output_filename(pdb) for pdb in pdb_files]
    
    if len(output_files) != n_pdbs:
        raise ValueError(f"Number of output files ({len(output_files)}) must match number of input files ({n_pdbs})")
    
    print("="*70)
    print(f"EXTRACT COMMON RESIDUES ACROSS {n_pdbs} STRUCTURES")
    print("="*70)
    
    # Load all structures
    print("\nLoading structures...")
    print("-"*70)
    universes = []
    all_chains = []
    
    for i, pdb_file in enumerate(pdb_files):
        print(f"  [{i+1}/{n_pdbs}] {pdb_file}...")
        u = mda.Universe(pdb_file)
        universes.append(u)
        chains = get_chain_info(u)
        all_chains.append(chains)
        
        n_chains = len(chains)
        total_residues = sum(c['n_residues'] for c in chains.values())
        print(f"       {n_chains} chain(s), {total_residues} residues")
    
    # Use first structure as reference
    reference_chains = all_chains[0]
    
    print("\n" + "-"*70)
    print("Reference structure (structure 1):")
    print("-"*70)
    for cid, cdata in reference_chains.items():
        print(f"  Chain {cid}: {cdata['n_residues']} residues")
        seq_preview = cdata['sequence'][:60]
        if len(cdata['sequence']) > 60:
            seq_preview += '...'
        print(f"    Sequence: {seq_preview}")
    
    # Match all structures to reference
    print("\n" + "-"*70)
    print("Matching chains to reference:")
    print("-"*70)
    
    chain_matches = []  # List of dicts: {ref_chain_id: target_chain_id}
    
    for i in range(1, n_pdbs):
        print(f"\n  Structure {i+1} -> Reference:")
        matches = match_chains(reference_chains, all_chains[i], identity_threshold)
        chain_matches.append(matches)
        
        if not matches:
            raise ValueError(f"No matching chains found between reference and structure {i+1}!")
    
    # Find common residues across all structures
    print("\n" + "-"*70)
    print("Finding common residues across all structures:")
    print("-"*70)
    
    # Store common residues for each structure
    # common_residues_all[structure_idx][chain_id] = [residues]
    common_residues_all = [{} for _ in range(n_pdbs)]
    
    # Process each reference chain
    for ref_cid in reference_chains.keys():
        # Check if this chain exists in all structures
        chain_exists_in_all = all(
            ref_cid in matches for matches in chain_matches
        )
        
        if not chain_exists_in_all:
            print(f"\n  Chain {ref_cid}: Not present in all structures, skipping")
            continue
        
        print(f"\n  Chain {ref_cid}:")
        
        # Step 1: Align all structures with reference
        # alignments[i] = (ref_indices, target_indices) for structure i
        alignments = []
        
        ref_residues = reference_chains[ref_cid]['residues']
        
        for i in range(1, n_pdbs):
            target_cid = chain_matches[i-1][ref_cid]
            _, _, ref_idx, tgt_idx = align_chain_sequences_with_indices(
                reference_chains[ref_cid],
                all_chains[i][target_cid]
            )
            alignments.append((ref_idx, tgt_idx))
        
        # Step 2: Find intersection - which reference indices are present in ALL alignments
        # Start with all possible reference indices
        if alignments:
            common_ref_indices = set(alignments[0][0])
            for ref_idx, _ in alignments[1:]:
                common_ref_indices &= set(ref_idx)
            common_ref_indices = sorted(common_ref_indices)
        else:
            common_ref_indices = list(range(len(ref_residues)))
        
        # Step 3: For each structure, extract the residues at positions that map to common_ref_indices
        n_common = len(common_ref_indices)
        
        if n_common > 0:
            # Reference structure
            common_residues_all[0][ref_cid] = [ref_residues[i] for i in common_ref_indices]
            
            # Other structures
            for i in range(1, n_pdbs):
                target_cid = chain_matches[i-1][ref_cid]
                ref_idx, tgt_idx = alignments[i-1]
                
                # Build mapping from ref index to target index
                ref_to_tgt = {r: t for r, t in zip(ref_idx, tgt_idx)}
                
                # Extract target residues that correspond to common_ref_indices
                tgt_residues = all_chains[i][target_cid]['residues']
                common_tgt_residues = [tgt_residues[ref_to_tgt[r]] for r in common_ref_indices]
                
                common_residues_all[i][target_cid] = common_tgt_residues
            
            print(f"    Found {n_common} common residues across all structures")
            for i in range(n_pdbs):
                chain_id = ref_cid if i == 0 else chain_matches[i-1][ref_cid]
                atoms = sum(r['n_atoms'] for r in common_residues_all[i][chain_id])
                print(f"      Structure {i+1}: {atoms} atoms")
        else:
            print(f"    No common residues found")
    
    # Calculate totals
    total_common_residues = sum(len(residues) 
                                for residues in common_residues_all[0].values())
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Total common residues: {total_common_residues}")
    for i in range(n_pdbs):
        n_residues = sum(len(residues) for residues in common_residues_all[i].values())
        total_atoms = sum(r['n_atoms'] 
                         for chain_residues in common_residues_all[i].values() 
                         for r in chain_residues)
        print(f"  Structure {i+1}: {n_residues} residues, {total_atoms} atoms")
    print(f"{'='*70}")
    
    if total_common_residues == 0:
        raise ValueError("No common residues found across all structures!")
    
    # Verify all structures have same number of residues
    residue_counts = [sum(len(residues) for residues in common_residues_all[i].values()) 
                      for i in range(n_pdbs)]
    if len(set(residue_counts)) != 1:
        raise ValueError(f"Internal error: Residue counts differ: {residue_counts}")
    
    # Build selections and write files
    print("\n" + "-"*70)
    print("Writing output files (filtering alternative locations):")
    print("-"*70)
    
    for i in range(n_pdbs):
        selection = build_selection_from_residues(common_residues_all[i])
        common_atoms = universes[i].select_atoms(selection)
        
        print(f"\n  [{i+1}/{n_pdbs}] {pdb_files[i]} -> {output_files[i]}")
        print(f"       Atoms before altloc filtering: {len(common_atoms)}")
        
        # Filter alternative locations
        filtered_atoms = filter_altlocs(common_atoms)
        
        n_residues = sum(len(residues) for residues in common_residues_all[i].values())
        
        print(f"       Atoms after altloc filtering: {len(filtered_atoms)}")
        print(f"       Residues: {n_residues}")
        
        if len(filtered_atoms) == 0:
            raise ValueError(f"No atoms remaining after altloc filtering for structure {i+1}!")
        
        filtered_atoms.write(output_files[i])
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
    print(f"Output files (original coordinates, not aligned):")
    for out_file in output_files:
        print(f"  {out_file}")
    print(f"\nCommon residues: {total_common_residues} (same in all structures)")
    print(f"Alternative locations: Kept only altloc A (or blank)")
    print("="*70)
    
    return universes, common_residues_all


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("="*70)
        print("Extract Common Residues from Multiple PDBs")
        print("="*70)
        print("\nUsage:")
        print("  python prepare_PDBs.py <pdb1> <pdb2> [pdb3 ...] [options]")
        print("\nArguments:")
        print("  pdb1, pdb2, ... : Input PDB files (minimum 2)")
        print("\nOptions:")
        print("  --suffix, -s    : Output file suffix (default: _common)")
        print("  --threshold, -t : Min sequence identity (default: 0.5)")
        print("\nFeatures:")
        print("  • Finds residues common to ALL input structures")
        print("  • Sequence-based residue matching (first PDB is reference)")
        print("  • Handles multiple chains")
        print("  • Removes alternative locations (keeps only altloc A)")
        print("  • NO structural alignment (preserves original coordinates)")
        print("\nOutput:")
        print("  Files are named as: <input>_common.pdb")
        print("  Example: open.pdb -> open_common.pdb")
        print("\nExamples:")
        print("  python prepare_PDBs.py open.pdb closed.pdb")
        print("  python prepare_PDBs.py conf1.pdb conf2.pdb conf3.pdb")
        print("  python prepare_PDBs.py *.pdb --threshold 0.8")
        print("  python prepare_PDBs.py file1.pdb file2.pdb --suffix _filtered")
        print("="*70)
        sys.exit(1)
    
    # Parse arguments
    pdb_files = []
    output_suffix = "_common"
    identity_threshold = 0.5
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ['--suffix', '-s']:
            if i + 1 >= len(sys.argv):
                print("Error: --suffix requires a value")
                sys.exit(1)
            output_suffix = sys.argv[i + 1]
            i += 2
        elif arg in ['--threshold', '-t']:
            if i + 1 >= len(sys.argv):
                print("Error: --threshold requires a value")
                sys.exit(1)
            identity_threshold = float(sys.argv[i + 1])
            i += 2
        else:
            # Assume it's a PDB file
            pdb_files.append(arg)
            i += 1
    
    if len(pdb_files) < 2:
        print("Error: At least 2 PDB files are required")
        sys.exit(1)
    
    # Generate output filenames based on input filenames
    output_files = [generate_output_filename(pdb, output_suffix) for pdb in pdb_files]
    
    print(f"\nInput files: {len(pdb_files)}")
    for i, (pdb_in, pdb_out) in enumerate(zip(pdb_files, output_files)):
        print(f"  {i+1}. {pdb_in} -> {pdb_out}")
    print(f"\nIdentity threshold: {identity_threshold}")
    
    try:
        extract_common_residues_multi(pdb_files, output_files, identity_threshold)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
