from typing import Union
import MDAnalysis as mda
from MDAnalysis.analysis import align
import torch
import numpy as np

def pdb_parser_(fname: str, atom_selection: str = "name CA") -> torch.tensor:
    """
    Parses a pdb file and returns a coarsed grained atomic model of the protein.
    The atomic model is a 5xN array, where N is the number of residues in the protein.
    The first three rows are the x, y, z coordinates of the alpha carbons.

    Parameters
    ----------
    fname : str
        The path to the pdb file.

    Returns
    -------
    atomic_model : torch.tensor
        The coarse grained atomic model of the protein.
    """

    univ = mda.Universe(fname)
    univ.atoms.translate(-univ.atoms.center_of_mass())

    model = torch.from_numpy(univ.select_atoms(atom_selection).positions.T)

    return model


def pdb_parser(file_formatter, n_pdbs, output_file, start_index=1, **kwargs):
    """
    Parses multiple pdb files and returns an coarsed grained model of the protein. The atomic model is a 5xN array, where N is the number of atoms or residues in the protein. The first three rows are the x, y, z coordinates of the atoms or residues. The fourth row is the atomic number of the atoms or the density of the residues. The fifth row is the variance of the atoms or residues, which is the resolution of the cryo-EM map divided by pi squared.

    Parameters
    ----------
    file_formatter : str
        The path to the pdb file. The path must contain the placeholder {} for the pdb index. For example, if the path is "data/pdb/{}.pdb", then the placeholder is {}.
    n_pdbs : int
        The number of pdb files to parse.
    output_file : str
        The path to the output file. The output file must be a .pt file.
    mode : str
        The mode of the atomic model. Either "resid" or "all atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.
    """

    models = pdb_parser_(file_formatter.format(start_index), **kwargs)
    models = torch.zeros((n_pdbs, *models.shape))

    for i in range(0, n_pdbs):
        models[i] = pdb_parser_(file_formatter.format(start_index + i), **kwargs)

    if output_file.endswith("pt"):
        torch.save(models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return


def traj_parser_(top_file: str, traj_file: str) -> torch.tensor:
    """
    Parses a traj file and returns a coarsed grained atomic model of the protein.
    The atomic model is a Mx3xN array, where M is the number of frames in the trajectory,
    and N is the number of residues in the protein. The first three rows in axis 1 are the x, y, z coordinates of the alpha carbons.

    Parameters
    ----------
    top_file : str
        The path to the traj file.

    Returns
    -------
    atomic_model : torch.tensor
        The coarse grained atomic model of the protein.
    """

    ref = mda.Universe(top_file)
    ref.atoms.translate(-ref.atoms.center_of_mass())

    mobile = mda.Universe(top_file, traj_file)
    align.AlignTraj(mobile, ref, select="name CA", in_memory=True).run()

    atomic_models = torch.zeros(
        (mobile.trajectory.n_frames, 3, mobile.select_atoms("name CA").n_atoms)
    )

    for i in range(mobile.trajectory.n_frames):
        mobile.trajectory[i]

        atomic_models[i, 0:3, :] = torch.from_numpy(
            mobile.select_atoms("name CA").positions.T
        )

    return atomic_models


def traj_parser(top_file: str, traj_file: str, output_file: str) -> None:
    """
    Parses a traj file and returns an atomic model of the protein. The atomic model is a Mx5xN array, where M is the number of frames in the trajectory, and N is the number of atoms in the protein. The first three rows in axis 1 are the x, y, z coordinates of the atoms. The fourth row is the atomic number of the atoms. The fifth row is the variance of the atoms before the resolution is applied.

    Parameters
    ----------
    top_file : str
        The path to the topology file.
    traj_file : str
        The path to the trajectory file.
    output_file : str
        The path to the output file. Must be a .pt file.
    mode : str
        The mode of the atomic model. Either "resid" or "all-atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.

    Returns
    -------
    None
    """

    atomic_models = traj_parser_(top_file, traj_file)

    if output_file.endswith("pt"):
        torch.save(atomic_models, output_file)

    else:
        raise ValueError("Model file format not supported. Please use .pt.")

    return


def models_to_tensor(
        model_files, 
        output_file, 
        n_pdbs: Union[int, None] = None,
        top_file: Union[str, None] = None,
    ):
    """
    Converts different model files to a torch tensor.
    
    Parameters
    ----------
    model_files : list
        A list of model files to convert to a torch tensor.
        
    output_file : str
        The path to the output file. Must be a .pt file.
        
    n_models : int
        The number of models to convert to a torch tensor. Just needed for models in pdb files.

    top_file : str
        The path to the topology file. Just needed for models in trr files.
    
    Returns
    -------
        None
    """
    assert output_file.endswith("pt"), "The output file must be a .pt file."
    if model_files.endswith("trr"):
        assert top_file is not None, "Please provide a topology file."
        assert n_pdbs is None, "The number of pdb files is not needed for trr files."
        traj_parser(top_file, model_files, output_file)
    elif model_files.endswith("pdb"):
        assert n_pdbs is not None, "Please provide the number of pdb files."
        assert top_file is None, "The topology file is not needed for pdb files."
        pdb_parser(model_files, n_pdbs, output_file)


def get_atomistic_topology(resnames):
    """
    Extract scattering factors (A and B values) for given residues.
    
    Parameters
    ----------
    resnames : list
        List of residue names (3-letter codes for amino acids, 1-2 letter codes for nucleic acids).
    
    Returns
    -------
    torch.Tensor
        Tensor of shape [2, n_residues] containing A and B scattering factors.
    """
    # Scattering factors: A and B values
    # Protein: centered in CA, RNA/DNA: centered in C1'
    scattering = {
        # Amino acids
        'ALA': (11.7241, 27.9), 'ARG': (25.8910, 62.8), 'ASN': (18.4298, 42.6), 'ASP': (18.2001, 42.1),
        'CYS': (16.8838, 38.9), 'GLN': (20.9390, 49.2), 'GLU': (20.7093, 48.7), 'GLY': (9.2149, 26.2),
        'HIS': (23.6779, 55.8), 'ILE': (19.2517, 42.9), 'LEU': (19.2517, 44.8), 'LYS': (21.4648, 50.3),
        'MET': (21.9022, 51.4), 'PHE': (26.7793, 63.5), 'PRO': (16.7425, 36.9), 'SER': (13.7075, 32.0),
        'THR': (16.2167, 36.6), 'TRP': (34.0108, 83.5), 'TYR': (28.7627, 69.0), 'VAL': (16.7425, 37.6),
        # RNA
        'A': (53.5441, 97.9), 'C': (48.5921, 87.5), 'G': (55.5275, 101.3), 'U': (48.3624, 87.2),
        # DNA
        'DA': (51.5607, 98.3), 'DC': (46.6087, 88.7), 'DG': (53.5441, 102.8), 'DT': (48.8882, 92.7)
    }
    
    # Extract A and B values for each residue using list comprehension
    try:
        A_list = [scattering[r][0] for r in resnames]
        B_list = [scattering[r][1] for r in resnames]
    except KeyError as e:
        raise ValueError(f"Unknown residue name: {e}. Please check your topology.")
    
    # Create tensor with shape [2, n_residues]
    topo = torch.tensor([A_list, B_list], dtype=torch.float32)
    
    return topo

def get_calvados_topology(resnames):
    """
    Extract CALVADOS scattering factors (A and B values) for given residues.
    
    Parameters
    ----------
    resnames : list
        List of residue names (3-letter codes for amino acids).
    
    Returns
    -------
    torch.Tensor
        Tensor of shape [2, n_residues] containing A and B scattering factors
        for CALVADOS coarse-grained model.
    """
    # CALVADOS scattering factors: A and B values (centered in CA)
    scattering = {
        'ALA': (11.7241, 27.3), 'ARG': (25.8910, 73.2), 'ASN': (18.4298, 41.6), 'ASP': (18.2001, 41.2),
        'CYS': (16.8838, 38.3), 'GLN': (20.9390, 52.5), 'GLU': (20.7093, 52.2), 'GLY': (9.2149, 23.6),
        'HIS': (23.6779, 52.9), 'ILE': (19.2517, 41.5), 'LEU': (19.2517, 45.0), 'LYS': (21.4648, 57.1),
        'MET': (21.9022, 54.2), 'PHE': (26.7793, 58.4), 'PRO': (16.7425, 34.3), 'SER': (13.7075, 31.3),
        'THR': (16.2167, 35.0), 'TRP': (34.0108, 67.6), 'TYR': (28.7627, 61.0), 'VAL': (16.7425, 36.3)
    }
    
    # Extract A and B values for each residue using list comprehension
    try:
        A_list = [scattering[r][0] for r in resnames]
        B_list = [scattering[r][1] for r in resnames]
    except KeyError as e:
        raise ValueError(f"Unknown residue name: {e}. CALVADOS only supports standard amino acids.")
    
    # Create tensor with shape [2, n_residues]
    topo = torch.tensor([A_list, B_list], dtype=torch.float32)
    
    return topo

def models_to_tensor_topology(
        pdb_files,
        at_selection,
        output_models,
        topo_type=None,
        output_topology=None
    ):
    """
    Converts different model files in pdb format to a torch tensor and create topology.
    Parameters
    ----------
    pdb_files : list
        A list of PDB files to convert to a torch tensor.
    at_selection : str
        Selection of atoms to include in models.
    output_models : str
        The path to the output file for the tensor. Must be a .pt file.
    topo_type : str
        The type of topology (optional, 'atomistic' or 'Calvados').
    output_topology : str
        The path to the output topology file. Must be a .pt file.
    Returns
    -------
    None
    """
    
    # Initialize lists to store positions and residues from all models
    pos_list = []
    res_list = []
    
    # Loop through all PDB files to extract atomic information
    for pdb in pdb_files:
        # Create MDAnalysis Universe object from PDB file
        u = mda.Universe(pdb)
        # Atoms selections
        atoms = u.select_atoms(at_selection)
        # Extract atom positions as numpy array with shape [natoms, 3]
        pos = atoms.positions
        # Transpose and append positions to the list
        pos_list.append(pos.T)
        # Extract residue information (names, indices, etc.)
        res = atoms.residues
        # Append residues to the list
        res_list.append(res)

    # Validate that all models have the same number of atoms
    n_atoms = len(pos_list[0])
    for i, pos in enumerate(pos_list):
        if len(pos) != n_atoms:
            raise ValueError(
                f"Model {i} has {len(pos)} atoms, but model 0 has {n_atoms} atoms. "
                "All models must have the same number of atoms."
            )

    # Validate that all models have the same residue composition
    ref_resnames = [res.resname for res in res_list[0]]
    ref_resids = [res.resid for res in res_list[0]]
    
    for i, res in enumerate(res_list[1:], start=1):
        current_resnames = [r.resname for r in res]
        current_resids = [r.resid for r in res]
        
        if current_resnames != ref_resnames or current_resids != ref_resids:
            raise ValueError(
                f"Model {i} has different residues than model 0. "
                "All models must have the same residue composition."
            )

    # Convert list of numpy arrays to torch tensor [n_models, 3, n_atoms]
    model = torch.tensor(np.array(pos_list), dtype=torch.float32)

    # Center models by subtracting the geometric center of each model
    # Calculate center: mean along atoms dimension (dim=2), keep dims for broadcasting
    center = model.mean(dim=2, keepdim=True)  # Shape: [n_models, 3, 1]
    model = model - center  # Broadcast subtraction across all atoms

    # Save the tensor to the specified output file
    torch.save(model, output_models)
    print(f"Saved {len(pdb_files)} models to {output_models} with shape {model.shape}")

    # Prepare topology
    if(topo_type!=None):
       if(topo_type.lower()=="atomistic"):
          topo = get_atomistic_topology(ref_resnames)
          
       elif(topo_type.lower()=="calvados"):
          topo = get_calvados_topology(ref_resnames)

    if(topo_type!=None):
       torch.save(topo, output_topology)
       print(f"Saved topology to {output_topology} in {topo_type.upper()} format") 
