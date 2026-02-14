from typing import Union
import MDAnalysis as mda
from MDAnalysis.analysis import align
import torch
import numpy as np
import math

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


def get_allatom_topology(atypes):
    """
    Extract scattering factors (A and B values) for given atoms.
    
    Parameters
    ----------
    atypes : list
        List of atom types.
    
    Returns
    -------
    torch.Tensor
        Tensor of shape [2, n_atoms] containing A and B scattering factors.
    """
    # Scattering factors: 1-Gaussian fit of experimental scattering factors
    # 
    # Reciprocal space form: f(s) = A * exp(-B*s²)
    #   A: scattering amplitude (electrons)
    #   B: decay parameter (Ų, controls atomic size)
    #
    scattering = {
        # Atoms (single Gaussian: A, B)
        'C': (2.361558, 9.617784),     'O': (1.892568, 5.745714),     'N': (2.097861, 7.249698),
        'S': (4.837811, 9.925579),     'P': (5.099359, 11.913530)
    }
    # Transform scattering factors from reciprocal space to real space via Fourier transform
    # 
    # 3D real space Gaussian: f(r) = A * (π/B)^1.5 * exp(-π²*r²/B)
    #
    # For efficient 2D projection rendering, we use separable form: f(x,y) = f(x) * f(y)
    # where f(x) = A1 * exp(B1 * x²) and f(y) = A1 * exp(B1 * y²)
    #
    # Pre-computed auxiliary parameters:
    #   A1 = √(A*π/B)  - amplitude prefactor for 1D projections
    #   B1 = -π²/B     - exponent coefficient (negative for decay)
    #
    # Normalization: ∫∫ f(x)*f(y) dx dy = A (preserves scattering amplitude)
    #
    try:
        A1 = [math.sqrt(scattering[at][0] * math.pi / scattering[at][1]) for at in atypes]
        B1 = [-math.pi**2 / scattering[at][1] for at in atypes] 
    except KeyError as e:
        raise ValueError(f"Unknown atom type: {e}. Please check your topology.")

    # Store as tensor with shape [2, n_atoms] for efficient GPU computation
    # topo[0, :] = A1 coefficients, topo[1, :] = B1 coefficients
    topo = torch.tensor([A1, B1], dtype=torch.float32)

    return topo

def get_calvados_topology(resnames):
    """
    Extract CALVADOS scattering factors (A and B values) for given residues.
    These parameters can be used with any one bead per residue representation,
    where the bead is centered on the residue COM.
    
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
    # CALVADOS3 bead positions: centered on residue COM
    # Scattering factors: 1-Gaussian approximation in reciprocal space
    # see comments above 
    scattering = {
        'ALA': (11.6, 27.0), 'ARG': (32.8, 89.5), 'ASN': (19.0, 42.5), 'ASP': (18.6, 42.0),
        'CYS': (17.2, 39.0), 'GLN': (21.6, 54.0), 'GLU': (21.6, 54.0), 'GLY': (8.8, 23.0),
        'HIS': (25.0, 55.5), 'ILE': (19.0, 41.0), 'LEU': (18.8, 44.0), 'LYS': (21.2, 56.5),
        'MET': (22.8, 56.0), 'PHE': (29.0, 62.5), 'PRO': (17.2, 35.0), 'SER': (14.0, 32.0),
        'THR': (16.8, 36.0), 'TRP': (38.6, 75.5), 'TYR': (31.0, 65.0), 'VAL': (17.0, 36.5)
    }
    # Transform scattering factors from reciprocal space to real space via Fourier transform
    # see comments above 
    try:
        A1 = [math.sqrt(scattering[res][0] * math.pi / scattering[res][1]) for res in resnames]
        B1 = [-math.pi**2 / scattering[res][1] for res in resnames]
    except KeyError as e:
        raise ValueError(f"Unknown residue name: {e}. Please check your topology.")

    # Store as tensor with shape [2, n_residues] for efficient GPU computation
    # topo[0, :] = A1 coefficients, topo[1, :] = B1 coefficients
    topo = torch.tensor([A1, B1], dtype=torch.float32)

    return topo

def get_martini_topology(resnames, beadnames):
    """
    Extract Martini3 scattering factors (A and B values) for given residue-bead pairs.
    
    Parameters
    ----------
    resnames : list
        List of residue names (e.g., ['ALA', 'ALA', 'CYS', ...]).
    beadnames : list
        List of bead names corresponding to each residue (e.g., ['BB', 'SC1', 'BB', ...]).
    
    Returns
    -------
    torch.Tensor
        Tensor of shape [2, n_beads] containing A and B scattering factors
        for the Martini 3 coarse-grained model.
    """
    # Scattering factors for Martini 3 beads re-calculated using the approach in:
    #
    # Hoff SE, Thomasen FE, Lindorff-Larsen K, Bonomi M
    # PLoS Comput Biol 20(7): e1012180 (2024). https://doi.org/10.1371/journal.pcbi.1012180
    #
    # using a set of high-res X-ray structures. 
    # Keys are formatted as 'RESNAME_BEADNAME'.
    #
    # Reciprocal space form: f(s) = A * exp(-B*s²)
    scattering = {
        # Alanine (ALA)
        'ALA_BB': (8.9, 23.0), 'ALA_SC1': (1.3, 4.0),
        # Arginine (ARG)
        'ARG_BB': (8.9, 23.0), 'ARG_SC1': (6.9, 19.0), 'ARG_SC2': (8.7, 18.5),
        # Asparagine (ASN)
        'ASN_BB': (8.9, 23.0), 'ASN_SC1': (8.5, 18.5),
        # Aspartic Acid (ASP)
        'ASP_BB': (8.9, 23.0), 'ASP_SC1': (8.3, 18.0),
        # Cysteine (CYS)
        'CYS_BB': (8.9, 23.0), 'CYS_SC1': (5.7, 11.5),
        # Glutamine (GLN)
        'GLN_BB': (8.9, 23.0), 'GLN_SC1': (11.1, 25.0),
        # Glutamic Acid (GLU)
        'GLU_BB': (8.9, 23.0), 'GLU_SC1': (10.9, 24.0),
        # Glycine (GLY)
        'GLY_BB': (8.9, 23.0),
        # Histidine (HIS)
        'HIS_BB': (8.9, 23.0), 'HIS_SC1': (4.3, 13.0), 'HIS_SC2': (3.9, 10.5), 'HIS_SC3': (3.9, 10.0),
        # Isoleucine (ILE)
        'ILE_BB': (8.9, 23.0), 'ILE_SC1': (9.7, 26.0),
        # Leucine (LEU)
        'LEU_BB': (8.9, 23.0), 'LEU_SC1': (9.3, 22.5),
        # Lysine (LYS)
        'LYS_BB': (8.9, 23.0), 'LYS_SC1': (6.9, 19.0), 'LYS_SC2': (4.1, 12.0),
        # Methionine (MET)
        'MET_BB': (8.9, 23.0), 'MET_SC1': (11.3, 23.5),
        # Phenylalanine (PHE)
        'PHE_BB': (8.9, 23.0), 'PHE_SC1': (9.1, 21.0), 'PHE_SC2': (6.7, 17.0), 'PHE_SC3': (6.7, 17.0),
        # Proline (PRO)
        'PRO_BB': (9.1, 23.5), 'PRO_SC1': (6.9, 18.5),
        # Serine (SER)
        'SER_BB': (8.9, 23.0), 'SER_SC1': (3.7, 10.0),
        # Threonine (THR)
        'THR_BB': (8.9, 23.0), 'THR_SC1': (6.5, 17.5),
        # Tryptophan (TRP)
        'TRP_BB': (8.9, 23.0), 'TRP_SC1': (4.3, 13.0), 'TRP_SC2': (3.9, 10.5), 'TRP_SC3': (4.1, 11.5), 'TRP_SC4': (4.1, 11.5), 'TRP_SC5': (4.1, 11.5),
        # Tyrosine (TYR)
        'TYR_BB': (8.9, 23.0), 'TYR_SC1': (4.3, 13.0), 'TYR_SC2': (4.1, 11.5), 'TYR_SC3': (4.1, 11.5), 'TYR_SC4': (3.7, 10.0),
        # Valine (VAL)
        'VAL_BB': (9.1, 23.5), 'VAL_SC1': (6.9, 19.0)
    }
    # Transform scattering factors from reciprocal space to real space via Fourier transform 
    # see comments above 
    # Combine residue and bead names to create the lookup keys
    lookup_keys = [f"{res}_{bead}" for res, bead in zip(resnames, beadnames)]
    try:
        A1 = [math.sqrt(scattering[key][0] * math.pi / scattering[key][1]) for key in lookup_keys]
        B1 = [-math.pi**2 / scattering[key][1] for key in lookup_keys]
    except KeyError as e:
        raise ValueError(f"Unknown Martini residue-bead combination: {e}. "
                         "Please check that your PDB file and topology are consistent with the Martini 3 protein model.")

    # Store as tensor with shape [2, n_beads] for efficient GPU computation
    # topo[0, :] = A1 coefficients, topo[1, :] = B1 coefficients
    topo = torch.tensor([A1, B1], dtype=torch.float32)

    return topo


def models_to_tensor_topology(
        pdb_files,
        output_models,
        topo_type,
        output_topology
    ):
    """
    Converts different model files in pdb format to a torch tensor and create topology.
    Parameters
    ----------
    pdb_files : list
        A list of PDB files to convert to a torch tensor.
    output_models : str
        The path to the output file for the models. Must be a .pt file.
    topo_type : str
        The type of topology ('allatom', 'oneatom', 'calvados3', 'martini3').
    output_topology : str
        The path to the output topology file. Must be a .pt file.
    Returns
    -------
    None
    """
    
    # Initialize lists to store models data
    pos_list = []
    at_list = []
 
    # Loop through all PDB files to extract atomic information
    for pdb in pdb_files:

        # Create MDAnalysis Universe object from PDB file
        u = mda.Universe(pdb)
        # Select all heavy atoms / coarse-grained beads
        atoms = u.select_atoms("not type H")

        # Get positions
        if topo_type in ["allatom", "calvados3", "martini3"]:
           # Extract atom positions as numpy array with shape [natoms, 3]
           pos = atoms.positions

        elif(topo_type=="allatom_com"):
           # Compute positions of residues COM [nres, 3]
           pos = np.array([r.atoms.select_atoms("not type H").center_of_mass() for r in atoms.residues])

        # Transpose and append positions to the list
        pos_list.append(pos.T)
        # Append atoms to the list
        at_list.append(atoms)

    # Validate that all models have the same number of atoms
    n_atoms = len(pos_list[0])
    for i, pos in enumerate(pos_list):
        if len(pos) != n_atoms:
            raise ValueError(
                f"Model {i} has {len(pos)} atoms, but model 0 has {n_atoms} atoms. "
                "All models must have the same number of atoms."
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
    if(topo_type=="allatom"):
       # list of atom types
       atypes = [at.type for at in at_list[0]]
       # get topo
       topo = get_allatom_topology(atypes)
 
    elif(topo_type=="allatom_com" or topo_type=="calvados3"):
       # list of residue names
       resnames = [r.resname for r in at_list[0].residues]
       # get topo
       topo = get_calvados_topology(resnames)

    elif(topo_type=="martini3"):
       # lists of residue and bead names
       resnames = [at.residue.resname for at in at_list[0]]
       beadnames = [at.name for at in at_list[0]]
       # get topo
       topo = get_martini_topology(resnames, beadnames)

    # Save the tensor to the specified output file
    torch.save(topo, output_topology)
    print(f"Saved topology to {output_topology} in {topo_type.upper()} format") 
