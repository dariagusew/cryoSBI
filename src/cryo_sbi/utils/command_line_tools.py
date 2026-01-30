import argparse
from modulefinder import Module
import sys
import torch
import numpy as np
from cryo_sbi.utils.generate_models import models_to_tensor
from cryo_sbi.utils.generate_models import models_to_tensor_topology
from cryo_sbi.utils.estimate_param_simulation_from_star import estimate_param_simulation_RELION 
from cryo_sbi.utils.estimate_snr_from_mrc import run_analysis
from cryo_sbi.utils.process_mrc_stack import process_mrc_stack
from cryo_sbi.utils.pretrain_image_embed_v2 import pretrain_image_embed
from cryo_sbi.utils.infer_populations import PopulationOptimizer
from cryo_sbi.utils.infer_populations import run_inference_real_data
from cryo_sbi.utils.infer_populations import run_inference_real_data_bayes
import cryo_sbi.utils.estimator_utils as est_utils
from cryo_sbi.inference.models import build_models
from pathlib import Path
import mrcfile


def cl_models_to_tensor():
    cl_parser = argparse.ArgumentParser(
        description="Convert models to tensor for cryoSBI",
        epilog="pdb-files: The name for the pdbs must contain a {} to be replaced by the index of the pdb file. The index starts at 0. \
        For example protein_{}.pdb. trr-files: For .trr files you must provide a topology file."
    )
    cl_parser.add_argument(
        "--model_files", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--output_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--n_pdbs", action="store", type=int, required=False, default=None
    )
    cl_parser.add_argument(
        "--top_file", action="store", type=str, required=False, default=None
    )
    args = cl_parser.parse_args()
    models_to_tensor(
        model_files=args.model_files,
        output_file=args.output_file,
        n_pdbs=args.n_pdbs,
        top_file=args.top_file
    )


def cl_models_to_tensor_topology():
    """
    Command line interface for converting PDB models to tensor with topology.
    """
    cl_parser = argparse.ArgumentParser(
        description="Convert multiple PDB models to tensor and create topology",
        epilog=""
    )
    cl_parser.add_argument(
        "--pdb_files", 
        action="store", 
        type=str, 
        nargs='+',
        required=True,
        help="List of PDB files to convert."
    )
    cl_parser.add_argument(
        "--output_models", 
        action="store", 
        type=str, 
        required=True,
        help="Output file path for the tensor (must be .pt file)."
    )
    cl_parser.add_argument(
        "--topo_type",
        action="store",
        type=str,
        required=True,
        default=None,
        choices=['allatom', 'allatom_com', 'calvados3', 'martini3'],
        help="Topology type: 'allatom', 'allatom_com', 'calvados3', 'martini3'"
    )
    cl_parser.add_argument(
        "--output_topology",
        action="store",
        type=str,
        required=False,
        default="topology.pt",
        help="Output topology file path (Optional, default topology.pt)."
    )
    cl_parser.add_argument(
        "--add_garbage_model",
        action="store_true",  # Stores True if the flag is present, False otherwise.
        help="If specified, add a garbage collector model."
    )

    
    args = cl_parser.parse_args()
    
    # Call the main function
    models_to_tensor_topology(
        pdb_files=args.pdb_files,
        output_models=args.output_models,
        topo_type=args.topo_type,
        output_topology=args.output_topology,
        add_garbage_model=args.add_garbage_model 
    )


def cl_process_mrc_stack():
    parser = argparse.ArgumentParser(
        description='Process MRC particle stack: fix header and downsample (optimized for large files)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument('input', type=str, help='Input MRC file')
    parser.add_argument('output', type=str, help='Output MRC file')
    parser.add_argument('size', type=int, help='Output size (pixels)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU processing (default: 32, reduce if GPU OOM)')
    parser.add_argument('--normalize', type=str, default='per_particle',
                       choices=['per_particle', 'global', 'none'],
                       help='Normalization method (default: per_particle)')
    parser.add_argument('--voxel-size', type=float, default=None,
                       help='Pixel size in Angstroms (default: read from header)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Read every Nth particle (default: 1, read all)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--max-size-gb', type=float, default=None,
                       help='Maximum file size to process in GB (default: no limit)')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate input file without processing')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.size <= 0 or args.size > 2048:
        print(f"❌ Error: Invalid output size: {args.size} (must be 1-2048)")
        sys.exit(1)
    
    if args.stride < 1:
        print(f"❌ Error: Invalid stride: {args.stride} (must be >= 1)")
        sys.exit(1)
    
    if args.batch_size < 1:
        print(f"❌ Error: Invalid batch size: {args.batch_size} (must be >= 1)")
        sys.exit(1)
    
    # Check if output will overwrite existing file
    if Path(args.output).exists() and not args.validate:
        response = input(f"⚠️  Output file exists: {args.output}\n   Overwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Aborted by user")
            sys.exit(1)
    
    # Process
    try:
        success = process_mrc_stack(
            input_path=args.input,
            output_path=args.output,
            target_size=args.size,
            batch_size=args.batch_size,
            normalize=args.normalize,
            voxel_size=args.voxel_size,
            device=args.device,
            max_size_gb=args.max_size_gb,
            stride=args.stride,
            validate_only=args.validate
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def cl_estimate_param_simulation_from_star():
    parser = argparse.ArgumentParser(
        description='Extraction of simulation parameters from cryo-EM data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument('star_file', type=str,
                        help='Input STAR file')

    parser.add_argument('--star_format', type=str, required=True,
                        choices=['RELION'], 
                        help='Input STAR file from RELION')
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  CRYO-EM SIMULATION PARAMETER EXTRACTION".center(58) + "║")
    print("║" + "  FROM STAR FILES".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print(f"\n📁 Input: {args.star_file}")
    print(f"\n📁 ISTAR format: {args.star_format}")
    
    # Check file exists
    if not Path(args.star_file).exists():
        print(f"\n❌ Error: File not found: {args.star_file}")
        return 1
    
    # Call appropriate function
    if(args.star_format=="RELION"):
       estimate_param_simulation_RELION(args.star_file)
 
    # Success message
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "✓ EXTRACTION COMPLETE!".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print()
    

def cl_estimate_snr_from_mrc():
    parser = argparse.ArgumentParser(description='Cryo-EM Particle Stack SNR Analyzer.', formatter_class=argparse.RawTextHelpFormatter)
    g_core = parser.add_argument_group('Core Parameters')
    g_core.add_argument('--input', '-i', required=True, help='Particle stack path (.mrc, .mrcs)')
    g_core.add_argument('--output', '-o', type=Path, default=Path("./snr_analysis_results"), help='Output directory for plots and data')
    g_core.add_argument('--max_particles', '-n', type=int, default=None, help='Max number of particles to analyze from the stack')
    g_core.add_argument('--device', '-d', type=str, default=None, help='Computation device: cuda or cpu (default: auto-detect)')
    g_core.add_argument('--batch_size', '-b', type=int, default=128, help='GPU batch size for analysis')
    
    g_mask_frac = parser.add_argument_group('Masking (defined as a fraction of image size)')
    g_mask_frac.add_argument('--signal_radius', type=float, default=0.5, help='Signal region radius as a fraction of image size. (Default: 0.5)')
    g_mask_frac.add_argument('--background_inner', type=float, default=0.6, help='Background annulus inner radius as a fraction of image size. (Default: 0.6)')
    g_mask_frac.add_argument('--background_outer', type=float, default=0.9, help='Background annulus outer radius as a fraction of image size. (Default: 0.9)')
    
    g_ctrl = parser.add_argument_group('Execution Control')
    g_ctrl.add_argument('--show-plots', action='store_true', help="Display plots at the end (in addition to saving them).")
    
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(exist_ok=True, parents=True)
        print(f"Results will be saved to: {args.output.resolve()}")
    run_analysis(args)
    print("\n✓ Analysis complete!")



def cl_pretrain_image_embed():
    """
    Parses command-line arguments and runs the pre-training script.
    """
    parser = argparse.ArgumentParser(
        description='Pre-training for image encoder on synthetic data with optional classification loss.'
    )

    # --- Required Arguments ---
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to image config JSON')

    # --- Model & Architecture ---
    parser.add_argument('--embedding', type=str, default='SPATIAL_CRYO',
                       choices=['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER', 'SPATIAL_CRYO_GAUSS_FFT_FILTER', 'RESNET18', 'RESNET18_FFT_FILTER'],
                       help='Embedding network architecture to use')
    parser.add_argument('--embedding_dim', type=int, default=16,
                       help='Output dimension of the embedding network (default: 16)')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Training batch size for the optimizer step (default: 256)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate for the AdamW optimizer (default: 2e-4)')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                       help='Weight for L2 regularization on embeddings (default: 0.0)')

    # --- I/O and Checkpoints ---
    parser.add_argument('--save_path', type=str, default='pretrained_image_embed.pt',
                       help='Output path for the saved encoder weights')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a full model checkpoint to resume training from')

    # --- Execution & Other ---
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training: "cpu", "cuda", "cuda:0", etc. (default: "cuda")')
    parser.add_argument('--simulation_batch_size', type=int, default=1024,
                       help='Number of images to simulate at once (default: 1024)')
    parser.add_argument('--n_batches_per_epoch', type=int, default=100,
                       help='Number of simulation batches to generate per epoch (default: 100)')
    parser.add_argument('--check_frequency', type=int, default=5,
                       help='How often to print detailed stats, in epochs (default: 5)')

    args = parser.parse_args()

    # --- Validate Device ---
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"❌ CUDA not available! Falling back to CPU")
            args.device = 'cpu'
        else:
            if ':' in args.device:
                try:
                    gpu_id = int(args.device.split(':')[1])
                    if gpu_id >= torch.cuda.device_count():
                        print(f"❌ GPU {gpu_id} not available! Available GPUs: 0-{torch.cuda.device_count()-1}")
                        print(f"   Falling back to cuda:0")
                        args.device = 'cuda:0'
                except ValueError:
                    print(f"❌ Invalid CUDA device format: {args.device}. Using default cuda:0.")
                    args.device = 'cuda:0'

            print(f"✅ Using device: {args.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(args.device)}")

    # --- Run Training ---
    model, final_loss = pretrain_image_embed(
        image_config_path=args.image_config,
        resume_from=args.resume_from,
        embedding_name=args.embedding,
        device=args.device,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        simulation_batch_size=args.simulation_batch_size,
        save_path=args.save_path,
        check_frequency=args.check_frequency,
        n_batches_per_epoch=args.n_batches_per_epoch,
        l2_weight=args.l2_weight
    )

    print(f"\n✅ Pre-training complete!")

def cl_pinfer_populations():

    parser = argparse.ArgumentParser(
        description="CLI for running PopulationOptimizer for Cryo-EM ensembles."
    )

    # ---- Required inputs ----
    parser.add_argument("--models_file", type=str, required=True,
                        help="Path to tensor file containing models (PyTorch .pt)")

    parser.add_argument("--train_config_file", type=str, required=True,
                        help="Training config JSON used to load estimator")

    parser.add_argument("--image_config_file", type=str, required=True,
                        help="Simulation config JSON for CryoEmSimulator")


    parser.add_argument("--estimator_file", type=str, required=True,
                        help="Estimator file name inside training directory.")
    parser.add_argument("--population_steps", type=int, default=11,
                        help="Number of population fractions between states (default 11)")

    # ---- Optional arguments ----
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: cuda or cpu")
    
    parser.add_argument("--states", type=int,default=1, help="Total number of states")

    parser.add_argument("--fractions", type=float,nargs="+",default=[0.0, 0.25, 0.5, 0.75, 1.0], help="Number of population fractions between states")
    
    parser.add_argument("--num_sim", type=int, default=100000,
                        help="Number of images to simulate (default 100000)")

    parser.add_argument("--output_prefix", type=str, default="results",
                        help="Prefix for results files (default 'results')")

    parser.add_argument("--verbose", action="store_true", help="Print detailed progress messages")

    parser.add_argument("--batch_size",type=int, default=10000, help="Batching for simulatin cryo images")
    
    parser.add_argument("--use_random",action="store_true",help="Use random population weights")




    args = parser.parse_args()

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------
    device = torch.device(args.device)
    models = torch.load(args.models_file).to(device)
    estimator = est_utils.load_estimator(args.train_config_file,
                                        args.image_config_file,
                                        build_models.build_nle_flow_model,
                                        args.estimator_file,
                                        device=device,
                                        )
    optimizer = PopulationOptimizer(
        models=models,
        estimator = estimator,
        device=device,
        population_steps=args.population_steps,
        num_sim=args.num_sim,
        use_random=args.use_random
    )

    rmse_vals, actual_weights, opt_weights, pop_fracs = optimizer.run_for_all_populations(
        sim_config=args.image_config_file,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    np.save(f"{args.output_prefix}_rmse.npy", rmse_vals)
    np.save(f"{args.output_prefix}_a_w.npy", actual_weights)
    np.save(f"{args.output_prefix}_opt_w.npy", opt_weights)
    np.save(f"{args.output_prefix}_pop.npy", pop_fracs)

    print("\n=== Optimization Complete ===")
    print(f"Saved RMSE values {args.output_prefix}_rmse.npy")
    print(f"Saved optimized weights {args.output_prefix}_opt_w.npy")
    print(f"Saved population fractions {args.output_prefix}_pop.npy")


def cl_run_inference_real_data():
    """
    Use a trained model to run inference on real data 
    """
    parser = argparse.ArgumentParser(description="Run inference to find model weights.")
    
    # --- Input Files ---
    parser.add_argument("--image-stack", type=str, required=True,
                        help="Path to the experimental particle stack (.mrcs file).")

    parser.add_argument("--models_file", type=str, required=True,
                        help="Path to tensor file containing models (PyTorch .pt)")

    parser.add_argument("--estimator_file", type=str, required=True,
                        help="Path to the saved estimator model weights (.pt file).")

    parser.add_argument("--train_config_file", type=str, required=True,
                        help="Path to the training parameters JSON file.")

    parser.add_argument("--image_config_file", type=str, required=True,
                        help="Path to the simulation parameters JSON file.")

    # --- Output File ---
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the resulting optimal weights tensor (.pt file).")

    # --- Performance and Hardware ---
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the computation on (e.g., 'cuda:0' or 'cpu').")

    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Batch size for pairwise likelihood evaluation. Adjust based on GPU memory.")
                        
    # parse arguments and run inference
    args = parser.parse_args() 

    # run inference
    run_inference_real_data(args)

def cl_run_inference_real_data_bayes():
    """
    Use a trained model to run inference on real data 
    """
    parser = argparse.ArgumentParser(description="Run inference to find model weights.")
    
    # --- Input Files ---
    parser.add_argument("--image-stack", type=str, required=True,
                        help="Path to the experimental particle stack (.mrcs file).")

    parser.add_argument("--models_file", type=str, required=True,
                        help="Path to tensor file containing models (PyTorch .pt)")

    parser.add_argument("--estimator_file", type=str, required=True,
                        help="Path to the saved estimator model weights (.pt file).")

    parser.add_argument("--train_config_file", type=str, required=True,
                        help="Path to the training parameters JSON file.")

    parser.add_argument("--image_config_file", type=str, required=True,
                        help="Path to the simulation parameters JSON file.")

    # --- Output File ---
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the resulting optimal weights tensor (.pt file).")

    # --- Performance and Hardware ---
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the computation on (e.g., 'cuda:0' or 'cpu').")

    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Batch size for pairwise likelihood evaluation. Adjust based on GPU memory.")
                        
    # parse arguments and run inference
    args = parser.parse_args() 

    # run inference
    run_inference_real_data_bayes(args)
