import argparse
from modulefinder import Module
import torch
import numpy as np
from cryo_sbi.utils.generate_models import models_to_tensor
from cryo_sbi.utils.pretrain_resnet18_v5 import pretrain_unsupervised
from cryo_sbi.utils.pretrain_spatial_cryo_v3 import pretrain_spatial_cryo
from cryo_sbi.utils.infer_populations import PopulationOptimizer
import cryo_sbi.utils.estimator_utils as est_utils
from cryo_sbi.inference.models import build_models


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

def cl_pretrain_resnet_18():
    parser = argparse.ArgumentParser(
        description='Unsupervised pre-training for cryo-EM embeddings (reconstruction task)'
    )
    
    # Required arguments
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to image config JSON')
    
    # Embedding architecture
    parser.add_argument('--embedding', type=str, default='RESNET18',
                       choices=['RESNET18', 'RESNET18_FFT_FILTER'],
                       help='Embedding architecture. Choices: RESNET18, RESNET18_FFT_FILTER')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension (default: 256)')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                       help='L2 regularization weight on embeddings (default: 0.0)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='pretrained_embedding.pt',
                       help='Output path for pretrained weights')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: "cpu", "cuda", "cuda:0", "cuda:1", etc.')
    
    # Other
    parser.add_argument('--simulation_batch_size', type=int, default=1024,
                       help='Simulation batch size (default: 1024)')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                       help='Number of simulation batches per epoch (default: 100)')
    parser.add_argument('--check_frequency', type=int, default=5,
                       help='Print detailed stats every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"❌ CUDA not available! Falling back to CPU")
            args.device = 'cpu'
        else:
            if ':' in args.device:
                gpu_id = int(args.device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    print(f"❌ GPU {gpu_id} not available!")
                    print(f"   Available GPUs: 0-{torch.cuda.device_count()-1}")
                    print(f"   Falling back to cuda:0")
                    args.device = 'cuda:0'
            
            print(f"✅ Using device: {args.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(args.device)}")
    
    # Run pretraining
    model, final_loss = pretrain_unsupervised(
        image_config_path=args.image_config,
        embedding_name=args.embedding,
        device=args.device,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        simulation_batch_size=args.simulation_batch_size,
        save_path=args.output,
        n_batches_per_epoch=args.batches_per_epoch,
        check_frequency=args.check_frequency,
        l2_weight=args.l2_weight,
    )
    
    #if model is None:
    #    return 1
    
    print(f"\n✅ Unsupervised pre-training complete!")
    print(f"   Embedding: {args.embedding}")
    print(f"   L2 regularization weight: {args.l2_weight}")
    print(f"   Final reconstruction loss: {final_loss:.6f}")
    print(f"   Weights saved to: {args.output}")
    
    #return 0


#if __name__ == "__main__":
#    exit(main())
def cl_pretrain_spatial_cryo_v3():
    parser = argparse.ArgumentParser(
        description='Unsupervised pre-training for SPATIAL_CRYO encoder with L2 regularization'
    )
    
    # Required arguments
    parser.add_argument('--image_config', type=str, required=True,
                       help='Path to image config JSON')
    
    # Embedding architecture
    parser.add_argument('--embedding', type=str, default='SPATIAL_CRYO',
                       choices=['SPATIAL_CRYO', 'SPATIAL_CRYO_FFT_FILTER'],
                       help='Embedding architecture. Choices: SPATIAL_CRYO, SPATIAL_CRYO_FFT_FILTER')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 0.0002)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension (default: 256)')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                       help='L2 regularization weight on embeddings (default: 0.0)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='pretrained_spatial_cryo.pt',
                       help='Output path for pretrained weights')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: "cpu", "cuda", "cuda:0", "cuda:1", etc.')
    
    # Other
    parser.add_argument('--simulation_batch_size', type=int, default=1024,
                       help='Simulation batch size (default: 1024)')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                       help='Number of simulation batches per epoch (default: 100)')
    parser.add_argument('--check_frequency', type=int, default=5,
                       help='Print detailed stats every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"❌ CUDA not available! Falling back to CPU")
            args.device = 'cpu'
        else:
            if ':' in args.device:
                gpu_id = int(args.device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    print(f"❌ GPU {gpu_id} not available!")
                    print(f"   Available GPUs: 0-{torch.cuda.device_count()-1}")
                    print(f"   Falling back to cuda:0")
                    args.device = 'cuda:0'
            
            print(f"✅ Using device: {args.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(args.device)}")
    
    # Run pretraining
    model, final_loss = pretrain_spatial_cryo(
        image_config_path=args.image_config,
        embedding_name=args.embedding,
        device=args.device,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        simulation_batch_size=args.simulation_batch_size,
        save_path=args.output,
        n_batches_per_epoch=args.batches_per_epoch,
        check_frequency=args.check_frequency,
        l2_weight=args.l2_weight,
    )
    
    # if model is None:
    #     return 1
    
    print(f"\n✅ Unsupervised pre-training complete!")
    print(f"   Architecture: {args.embedding}")
    print(f"   L2 regularization weight: {args.l2_weight}")
    print(f"   Final reconstruction loss: {final_loss:.6f}")
    print(f"   Encoder weights saved to: {args.output}")
    
#     return 0


# if __name__ == "__main__":
#     exit(main())


def cli_pinfer_populations():

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

    # ---- Optional arguments ----
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: cuda or cpu")

    parser.add_argument("--population_steps", type=int, default=11,
                        help="Number of population fractions between states (default 11)")

    parser.add_argument("--num_sim", type=int, default=100000,
                        help="Number of images to simulate (default 100000)")

    parser.add_argument("--output_prefix", type=str, default="results",
                        help="Prefix for results files (default 'results')")

    parser.add_argument("--verbose", action="store_true", help="Print detailed progress messages")

    parser.add_argument("--batch_size",type=int, default=10000, help="Batching for simulatin cryo images")

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
        num_sim=args.num_sim
    )

    rmse_vals, pop_fracs = optimizer.run_for_all_populations(
        sim_config=args.image_config_file,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    np.save(f"{args.output_prefix}_rmse.npy", rmse_vals)
    np.save(f"{args.output_prefix}_pop.npy", pop_fracs)

    print("\n=== Optimization Complete ===")
    print(f"Saved RMSE values {args.output_prefix}_rmse.npy")
    print(f"Saved population fractions {args.output_prefix}_pop.npy")
