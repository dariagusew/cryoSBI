from typing import Union, Optional
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm
from lampe.data import JointLoader, H5Dataset
from lampe.inference import NPELoss
from lampe.utils import GDStep
from itertools import islice

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.build_models import build_nle_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params
from cryo_sbi.inference.validate_train_config import check_train_params
import cryo_sbi.utils.image_utils as img_utils


def generate_validation_set(prior_loader, models, simulation_param, val_size, device):
    """
    Generates a fixed set of validation images with a specific noise model.
    """
    print(f"\nGenerating {val_size} validation images with Gaussian noise...")
    
    val_images = []
    val_indexes = []
    generated_count = 0
    val_iter = iter(prior_loader)

    with tqdm(total=val_size, desc="  Generating val set") as pbar:
        while generated_count < val_size:
            try:
                parameters = next(val_iter)
            except StopIteration:
                val_iter = iter(prior_loader) # Reset if we run out
                parameters = next(val_iter)
            
            (indices, quaternions, shift, defocus, b_factor, amp, snr) = parameters

            # change b_factors from 50.0 to 200.0
            ndata = indices.shape[0]
            b_factor = 50.0 + (200.0 - 50.0) * torch.rand(ndata, 1, 1, device=device)

            images, _ = cryo_em_simulator(
                models,
                indices.to(device, non_blocking=True),
                quaternions.to(device, non_blocking=True),
                shift.to(device, non_blocking=True),
                defocus.to(device, non_blocking=True),
                b_factor.to(device, non_blocking=True),
                amp.to(device, non_blocking=True),
                snr.to(device, non_blocking=True),
                simulation_param,
                "Gaussian"
            )
            
            val_images.append(images)
            val_indexes.append(indices.to(device))
            generated_count += len(images)
            pbar.update(len(images))

    # Concatenate all generated images and indexes
    all_images = torch.cat(val_images, dim=0)
    all_indexes = torch.cat(val_indexes, dim=0)
    # Compute memory
    val_mem_gb = all_images.nelement() * all_images.element_size() / 1024**3

    print(f"✅ Validation set created with {len(all_images)} images, consuming {val_mem_gb:.2f} GB of VRAM.")
    return all_images, all_indexes

def load_model(
    train_config: str,
    model_state_dict: str,
    device: str,
    train_from_checkpoint: bool,
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = False,
    image_size: int = 128
) -> torch.nn.Module:
    """
    Load model from checkpoint or from scratch.
    Optionally load pretrained embedding weights.

    Args:
        train_config (str): path to train config file
        model_state_dict (str): path to model state dict
        device (str): device to load model to
        train_from_checkpoint (bool): whether to load model from checkpoint
        pretrained_embedding_path (str, optional): path to pretrained embedding weights
        freeze_embedding (bool): if True, freeze embedding parameters
        image_size (int): number pixel images to create embedding
    """

    check_train_params(train_config)
    estimator = build_nle_flow_model(train_config, image_size)

    # Load pretrained embedding if provided
    if pretrained_embedding_path is not None:
        print(f"\n{'='*70}")
        print("LOADING PRETRAINED EMBEDDING")
        print(f"{'='*70}")
        print(f"Loading from: {pretrained_embedding_path}")

        try:
            pretrained_state = torch.load(pretrained_embedding_path, map_location='cpu')
            estimator.embedding.load_state_dict(pretrained_state, strict=False)
            print("✅ Pretrained embedding loaded successfully")
        except Exception as e:
            print(f"❌ Error loading pretrained embedding: {e}")
            raise

        # Optionally freeze embedding
        if freeze_embedding:
            print("\nFreezing embedding parameters...")
            for param in estimator.embedding.parameters():
                param.requires_grad = False

            frozen_count = sum(not p.requires_grad for p in estimator.embedding.parameters())
            total_count = len(list(estimator.embedding.parameters()))
            print(f"  Frozen: {frozen_count}/{total_count} parameters")
            print("  ✅ Embedding is frozen - will not be updated during training")
        else:
            print("\n⚠️  Embedding will be fine-tuned during training")
            print("   Using differential learning rate is recommended")

        print(f"{'='*70}\n")

    # Load from checkpoint
    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))

    estimator.to(device=device)

    # Always set to training mode
    estimator.train()
    print(f"✅ Model in training mode: {estimator.training}")

    return estimator


# Modify nle_train_no_saving function signature:
def nle_train_no_saving(
    image_config: str,
    train_config: str,
    epochs: int,
    estimator_file: str,
    loss_file: str,
    train_from_checkpoint: bool = False,
    model_state_dict: Union[str, None] = None,
    n_workers: int = 4,
    device: str = "cuda",
    saving_frequency: int = 10,
    simulation_batch_size: int = 2048,
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = False,
    use_differential_lr: bool = False,
    embedding_lr_factor: float = 0.01,
) -> None:
    """
    Train NLE model by simulating training data on the fly.
    Args:
        image_config (str): path to image config file
        train_config (str): path to train config file
        epochs (int): number of epochs
        estimator_file (str): path to estimator file
        loss_file (str): path to loss file
        train_from_checkpoint (bool, optional): train from checkpoint. Defaults to False.
        model_state_dict (str, optional): path to pretrained model state dict. Defaults to None.
        n_workers (int, optional): number of workers. Defaults to 1.
        device (str, optional): training device. Defaults to "cpu".
        saving_frequency (int, optional): frequency of saving model. Defaults to 20.
        pretrained_embedding_path: Path to pretrained ResNet18 weights
        freeze_embedding: If True, freeze embedding during training
        use_differential_lr: If True, use lower LR for embedding
        embedding_lr_factor: LR multiplier for embedding (if not frozen)

    Raises:
        Warning: No model state dict specified! --model_state_dict is empty

    Returns:
        None
    """
    train_config = json.load(open(train_config))
    check_train_params(train_config)
    image_config = json.load(open(image_config))

    assert simulation_batch_size >= train_config["BATCH_SIZE"]
    assert simulation_batch_size % train_config["BATCH_SIZE"] == 0

    if image_config["MODEL_FILE"].endswith("npy"):
        models = (
            torch.from_numpy(
                np.load(image_config["MODEL_FILE"]),
            )
            .to(device)
            .to(torch.float32)
        )
    else:
        models = torch.load(image_config["MODEL_FILE"]).to(device).to(torch.float32)

    image_prior = get_image_priors(len(models) - 1, image_config, models, device="cpu")
    prior_loader = PriorLoader(
        image_prior, batch_size=simulation_batch_size, num_workers=n_workers
    )

    # load simulation parameters into dictionary
    simulation_param = create_simulation_param(image_config, models, device=device)

    # Generate a fixed validation set before training loop
    n_val_images = 5 * simulation_batch_size
    validation_images, validation_indexes = generate_validation_set(
        prior_loader, models, simulation_param, n_val_images, device
    )

    # Load model with optional pretrained embedding
    estimator = load_model(
        train_config,
        model_state_dict,
        device,
        train_from_checkpoint,
        pretrained_embedding_path=pretrained_embedding_path,
        freeze_embedding=freeze_embedding,
        image_size = image_config["N_PIXELS"]
    )

    loss = NPELoss(estimator)

    # Setup optimizer with optional differential learning rates
    if freeze_embedding:
        # Only train flow
        print("\n" + "="*70)
        print("OPTIMIZER: TRAINING FLOW ONLY (EMBEDDING FROZEN)")
        print("="*70)
        print(f"Flow learning rate: {train_config['LEARNING_RATE']:.2e}")
        print("="*70 + "\n")

        optimizer = optim.AdamW(
            estimator.nle.parameters(),
            lr=train_config["LEARNING_RATE"],
            weight_decay=0.001
        )

    elif use_differential_lr and pretrained_embedding_path is not None:
        # Fine-tune embedding with lower LR
        flow_lr = train_config["LEARNING_RATE"]
        embedding_lr = flow_lr * embedding_lr_factor

        print("\n" + "="*70)
        print("OPTIMIZER: DIFFERENTIAL LEARNING RATES")
        print("="*70)
        print(f"Embedding learning rate: {embedding_lr:.2e}")
        print(f"Flow learning rate: {flow_lr:.2e}")
        print(f"Ratio: {embedding_lr_factor:.2f}")
        print("="*70 + "\n")

        optimizer = optim.AdamW([
            {
                'params': estimator.embedding.parameters(),
                'lr': embedding_lr,
                'weight_decay': 0.01,
                'name': 'embedding'
            },
            {
                'params': estimator.nle.parameters(),
                'lr': flow_lr,
                'weight_decay': 0.001,
                'name': 'flow'
            }
        ])

    else:
        # Standard: same LR for all
        print(f"\nOptimizer: Uniform learning rate = {train_config['LEARNING_RATE']:.2e}\n")

        optimizer = optim.AdamW(
            estimator.parameters(),
            lr=train_config["LEARNING_RATE"],
            weight_decay=0.001
        )

    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_loss = []
    val_losses = []

    print("Training neural network:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:

            losses = []
            for parameters in islice(prior_loader, 100):
                (
                    indices,
                    quaternions,
                    shift,
                    defocus,
                    b_factor,
                    amp,
                    snr,
                ) = parameters
                images, _ = cryo_em_simulator(
                    models,
                    indices.to(device, non_blocking=True),
                    quaternions.to(device, non_blocking=True),
                    shift.to(device, non_blocking=True),
                    defocus.to(device, non_blocking=True),
                    b_factor.to(device, non_blocking=True),
                    amp.to(device, non_blocking=True),
                    snr.to(device, non_blocking=True),
                    simulation_param,
                    simulation_param["noise"]
                )
                
                for _indices, _images in zip(
                    indices.split(train_config["BATCH_SIZE"]),
                    images.split(train_config["BATCH_SIZE"]),
                ):  
                    
                    losses.append(
                        step(
                            loss(
                                _images.to(device, non_blocking=True),
                               _indices.to(device, non_blocking=True)
                            )
                        )
                    )
            losses = torch.stack(losses)

            # Calculate mean loss 
            mean_loss.append(losses.mean().item())
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

            # Validation step at the end of each epoch 
            estimator.eval()
            with torch.no_grad():
                 val_loss = loss(validation_images, validation_indexes)
            val_losses.append(val_loss.item())
            # Set back to training mode
            estimator.train()

            # Update progress bar
            tq.set_postfix(loss=losses.mean().item(), val_loss=val_loss.item())

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
    torch.save(torch.tensor(val_losses), "val_"+loss_file)
