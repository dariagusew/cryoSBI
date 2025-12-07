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
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params
from cryo_sbi.inference.validate_train_config import check_train_params
import cryo_sbi.utils.image_utils as img_utils


def load_model(
    train_config: str,
    model_state_dict: str,
    device: str,
    train_from_checkpoint: bool,
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = False,
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
    """

    check_train_params(train_config)
    estimator = build_nle_flow_model(train_config)

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
    n_workers: int = 1,
    device: str = "cpu",
    saving_frequency: int = 20,
    simulation_batch_size: int = 1024,
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

    num_pixels = torch.tensor(
        image_config["N_PIXELS"], dtype=torch.float32, device=device
    )
    pixel_size = torch.tensor(
        image_config["PIXEL_SIZE"], dtype=torch.float32, device=device
    )

    voltage = image_config.get("VOLTAGE", 300.0)
    cs = image_config.get("SPHERICAL_ABERRATION", 0.0)

    # Load model with optional pretrained embedding
    estimator = load_model(
        train_config,
        model_state_dict,
        device,
        train_from_checkpoint,
        pretrained_embedding_path=pretrained_embedding_path,
        freeze_embedding=freeze_embedding,
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

    print("Training neural network:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:

            losses = []
            for parameters in islice(prior_loader, 100):
                (
                    indices,
                    quaternions,
                    res,
                    shift,
                    defocus,
                    b_factor,
                    amp,
                    snr,
                ) = parameters
                images = cryo_em_simulator(
                    models,
                    indices.to(device, non_blocking=True),
                    quaternions.to(device, non_blocking=True),
                    res.to(device, non_blocking=True),
                    shift.to(device, non_blocking=True),
                    defocus.to(device, non_blocking=True),
                    b_factor.to(device, non_blocking=True),
                    amp.to(device, non_blocking=True),
                    snr.to(device, non_blocking=True),
                    num_pixels,
                    pixel_size,
                    voltage,
                    cs
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

            tq.set_postfix(loss=losses.mean().item())
            mean_loss.append(losses.mean().item())
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
