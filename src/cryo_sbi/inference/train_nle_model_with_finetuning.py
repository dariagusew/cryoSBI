# "train_nle_model_with_finetuning.py"
from typing import Tuple, Dict, Union, Optional
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm
from lampe.data import JointLoader, H5Dataset
from lampe.inference import NPELoss
from lampe.utils import GDStep
from itertools import islice
import gc
from contextlib import contextmanager
import copy
from torch.utils.data import Subset
from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.build_models import build_nle_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params
import cryo_sbi.utils.image_utils as img_utils


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

    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))

    estimator.to(device=device)

    estimator.train()
    print(f"✅ Model in training mode: {estimator.training}")

    return estimator


def nle_train_no_saving_with_finetuning(
    image_config: str,
    train_config: str,
    epochs: int,
    estimator_file: str,
    loss_file: str,
    train_from_checkpoint: bool = False,
    model_state_dict: Union[str, None] = None,
    n_workers: int = 4,
    device: str = "cuda",
    saving_frequency: int = 100,
    simulation_batch_size: int = 2048,
    n_batches_per_epoch: int = 100,
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = True,
    use_differential_lr: bool = False,
    embedding_lr_factor: float = 0.01
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
        n_workers (int, optional): number of workers. Defaults to 4.
        device (str, optional): training device. Defaults to "cuda".
        saving_frequency (int, optional): frequency of saving model. Defaults to 100.
        simulation_batch_size (int, optional): images generated per simulator call
        n_batches_per_epoch (int, optional): simulation calls per epoch
        pretrained_embedding_path (str, optional): Path to pretrained image embedding weights
        freeze_embedding (bool, optional): If True, freeze embedding during training
        use_differential_lr (bool, optional): If True, use lower LR for embedding
        embedding_lr_factor (float, optional): LR multiplier for embedding (if not frozen)
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

    n_models = len(models)

    image_prior = get_image_priors(n_models - 1, image_config, models, device="cpu")
    prior_loader = PriorLoader(
        image_prior, batch_size=simulation_batch_size, num_workers=n_workers
    )

    simulation_param = create_simulation_param(image_config, models, device=device)
 
    estimator = load_model(
        train_config,
        model_state_dict,
        device,
        train_from_checkpoint,
        pretrained_embedding_path=pretrained_embedding_path,
        freeze_embedding=freeze_embedding,
        image_size=image_config["N_PIXELS"]
    )

    loss = NPELoss(estimator)

    if freeze_embedding:
        print("\n" + "="*70)
        print("OPTIMIZER: TRAINING FLOW ONLY")
        print("="*70)
        print(f"Flow learning rate: {train_config['LEARNING_RATE']:.2e}")
        print("="*70 + "\n")

        optimizer = optim.AdamW([
            {
                'params': estimator.nle.parameters(),
                'lr': train_config["LEARNING_RATE"],
                'weight_decay': 0.001,
                'name': 'flow'
            }
        ])

    elif use_differential_lr and pretrained_embedding_path is not None:
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
        print(f"\nOptimizer: Uniform learning rate = {train_config['LEARNING_RATE']:.2e}\n")

        optimizer = optim.AdamW(
            estimator.parameters(),
            lr=train_config["LEARNING_RATE"],
            weight_decay=0.001
        )

    step = GDStep(optimizer, clip=train_config["CLIP_GRADIENT"])
    mean_loss = []

    # Simple cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    print("Training neural network:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = []

            for parameters in islice(prior_loader, n_batches_per_epoch):
                (
                    indices, quaternions, shift, defocus,
                    b_factor, amp, snr,
                ) = parameters
                
                noisy_images, _ = cryo_em_simulator(
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
                    noisy_images.split(train_config["BATCH_SIZE"]),
                ):  
                    losses.append(
                        step(
                            loss(
                                _images.to(device, non_blocking=True),
                                _indices.to(device, non_blocking=True)
                            )
                        )
                    )

            # calculate mean loss across mini-batches
            losses = torch.stack(losses)
            mean_train_loss = losses.mean().item()
            mean_loss.append(mean_train_loss)

            tq.set_postfix({
                'loss': mean_train_loss,
                'lr': scheduler.get_last_lr()[0],
            })

            # save model checkpoint
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

            # scheduler step
            scheduler.step()

    # save final stuff
    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
