# "train_nle_model_with_finetuning.py"
from typing import Tuple, Dict, Union, Optional
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from lampe.inference import NPELoss
from lampe.utils import GDStep
from itertools import islice
from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.build_models import build_nle_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param
import cryo_sbi.utils.image_utils as img_utils

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not installed. Real image loading disabled.")


# =======================================================================================
# REAL IMAGE DATASET
# =======================================================================================

class RealImageMRCDataset(Dataset):
    """
    Dataset for loading real images from an MRC stack.
    Uses memory-mapping so the full file is not loaded into RAM.
    Applies per-image z-score normalization and keeps a small in-memory cache.
    """
    def __init__(self, mrc_path, cache_size=10000):
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile not installed. Install with: pip install mrcfile")

        self.mrc_path = Path(mrc_path)
        self.cache_size = cache_size

        if not self.mrc_path.exists():
            raise FileNotFoundError(f"MRC file not found: {mrc_path}")

        print(f"  Opening MRC file: {mrc_path}")
        self.mrc_file = mrcfile.mmap(str(self.mrc_path), mode='r')
        self.mrc_data = self.mrc_file.data

        self.n_images = self.mrc_data.shape[0]
        self.image_shape = self.mrc_data.shape[1:]

        print(f"  Loaded MRC: {self.n_images} images of shape {self.image_shape}")
        print(f"  Loading method: mrcfile mmap")

        self.cache = {}
        self.cache_order = []

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img = np.asarray(self.mrc_data[idx], dtype=np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)

        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        self.cache[idx] = torch.from_numpy(img.copy())
        self.cache_order.append(idx)

        return self.cache[idx]


# =======================================================================================
# PREDICTOR HEAD (from pretraining)
# =======================================================================================

class FullParamPredictor(nn.Module):
    """
    Predicts all parameters (X, θ) from z.
    Used here only as a frozen teacher for real-image pseudo-labeling.
    """
    def __init__(self, embedding_dim: int, n_conformations: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conf_head    = nn.Linear(hidden_dim, n_conformations)
        self.orient_head  = nn.Linear(hidden_dim, 4)
        self.shift_head   = nn.Linear(hidden_dim, 2)
        self.defocus_head = nn.Linear(hidden_dim, 1)
        self.bfactor_head = nn.Linear(hidden_dim, 1)
        self.snr_head     = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {
            "conf":    self.conf_head(h),
            "orient":  self.orient_head(h),
            "shift":   self.shift_head(h),
            "defocus": self.defocus_head(h).squeeze(-1),
            "bfactor": self.bfactor_head(h).squeeze(-1),
            "snr":     self.snr_head(h).squeeze(-1),
        }


# =======================================================================================
# MODEL LOADING
# =======================================================================================

def load_model(
    train_config: str,
    model_state_dict: str,
    device: str,
    train_from_checkpoint: bool,
    pretrained_embedding_path: Optional[str] = None,
    pretrained_full_model_path: Optional[str] = None,
    freeze_embedding: bool = False,
    image_size: int = 128,
    n_conformations: Optional[int] = None,
) -> Tuple[torch.nn.Module, Optional[nn.Module]]:
    """
    Load model from checkpoint or from scratch.
    - If pretrained_embedding_path is given, load encoder weights only.
    - If pretrained_full_model_path is given, load encoder + predictor.
    The two options are mutually exclusive.
    Returns (estimator, predictor). Predictor is None unless full model is loaded.
    """

    if pretrained_embedding_path is not None and pretrained_full_model_path is not None:
        raise ValueError(
            "pretrained_embedding_path and pretrained_full_model_path are mutually exclusive."
        )

    check_train_params(train_config)
    estimator = build_nle_flow_model(train_config, image_size)
    predictor = None

    # ------------------------------------------------------------------
    # Load pretrained encoder / full model
    # ------------------------------------------------------------------
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

    elif pretrained_full_model_path is not None:
        if n_conformations is None:
            raise ValueError(
                "n_conformations must be provided when loading a full pretrained model."
            )

        print(f"\n{'='*70}")
        print("LOADING FULL PRETRAINED MODEL (encoder + predictor)")
        print(f"{'='*70}")
        print(f"Loading from: {pretrained_full_model_path}")

        full_state = torch.load(pretrained_full_model_path, map_location='cpu')

        # Extract and load encoder state
        encoder_state = {
            k[len('encoder.'):]: v
            for k, v in full_state.items()
            if k.startswith('encoder.')
        }
        estimator.embedding.load_state_dict(encoder_state, strict=False)
        print("✅ Pretrained embedding loaded successfully")

        # Extract and load predictor state
        predictor_state = {
            k[len('predictor.'):]: v
            for k, v in full_state.items()
            if k.startswith('predictor.')
        }
        embedding_dim = encoder_state['mu_head.weight'].shape[0]
        predictor = FullParamPredictor(embedding_dim, n_conformations).to(device)
        predictor.load_state_dict(predictor_state, strict=True)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad = False

        print("✅ Pretrained predictor loaded and frozen")

    # ------------------------------------------------------------------
    # Optionally freeze embedding
    # ------------------------------------------------------------------
    if pretrained_embedding_path is not None or pretrained_full_model_path is not None:
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

    # ------------------------------------------------------------------
    # Load training checkpoint if resuming
    # ------------------------------------------------------------------
    if train_from_checkpoint:
        if not isinstance(model_state_dict, str):
            raise Warning("No model state dict specified! --model_state_dict is empty")
        print(f"Loading model parameters from {model_state_dict}")
        estimator.load_state_dict(torch.load(model_state_dict))

    estimator.to(device=device)

    estimator.train()
    print(f"✅ Model in training mode: {estimator.training}")

    return estimator, predictor


# =======================================================================================
# MAIN TRAINING FUNCTION
# =======================================================================================

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
    pretrained_full_model_path: Optional[str] = None,
    freeze_embedding: bool = True,
    use_differential_lr: bool = False,
    embedding_lr_factor: float = 0.01,
    real_data_mrc: Optional[str] = None,
    real_data_finetune_fraction: float = 0.0,
    stochastic: bool = False,
) -> None:
    """
    Train NLE model by simulating training data on the fly.
    Optionally fine-tune on real data at the end using pseudo-labels
    from a frozen pretrained predictor.

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
        pretrained_embedding_path (str, optional): Path to pretrained encoder weights
        pretrained_full_model_path (str, optional): Path to full pretrained model (encoder + predictor)
        freeze_embedding (bool, optional): If True, freeze embedding during training
        use_differential_lr (bool, optional): If True, use lower LR for embedding
        embedding_lr_factor (float, optional): LR multiplier for embedding (if not frozen)
        real_data_mrc (str, optional): Path to .mrc stack of real images
        real_data_finetune_fraction (float, optional): Fraction of final epochs to fine-tune
                                                       on real data. Defaults to 0.0 (disabled).
        stochastic (bool, optional): If True, sample pseudo-labels from predictor probabilities.
                                     If False, use argmax.
    """
    assert 0.0 <= real_data_finetune_fraction <= 1.0, "real_data_finetune_fraction must be in [0, 1]"

    train_config = json.load(open(train_config))
    check_train_params(train_config)
    image_config = json.load(open(image_config))

    split_epoch = int(epochs * (1.0 - real_data_finetune_fraction))

    assert simulation_batch_size >= train_config["BATCH_SIZE"]
    assert simulation_batch_size % train_config["BATCH_SIZE"] == 0
    assert split_epoch >= 0, "real_data_finetune_fraction cannot exceed 1.0"

    if real_data_finetune_fraction > 0 and real_data_mrc is None:
        raise ValueError(
            "real_data_mrc must be provided when real_data_finetune_fraction > 0."
        )

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

    estimator, predictor = load_model(
        train_config,
        model_state_dict,
        device,
        train_from_checkpoint,
        pretrained_embedding_path=pretrained_embedding_path,
        pretrained_full_model_path=pretrained_full_model_path,
        freeze_embedding=freeze_embedding,
        image_size=image_config["N_PIXELS"],
        n_conformations=n_models,
    )

    if real_data_finetune_fraction > 0 and predictor is None:
        raise ValueError(
            "Real-data fine-tuning requires a full pretrained model containing a predictor. "
            "Use --pretrained_full_model_path."
        )

    # ------------------------------------------------------------------
    # Real data loader for fine-tuning phase
    # ------------------------------------------------------------------
    real_data_loader = None
    real_data_iter = None

    if real_data_finetune_fraction > 0:
        print(f"\n{'='*70}")
        print("SETTING UP REAL DATA FINE-TUNING")
        print(f"{'='*70}")
        print(f"  Real data MRC:       {real_data_mrc}")
        print(f"  Fine-tuning epochs:  {epochs - split_epoch}")
        print(f"  Starts at epoch:     {split_epoch}")
        print(f"  Stochastic labels:   {stochastic}")
        print(f"{'='*70}\n")

        real_dataset = RealImageMRCDataset(real_data_mrc)
        real_data_loader = DataLoader(
            real_dataset,
            batch_size=train_config["BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        real_data_iter = iter(real_data_loader)

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

    elif use_differential_lr and (pretrained_embedding_path is not None or pretrained_full_model_path is not None):
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

            if epoch >= split_epoch and real_data_finetune_fraction > 0:
                # ----------------------------------------------------------
                # FINE-TUNING PHASE: real data with pseudo-labels
                # ----------------------------------------------------------
                tq.set_description("Fine-tuning (Real Data)")

                for _ in range(n_batches_per_epoch):
                    try:
                        real_images_batch = next(real_data_iter)
                    except StopIteration:
                        real_data_iter = iter(real_data_loader)
                        real_images_batch = next(real_data_iter)

                    real_images_batch = real_images_batch.to(device, non_blocking=True)

                    # Assign pseudo-labels with frozen predictor
                    with torch.no_grad():
                        embeddings = estimator.embedding(real_images_batch)
                        pred_dict = predictor(embeddings)
                        real_logits = pred_dict["conf"]

                        if stochastic:
                            probs = F.softmax(real_logits, dim=-1)
                            inferred_indices = torch.multinomial(probs, num_samples=1)
                        else:
                            inferred_indices = torch.argmax(real_logits, dim=-1).unsqueeze(-1)

                    losses.append(
                        step(
                            loss(
                                real_images_batch,
                                inferred_indices.to(device, non_blocking=True)
                            )
                        )
                    )

            else:
                # ----------------------------------------------------------
                # STANDARD PHASE: synthetic data only
                # ----------------------------------------------------------
                tq.set_description("Training (Simulated Data)")

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

            postfix_dict = {
                'loss': mean_train_loss,
                'lr': scheduler.get_last_lr()[0],
            }
            if real_data_finetune_fraction > 0:
                postfix_dict['phase'] = 'real' if epoch >= split_epoch else 'syn'

            tq.set_postfix(postfix_dict)

            # save model checkpoint
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

            # scheduler step
            scheduler.step()

    # save final stuff
    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)
