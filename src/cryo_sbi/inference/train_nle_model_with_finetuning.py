from typing import Tuple, Dict, Union, Optional, List
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
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
# EXPONENTIAL MOVING AVERAGE
# =======================================================================================

class EMA:
    """
    Exponential moving average of model parameters.
    Only trainable parameters are shadowed; buffers are left unchanged.
    """
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        start_step: int = 0,
    ):
        self.decay = decay
        self.start_step = start_step
        self.num_updates = 0

        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1

        if self.num_updates < self.start_step:
            for name, param in model.named_parameters():
                self.shadow[name].copy_(param.data)
            return

        decay = self.decay
        for name, param in model.named_parameters():
            self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.decay = state_dict["decay"]
        self.start_step = state_dict.get("start_step", 0)
        self.num_updates = state_dict["num_updates"]
        self.shadow = state_dict["shadow"]


# =======================================================================================
# MODEL LOADING
# =======================================================================================

def load_model(
    train_config: str,
    model_state_dict: str,
    device: str,
    train_from_checkpoint: bool,
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = False,
    image_size: int = 128,
) -> torch.nn.Module:

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


# =======================================================================================
# MAIN TRAINING FUNCTION
# =======================================================================================

def nle_train_no_saving_with_finetuning(
    image_config: str,
    train_config: str,
    epochs: int,
    estimator_file: str,
    loss_file: str,
    log_file: str,
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
    embedding_lr_factor: float = 0.01,
    real_data_mrc: Optional[str] = None,
    beta_real: float = 0.0,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    ema_start_step: int = 0,
    ema_save_both: bool = False,
) -> None:

    train_config = json.load(open(train_config))
    check_train_params(train_config)
    image_config = json.load(open(image_config))

    assert simulation_batch_size >= train_config["BATCH_SIZE"]
    assert simulation_batch_size % train_config["BATCH_SIZE"] == 0

    if beta_real > 0 and real_data_mrc is None:
        raise ValueError("real_data_mrc must be provided when beta_real > 0.")

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
        image_size=image_config["N_PIXELS"],
    )

    # ------------------------------------------------------------------
    # Real data loader for marginal likelihood regularization
    # ------------------------------------------------------------------
    real_data_loader = None
    real_data_iter = None

    if beta_real > 0:
        print(f"\n{'='*70}")
        print("SETTING UP REAL DATA REGULARIZATION")
        print(f"{'='*70}")
        print(f"  Real data MRC:  {real_data_mrc}")
        print(f"  beta_real:      {beta_real}")
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

    warmup_epochs = max(1, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    # ------------------------------------------------------------------
    # EMA setup
    # ------------------------------------------------------------------
    ema = None
    if use_ema:
        ema = EMA(
            estimator,
            decay=ema_decay,
            start_step=ema_start_step,
        )
        print("\n" + "="*70)
        print("EMA ENABLED")
        print("="*70)
        print(f"  Decay:       {ema_decay}")
        print(f"  Start step:  {ema_start_step}")
        print(f"  Save both:   {ema_save_both}")
        print("="*70 + "\n")

        if train_from_checkpoint and model_state_dict is not None:
            ema_path = str(Path(model_state_dict).with_suffix('.ema'))
            if Path(ema_path).is_file():
                ema.load_state_dict(torch.load(ema_path, map_location='cpu'))
                for name, param in estimator.named_parameters():
                    if name in ema.shadow:
                        ema.shadow[name] = ema.shadow[name].to(param.device)
                print(f"  Resumed EMA state from {ema_path}")

    # ------------------------------------------------------------------
    # Checkpoint saving helper
    # ------------------------------------------------------------------
    def _save_checkpoint(path: str) -> None:
        if ema is not None:
            ema.apply_shadow(estimator)
            torch.save(estimator.state_dict(), path)
            torch.save(ema.state_dict(), path + ".ema")
            if ema_save_both:
                ema.restore(estimator)
                torch.save(estimator.state_dict(), path + "_non_ema")
            else:
                ema.restore(estimator)
        else:
            torch.save(estimator.state_dict(), path)

    # ------------------------------------------------------------------
    # Log writing helper
    # ------------------------------------------------------------------
    def _write_log_entry(
        epoch: int,
        lr: float,
        mean_total: float,
        mean_synthetic: float,
        mean_real: Optional[float],
    ) -> None:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Epoch {epoch:>6d} | LR: {lr:.6e}\n")
            f.write(f"  Total loss:     {mean_total:.6f}\n")
            f.write(f"  Synthetic loss: {mean_synthetic:.6f}\n")
            if mean_real is not None:
                f.write(f"  Real loss:      {mean_real:.6f}\n")
            f.write(f"{'='*60}\n")

    # Initialise log file with a header
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING LOG\n")
        f.write(f"  epochs:            {epochs}\n")
        f.write(f"  beta_real:         {beta_real}\n")
        f.write(f"  real_data_mrc:     {real_data_mrc}\n")
        f.write(f"  freeze_embedding:  {freeze_embedding}\n")
        f.write(f"  use_ema:           {use_ema}\n")
        f.write("="*60 + "\n")

    # ------------------------------------------------------------------
    # Loss history
    # ------------------------------------------------------------------
    history: Dict[str, List] = {
        'epoch':           [],
        'total_loss':      [],
        'synthetic_loss':  [],
        'real_loss':       [],   # None entries when beta_real == 0
        'lr':              [],
    }

    print("Training neural network:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = []
            synthetic_losses_epoch: List[float] = []
            real_losses_epoch: List[float] = []
            tq.set_description("Training")

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
                    synthetic_nll = loss(
                        _images.to(device, non_blocking=True),
                        _indices.to(device, non_blocking=True)
                    )
                    synthetic_losses_epoch.append(synthetic_nll.item())

                    if beta_real > 0 and real_data_iter is not None:
                        try:
                            real_batch = next(real_data_iter)
                        except StopIteration:
                            real_data_iter = iter(real_data_loader)
                            real_batch = next(real_data_iter)

                        real_batch = real_batch.to(device, non_blocking=True)
                        B = real_batch.shape[0]

                        # each real image repeated n_models times: [B*n_models, H, W]
                        real_exp = real_batch.repeat_interleave(n_models, dim=0)
                        # conformation indices 0..n_models-1 for each image: [B*n_models]
                        conf_idx = torch.arange(n_models, device=device).repeat(B)

                        # log p(d_real | X_i) for all i and all images in batch
                        log_p = estimator(real_exp, conf_idx)
                        # log sum_i p(d_real | X_i) per image, then mean over batch
                        log_marginal = torch.logsumexp(log_p.reshape(B, n_models), dim=1) - math.log(n_models)
                        real_reg = -log_marginal.mean()
                        real_losses_epoch.append(real_reg.item())

                        total_loss = synthetic_nll + beta_real * real_reg
                    else:
                        total_loss = synthetic_nll

                    losses.append(step(total_loss))
                    if ema is not None:
                        ema.update(estimator)

            losses = torch.stack(losses)
            mean_train_loss = losses.mean().item()
            mean_loss.append(mean_train_loss)

            mean_synthetic = float(np.mean(synthetic_losses_epoch))
            mean_real = float(np.mean(real_losses_epoch)) if real_losses_epoch else None
            current_lr = scheduler.get_last_lr()[0]

            # Update history
            history['epoch'].append(epoch)
            history['total_loss'].append(mean_train_loss)
            history['synthetic_loss'].append(mean_synthetic)
            history['real_loss'].append(mean_real)
            history['lr'].append(current_lr)

            postfix_dict = {
                'loss': mean_train_loss,
                'syn':  mean_synthetic,
                'lr':   current_lr,
            }
            if mean_real is not None:
                postfix_dict['real'] = mean_real
            if ema is not None:
                postfix_dict['ema_updates'] = ema.num_updates

            tq.set_postfix(postfix_dict)

            if epoch % saving_frequency == 0:
                _save_checkpoint(estimator_file + f"_epoch={epoch}")
                _write_log_entry(epoch, current_lr, mean_train_loss, mean_synthetic, mean_real)

            scheduler.step()

    _save_checkpoint(estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)

    # Write full history to log file
    with open(log_file, 'a') as f:
        f.write("\n\n" + "="*60 + "\n")
        f.write("TRAINING COMPLETE - FULL HISTORY\n")
        f.write("="*60 + "\n")
        json.dump(history, f, indent=2)
        f.write("\n")
