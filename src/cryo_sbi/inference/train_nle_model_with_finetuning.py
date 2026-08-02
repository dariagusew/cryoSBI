from typing import Tuple, Dict, Union, Optional, List
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
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
# FULL PARAM PREDICTOR
# =======================================================================================

class FullParamPredictor(nn.Module):
    """
    Predicts all parameters (X, θ) from z.
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
# GPU REAL IMAGE PRE-LOADER
# =======================================================================================

def load_real_images_to_gpu(
    mrc_path: str,
    n_images: int,
    device: str,
) -> torch.Tensor:
    if not MRCFILE_AVAILABLE:
        raise ImportError("mrcfile not installed. Install with: pip install mrcfile")

    mrc_p = Path(mrc_path)
    if not mrc_p.exists():
        raise FileNotFoundError(f"MRC file not found: {mrc_path}")

    print(f"  Opening MRC file: {mrc_path}")
    with mrcfile.mmap(str(mrc_p), mode='r') as mrc:
        total_available = mrc.data.shape[0]
        actual_n = min(n_images, total_available)
        indices = np.random.choice(total_available, size=actual_n, replace=False)
        indices.sort()
        imgs_np = np.array(mrc.data[indices], dtype=np.float32)

    real_tensor = torch.from_numpy(imgs_np).to(device)
    if real_tensor.ndim == 3:
        real_tensor = real_tensor.unsqueeze(1)

    mean = real_tensor.mean(dim=(-1, -2), keepdim=True)
    std  = real_tensor.std(dim=(-1, -2), keepdim=True) + 1e-8
    real_tensor = (real_tensor - mean) / std

    mem_gb = (real_tensor.element_size() * real_tensor.nelement()) / (1024 ** 3)
    print(f"  Loaded MRC          : {actual_n} images of shape {tuple(real_tensor.shape[1:])}")
    print(f"  Allocated on GPU    : {device} ({mem_gb:.2f} GB)")

    return real_tensor


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
) -> Tuple[torch.nn.Module, Optional[Dict[str, torch.Tensor]]]:

    check_train_params(train_config)
    estimator = build_nle_flow_model(train_config, image_size)
    full_checkpoint_state = None

    add_jitter_scale = float(train_config.get("ADD_JITTER", 0.0))

    if pretrained_embedding_path is not None:
        print(f"\n{'='*70}")
        print("LOADING PRETRAINED EMBEDDING")
        print(f"{'='*70}")
        print(f"Loading from: {pretrained_embedding_path}")

        try:
            pretrained_state = torch.load(pretrained_embedding_path, map_location='cpu')
            full_checkpoint_state = pretrained_state

            if any(k.startswith("encoder.") for k in pretrained_state.keys()):
                encoder_state = {
                    k.replace("encoder.", ""): v
                    for k, v in pretrained_state.items()
                    if k.startswith("encoder.")
                }
            else:
                encoder_state = {
                    k: v for k, v in pretrained_state.items()
                    if not k.startswith("predictor.")
                }

            estimator.embedding.load_state_dict(encoder_state, strict=False)
            print("✅ Pretrained embedding loaded successfully")

            # Configure Latent Jitter Scale
            if add_jitter_scale > 0.0:
                sigma_emb = getattr(estimator.embedding, "sigma_emb", None)
                if sigma_emb is not None and sigma_emb.abs().sum() > 0:
                    estimator.embedding.jitter_scale.fill_(add_jitter_scale)
                    effective_mean_sigma = (add_jitter_scale * sigma_emb).mean().item()
                    print(f"  ✅ ADD_JITTER = {add_jitter_scale}: Latent jitter active (effective mean sigma = {effective_mean_sigma:.4f})")
                else:
                    raise ValueError(
                        f"ADD_JITTER = {add_jitter_scale} requested in train_config, but 'sigma_emb' buffer was not found or is zero in pretrained encoder checkpoint!"
                    )
            else:
                if hasattr(estimator.embedding, "jitter_scale"):
                    estimator.embedding.jitter_scale.zero_()
                print("  ℹ️  ADD_JITTER is 0.0: Latent jittering disabled during flow training.")

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

    return estimator, full_checkpoint_state

# =======================================================================================
# VALIDATION SCORE EVALUATOR
# =======================================================================================

def evaluate_real_validation_score(
    estimator: nn.Module,
    real_images: torch.Tensor,
    n_models: int,
    batch_size: int,
) -> float:
    estimator.eval()
    total_log_marginal = 0.0
    n_images = len(real_images)

    with torch.no_grad():
        for i in range(0, n_images, batch_size):
            batch = real_images[i : i + batch_size]
            B = batch.shape[0]

            real_exp = batch.repeat_interleave(n_models, dim=0)
            conf_idx = torch.arange(n_models, device=batch.device).repeat(B).unsqueeze(-1)

            log_p = estimator.forward_inference(real_exp, conf_idx)
            log_marginal = torch.logsumexp(log_p.reshape(B, n_models), dim=1) - math.log(n_models)
            total_log_marginal += log_marginal.sum().item()

    estimator.train()
    return total_log_marginal / n_images


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
    n_real_val: int = 3000,
    fraction_finetune_epochs: float = 0.0,
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

    # ------------------------------------------------------------------
    # Pre-load real images to GPU if MRC provided
    # ------------------------------------------------------------------
    val_real_tensor: Optional[torch.Tensor] = None
    if real_data_mrc is not None:
        val_real_tensor = load_real_images_to_gpu(
            mrc_path=real_data_mrc,
            n_images=n_real_val,
            device=device,
        )

    estimator, pretrained_state_dict = load_model(
        train_config,
        model_state_dict,
        device,
        train_from_checkpoint,
        pretrained_embedding_path=pretrained_embedding_path,
        freeze_embedding=freeze_embedding,
        image_size=image_config["N_PIXELS"],
    )

    # ------------------------------------------------------------------
    # Predictor check for fine-tuning phase
    # ------------------------------------------------------------------
    finetune_start_epoch = epochs + 1
    predictor: Optional[FullParamPredictor] = None

    if fraction_finetune_epochs > 0.0:
        if real_data_mrc is None or val_real_tensor is None:
            raise ValueError("real_data_mrc must be provided when fraction_finetune_epochs > 0.")

        if pretrained_embedding_path is None or pretrained_state_dict is None:
            raise ValueError("Fine-tuning requires pretrained_embedding_path to be specified.")

        predictor_keys = [k for k in pretrained_state_dict.keys() if k.startswith("predictor.")]
        if not predictor_keys:
            raise ValueError(
                f"Pretrained checkpoint at '{pretrained_embedding_path}' does not contain predictor weights "
                f"(e.g. 'predictor.conf_head...'). A full model checkpoint is required for fine-tuning."
            )

        print(f"\n{'='*70}")
        print("SETTING UP FINE-TUNING PHASE")
        print(f"{'='*70}")
        finetune_start_epoch = int(epochs * (1.0 - fraction_finetune_epochs))
        print(f"  Fraction fine-tune epochs: {fraction_finetune_epochs:.2f}")
        print(f"  Fine-tuning active from epoch {finetune_start_epoch} to {epochs - 1}")

        predictor_state = {
            k.replace("predictor.", ""): v
            for k, v in pretrained_state_dict.items()
            if k.startswith("predictor.")
        }

        emb_dim = predictor_state["trunk.0.weight"].shape[1]
        n_conformations = predictor_state["conf_head.weight"].shape[0]

        predictor = FullParamPredictor(embedding_dim=emb_dim, n_conformations=n_conformations).to(device)
        predictor.load_state_dict(predictor_state)
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad = False

        print(f"  ✅ Predictor head successfully loaded (emb_dim={emb_dim}, n_conformations={n_conformations})")
        print(f"{'='*70}\n")

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
        val_score: Optional[float],
        is_finetuning: bool,
    ) -> None:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Epoch {epoch:>6d} | Mode: {'FINE-TUNING' if is_finetuning else 'STANDARD'} | LR: {lr:.6e}\n")
            f.write(f"  Train loss:        {mean_total:.6f}\n")
            if val_score is not None:
                f.write(f"  Val marginal score: {val_score:.6f}\n")
            f.write(f"{'='*60}\n")

    # Initialise log file with a header
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING LOG\n")
        f.write(f"  epochs:                   {epochs}\n")
        f.write(f"  real_data_mrc:            {real_data_mrc}\n")
        f.write(f"  fraction_finetune_epochs: {fraction_finetune_epochs}\n")
        f.write(f"  finetune_start_epoch:     {finetune_start_epoch}\n")
        f.write(f"  freeze_embedding:         {freeze_embedding}\n")
        f.write(f"  use_ema:                  {use_ema}\n")
        f.write("="*60 + "\n")

    # ------------------------------------------------------------------
    # Loss history
    # ------------------------------------------------------------------
    history: Dict[str, List] = {
        'epoch':               [],
        'train_loss':          [],
        'val_marginal_score':  [],
        'lr':                  [],
    }

    print("Training neural network:")
    estimator.train()
    bs = train_config["BATCH_SIZE"]

    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = []
            is_finetuning_epoch = epoch >= finetune_start_epoch

            if is_finetuning_epoch:
                tq.set_description("Fine-tuning")
                n_real_imgs = len(val_real_tensor)

                for i in range(0, n_real_imgs, bs):
                    real_batch = val_real_tensor[i : i + bs]

                    with torch.no_grad():
                        out = estimator.embedding.forward_inference(real_batch)
                        z = out[0] if isinstance(out, tuple) else out
                        conf_logits = predictor(z)["conf"]
                        assigned_indices = torch.argmax(conf_logits, dim=-1, keepdim=True)

                    finetune_nll = loss(real_batch, assigned_indices)
                    losses.append(step(finetune_nll))

                    if ema is not None:
                        ema.update(estimator)

            else:
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
                        indices.split(bs),
                        noisy_images.split(bs),
                    ):
                        synthetic_nll = loss(
                            _images.to(device, non_blocking=True),
                            _indices.to(device, non_blocking=True)
                        )
                        losses.append(step(synthetic_nll))

                        if ema is not None:
                            ema.update(estimator)

            losses = torch.stack(losses)
            mean_train_loss = losses.mean().item()
            mean_loss.append(mean_train_loss)

            current_lr = scheduler.get_last_lr()[0]

            # ----------------------------------------------------------
            # Validation Score Calculation
            # ----------------------------------------------------------
            val_score: Optional[float] = None
            if (epoch % saving_frequency == 0) or (epoch == epochs - 1) or (epoch == 0):
                if val_real_tensor is not None:
                    if ema is not None:
                        ema.apply_shadow(estimator)

                    val_score = evaluate_real_validation_score(
                        estimator=estimator,
                        real_images=val_real_tensor,
                        n_models=n_models,
                        batch_size=bs,
                    )

                    if ema is not None:
                        ema.restore(estimator)

                    print(f"\n  Epoch {epoch:3d} Validation Marginal Score: {val_score:.6f}")

            # Update history
            history['epoch'].append(epoch)
            history['train_loss'].append(mean_train_loss)
            history['val_marginal_score'].append(val_score)
            history['lr'].append(current_lr)

            postfix_dict = {
                'loss': mean_train_loss,
                'lr':   current_lr,
            }
            if val_score is not None:
                postfix_dict['val_score'] = val_score
            if ema is not None:
                postfix_dict['ema_updates'] = ema.num_updates

            tq.set_postfix(postfix_dict)

            if epoch % saving_frequency == 0:
                _save_checkpoint(estimator_file + f"_epoch={epoch}")
                _write_log_entry(epoch, current_lr, mean_train_loss, val_score, is_finetuning_epoch)

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
