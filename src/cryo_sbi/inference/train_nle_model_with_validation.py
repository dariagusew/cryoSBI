from typing import Union, Optional
import json
import torch
import numpy as np
import torch.optim as optim
# New imports required for validation functionality
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
# End of new imports
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

# === NEW: MRC FILE HANDLING AND VALIDATION CODE (self-contained block) ===
try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not installed. Real image loading for validation disabled.")

def open_mrc_robust(filepath: Path):
    """Robustly open an MRC file, trying standard and permissive modes."""
    try:
        with mrcfile.open(filepath, mode='r') as mrc:
            return mrc.data, True, "Standard"
    except Exception:
        try:
            with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
                if mrc.data is not None and mrc.data.size > 0:
                    return mrc.data, True, "Permissive"
        except Exception as e:
            return None, False, f"Failed: {str(e)}"
    return None, False, "All methods failed"

class RealImageMRCDataset(Dataset):
    """Dataset for loading and normalizing real images from an MRC stack."""
    def __init__(self, mrc_path: str):
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile not installed. Cannot load real images.")

        self.mrc_path = Path(mrc_path)
        data, success, method = open_mrc_robust(self.mrc_path)
        if not success:
            raise RuntimeError(f"Failed to open MRC file '{mrc_path}': {method}")

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        
        self.mrc_data = torch.from_numpy(data.copy()).to(torch.float32)
        if self.mrc_data.dim() == 3:
            self.mrc_data = self.mrc_data.unsqueeze(1)
            
        self.n_images = self.mrc_data.shape[0]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img = self.mrc_data[idx]
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img

def create_real_image_loader(mrc_path: str, batch_size: int, num_workers: int):
    """Creates a DataLoader for real images from an MRC file."""
    dataset = RealImageMRCDataset(mrc_path)
    print(f"  Loaded {len(dataset)} real images for validation.")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return dataloader

@torch.no_grad()
def run_validation_step(
    estimator: torch.nn.Module,
    loss_fn: torch.nn.Module,
    validation_loader: DataLoader,
    prior_loader: PriorLoader,
    device: str
) -> Optional[float]:
    """Calculates validation loss on real images with random parameters."""
    estimator.eval()
    try:
        real_images_batch = next(iter(validation_loader)).to(device, non_blocking=True)
        (indices, _, _, _, _, _, _,) = next(iter(prior_loader))
        
        num_samples = min(real_images_batch.shape[0], indices.shape[0])
        real_images_batch = real_images_batch[:num_samples]
        indices = indices[:num_samples].to(device, non_blocking=True)
        
        val_loss = loss_fn(real_images_batch, indices)
        return val_loss.item()
    except StopIteration:
        return None
    finally:
        estimator.train()
# === END OF NEW CODE BLOCK ===


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


# Modified nle_train_no_saving function signature and docstring:
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
    saving_frequency: int = 100,
    simulation_batch_size: int = 2048,
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = False,
    use_differential_lr: bool = False,
    embedding_lr_factor: float = 0.01,
    # New arguments for validation
    validation_mrc_path: Optional[str] = None,
    validation_loss_file: str = 'validation_loss.pt',
    n_validation_images: int = 2048,
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
        validation_mrc_path (str, optional): Path to .mrc file for validation.
        validation_loss_file (str, optional): Path to save validation loss history.
        n_validation_images (int, optional): Number of real images for validation loss.

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

    # New: Setup validation loader
    validation_loader = None
    if validation_mrc_path:
        print("\n--- Setting up validation ---")
        try:
            validation_loader = create_real_image_loader(
                validation_mrc_path,
                batch_size=n_validation_images,
                num_workers=n_workers
            )
        except Exception as e:
            print(f"Warning: Could not create validation loader: {e}. Training without validation.")
            validation_loader = None
        print("---------------------------\n")

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
    validation_losses = [] # New: list to store validation losses

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
                    simulation_param 
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
            
            # Modified: store mean train loss and update progress bar
            mean_train_loss = losses.mean().item()
            mean_loss.append(mean_train_loss)

            # New: Perform validation at the end of the epoch
            if validation_loader:
                val_loss = run_validation_step(estimator, loss, validation_loader, prior_loader, device)
                if val_loss is not None:
                    validation_losses.append(val_loss)
                    tq.set_postfix(train_loss=mean_train_loss, val_loss=val_loss)
                else:
                    tq.set_postfix(loss=mean_train_loss) # Fallback if validation fails
            else:
                tq.set_postfix(loss=mean_train_loss)


            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)

    # New: Save validation losses if they were calculated
    if validation_losses:
        torch.save(torch.tensor(validation_losses), validation_loss_file)
