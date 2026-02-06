# "train_nle_model_with_validation.py"
from typing import Union, Optional
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

from cryo_sbi.inference.priors import get_image_priors, PriorLoader
from cryo_sbi.inference.models.build_models import build_nle_flow_model
from cryo_sbi.inference.validate_train_config import check_train_params
from cryo_sbi.wpa_simulator.cryo_em_simulator import cryo_em_simulator, create_simulation_param
from cryo_sbi.wpa_simulator.validate_image_config import check_image_params
import cryo_sbi.utils.image_utils as img_utils

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not installed. Real image loading for validation disabled.")

# =======================================================================================
# ROBUST MRC FILE HANDLING
# =======================================================================================

def check_mrc_file_size(filepath):
    """Check MRC file size in bytes and GB."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    file_size_gb = file_size / (1024**3)
    return file_size, file_size_gb

def validate_mrc_data(data):
    """Validate MRC data after reading."""
    if data is None or data.size == 0 or data.ndim not in [2, 3]:
        return False, f"Invalid data shape or type: {data.shape if hasattr(data, 'shape') else 'None'}"
    try:
        test_data = data[0] if data.ndim == 3 else data
        if np.all(test_data == 0): return False, "All data is zero"
        if np.any(np.isnan(test_data)): return False, "Data contains NaN"
        if np.any(np.isinf(test_data)): return False, "Data contains inf"
        if np.std(test_data) == 0: return False, "Zero variance"
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def read_mrc_header_raw(filepath):
    """Read MRC header manually."""
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(1024)
            if len(header_bytes) < 1024: return None
            import struct
            nx, ny, nz = struct.unpack('iii', header_bytes[0:12])
            mode = struct.unpack('i', header_bytes[12:16])[0]
            return {'nx': nx, 'ny': ny, 'nz': nz, 'mode': mode}
    except:
        return None

def get_dtype_from_mode(mode):
    """Convert MRC mode to numpy dtype."""
    dtype_map = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    return dtype_map.get(mode, np.float32)

def validate_mrc_dimensions(nx, ny, nz):
    """Check if dimensions are reasonable."""
    if nx <= 0 or ny <= 0 or nz <= 0: return False, f"Non-positive: {nz}×{ny}×{nx}"
    if nx > 8192 or ny > 8192: return False, f"Too large: {ny}×{nx}"
    return True, "Valid"

@contextmanager
def open_mrc_memmap(filepath):
    """Context manager for opening MRC as memmap (never loads into RAM)."""
    filepath = Path(filepath)
    memmap_obj = None
    try:
        if not filepath.exists():
            yield None, False, "File not found"
            return
        try:
            with mrcfile.open(filepath, permissive=True, mode='r') as mrc:
                nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
                dtype = mrc.data.dtype
            memmap_obj = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            if validate_mrc_data(memmap_obj)[0]:
                yield memmap_obj, True, "Memmap via mrcfile header"
                return
        except Exception:
            pass
        header_info = read_mrc_header_raw(filepath)
        if header_info:
            nx, ny, nz, mode = header_info['nx'], header_info['ny'], header_info['nz'], header_info['mode']
            if not validate_mrc_dimensions(nx, ny, nz)[0]:
                yield None, False, f"Invalid dimensions from raw header: {nz}x{ny}x{nx}"
                return
            dtype = get_dtype_from_mode(mode)
            memmap_obj = np.memmap(filepath, dtype=dtype, mode='r', offset=1024, shape=(nz, ny, nx))
            if validate_mrc_data(memmap_obj)[0]:
                yield memmap_obj, True, "Memmap via manual header read"
                return
        yield None, False, "All MRC opening methods failed"
    finally:
        if memmap_obj is not None: del memmap_obj
        gc.collect()


class RealImageMRCDataset(Dataset):
    """
    Dataset for loading real images from MRC stack
    Efficient streaming without loading all into memory
    """
    def __init__(self, mrc_path, cache_size=10000):
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile not installed. Install with: pip install mrcfile")
        
        self.mrc_path = mrc_path
        self.cache_size = cache_size
        
        print(f"  Opening MRC file: {mrc_path}")
        self._mrc_context = open_mrc_memmap(mrc_path)
        self.mrc_data, success, method = self._mrc_context.__enter__()

        if not success:
            raise RuntimeError(f"Failed to open MRC file: {method}")
        
        self.n_images = self.mrc_data.shape[0]
        self.image_shape = self.mrc_data.shape[1:]
        
        print(f"  Loaded MRC: {self.n_images} images of shape {self.image_shape}")
        print(f"  Loading method: {method}")
        
        self.cache = {}
        self.cache_order = []
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        img = self.mrc_data[idx].astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        # Use torch.from_numpy and .copy() for safety with multiprocessing
        self.cache[idx] = torch.from_numpy(img.copy())
        self.cache_order.append(idx)
        
        return self.cache[idx]
    
    def __del__(self):
        if hasattr(self, '_mrc_context'):
            self._mrc_context.__exit__(None, None, None)


def generate_real_validation_set(mrc_path: str, val_size: int, num_workers: int, device):
    """
    Extract a fixed set of real validation images.
    """
    print(f"\nExtracting {val_size} real validation images...")
 
    dataset = RealImageMRCDataset(mrc_path)

    if val_size > len(dataset):
        print(f"  Warning: Requested {val_size} images, but only {len(dataset)} are available. Using {len(dataset)}.")
        val_size = len(dataset)

    dataloader = DataLoader(
        dataset, batch_size=val_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    print(f"  Extracting one batch of {val_size} real images for validation.")
 
    real_images = next(iter(dataloader)).to(device, non_blocking=True)

    val_mem_gb = real_images.nelement() * real_images.element_size() / 1024**3

    print(f"✅ Validation set created with {len(real_images)} real images, consuming {val_mem_gb:.2f} GB of VRAM.")
    return real_images


def generate_synthetic_validation_set(prior_loader, models, simulation_param, val_size, device):
    """
    Generates a fixed set of synthetic validation images.
    """
    print(f"\nGenerating {val_size} synthetic validation images...")
    
    val_images = []
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
            
            val_images.append(images)
            generated_count += len(images)
            pbar.update(len(images))

    all_images = torch.cat(val_images, dim=0)[:val_size]
    val_mem_gb = all_images.nelement() * all_images.element_size() / 1024**3

    print(f"✅ Validation set created with {len(all_images)} synthetic images, consuming {val_mem_gb:.2f} GB of VRAM.")
    return all_images


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


def nle_train_no_saving_with_validation(
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
    validation_mrc_path: Optional[str] = None,
    validation_loss_file: str = 'validation_loss.pt',
    n_validation_images: int = 10240,
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

    simulation_param = create_simulation_param(image_config, models, device=device)
 
    estimator = load_model(
        train_config,
        model_state_dict,
        device,
        train_from_checkpoint,
        pretrained_embedding_path=pretrained_embedding_path,
        freeze_embedding=freeze_embedding,
        image_size = image_config["N_PIXELS"]
    )
 
    if validation_mrc_path:
        print("\n--- Setting up validation ---")
        try:
            real_val_images = generate_real_validation_set(
                validation_mrc_path,
                n_validation_images,
                n_workers,
                device
            )
            syn_val_images = generate_synthetic_validation_set(
                prior_loader, models, simulation_param, n_validation_images, device
            )

        except Exception as e:
            print(f"Warning: Could not create validation set: {e}. Training without validation.")
            validation_mrc_path = None # Disable validation if setup fails
        print("---------------------------\n")

    loss = NPELoss(estimator)

    if freeze_embedding:
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
    # validation_losses = [] # Will be used later

    print("Training neural network:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = []
            # islice(prior_loader, 100) sets 100 simulation batches per epoch
            for parameters in islice(prior_loader, 100):
                (
                    indices, quaternions, shift, defocus,
                    b_factor, amp, snr,
                ) = parameters
                
                # BUGFIX: Pass full simulation_param dict, not just the "noise" key
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
            mean_train_loss = losses.mean().item()
            mean_loss.append(mean_train_loss)

            # TODO: Add validation step here using real_val_images and syn_val_images
            # if validation_mrc_path:
            #     ...
            
            tq.set_postfix(loss=mean_train_loss)

            if epoch > 0 and epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)

    # if validation_losses:
    #    torch.save(torch.tensor(validation_losses), validation_loss_file)
