# "train_nle_model_with_validation.py"
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


def generate_real_validation_set(mrc_path: str, val_size: int, device):
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
        num_workers=0, pin_memory=True, drop_last=True
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


# =======================================================================================
# VALIDATION METRICS
# =======================================================================================

@torch.no_grad()
def get_per_image_scores(
    estimator: torch.nn.Module,
    images: torch.Tensor,
    n_models: int,
    batch_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to get per-image predictions, APE, and NAMLL scores.
    """
    all_preds, all_apes, all_namlls = [], [], []
    # batch images
    for batch in images.split(batch_size):
        # Get log-likelihoods
        log_probs = []
        for i in range(n_models):
            indices_i = torch.full((batch.shape[0], 1), i, device=batch.device)
            log_probs.append(estimator(batch, indices_i).unsqueeze(-1))
        log_probs = torch.cat(log_probs, dim=-1)

        # Calculate NAMLL per image
        namlls = -torch.logsumexp(log_probs, dim=-1)
        all_namlls.append(namlls)

        # Calculate APE per image
        log_posterior = torch.log_softmax(log_probs, dim=-1)
        apes = -torch.sum(torch.exp(log_posterior) * log_posterior, dim=-1)
        all_apes.append(apes)

    return torch.cat(all_apes), torch.cat(all_namlls)


@torch.no_grad()
def calculate_raw_metrics(
    estimator: torch.nn.Module,
    real_images: torch.Tensor,
    sim_images: torch.Tensor,
    n_models: int,
    batch_size: int = 256
) -> Dict[str, float]:
    """
    Calculates the raw APE and NAMLL scores for both real and synthetic domains.

    Returns a dictionary with four key metrics:
    - ape_real: Average Posterior Entropy on Real data.
    - ape_sim: Average Posterior Entropy on Synthetic data.
    - namll_real: Negative AMLL on Real data.
    - namll_sim: Negative AMLL on Synthetic data.
    """

    print("\nCalculating raw validation metrics...")
    # Step 1: Get per-image scores for all images
    print("  Processing real images...")
    real_apes, real_namlls = get_per_image_scores(estimator, real_images, n_models, batch_size)
    print("  Processing synthetic images...")
    sim_apes, sim_namlls = get_per_image_scores(estimator, sim_images, n_models, batch_size)

    # Step 2: Calculate averages
    ape_real   = torch.mean(real_apes).item()
    ape_sim    = torch.mean(sim_apes).item()
    namll_real = torch.mean(real_namlls).item()
    namll_sim  = torch.mean(sim_namlls).item()

    metrics = {
        'ape_R':  ape_real,
        'ape_S':  ape_sim,
        'amll_R': namll_real,
        'amll_S': namll_sim
    }
    return metrics

@torch.no_grad()
def calculate_apes(
    estimator: torch.nn.Module,
    images: torch.Tensor,
    n_models: int
) -> torch.Tensor:
    """
    Efficiently calculates the Average Posterior Entropy (APE) for a batch of 2D images.

    Args:
        estimator (torch.nn.Module): The trained NLE model.
        images (torch.Tensor): A batch of 2D images, of shape
                               (batch_size, height, width).
        n_models (int): The total number of conformational classes.

    Returns:
        torch.Tensor: A 1D tensor of APE scores of shape (batch_size,),
                      moved to the CPU.
    """
    batch_size = images.shape[0]
    device = images.device

    # 1. Prepare inputs for a single, vectorized forward pass.
    repeated_images = images.repeat_interleave(n_models, dim=0)

    # Create a corresponding tensor of model indices.
    repeated_indices = torch.arange(n_models, device=device).repeat(batch_size).unsqueeze(-1)

    # 2. Perform a single forward pass to get log-likelihoods for all pairs.
    # The output `log_probs_flat` will have shape (batch_size * n_models,).
    log_probs_flat = estimator(repeated_images, repeated_indices)

    # 3. Reshape the log-likelihoods to group them by the original image.
    # Shape becomes: (batch_size, n_models)
    log_probs = log_probs_flat.view(batch_size, n_models)

    # 4. Calculate APE directly from the log-likelihoods.
    log_posterior = torch.log_softmax(log_probs, dim=-1)
    apes = -torch.sum(torch.exp(log_posterior) * log_posterior, dim=-1)

    return apes


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
    pretrained_embedding_path: Optional[str] = None,
    freeze_embedding: bool = True,
    use_differential_lr: bool = False,
    embedding_lr_factor: float = 0.01,
    validation_mrc_path: Optional[str] = None,
    validation_log_file: str = 'validation_scores.pt',
    n_validation_images: int = 10240,
    real_data_finetune_fraction: float = 0.0,
    sample_indices: bool = False
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
        validation_log_file (str, optional): Path to save validation loss history.
        n_validation_images (int, optional): Number of real images for validation loss.
        real_data_finetune_fraction (float, optional): Fraction of final epochs to fine-tune on real data. Defaults to 0.0 (disabled).
    """
    train_config = json.load(open(train_config))
    check_train_params(train_config)
    image_config = json.load(open(image_config))

    assert simulation_batch_size >= train_config["BATCH_SIZE"]
    assert simulation_batch_size % train_config["BATCH_SIZE"] == 0

    if real_data_finetune_fraction > 0 and not validation_mrc_path:
        raise ValueError("A `validation_mrc_path` must be provided to use real data fine-tuning.")
    
    split_epoch = int(epochs * (1.0 - real_data_finetune_fraction))

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
        image_size = image_config["N_PIXELS"]
    )


    if validation_mrc_path:
        print("\n--- Setting up validation ---")
        try:
            real_val_images = generate_real_validation_set(
                validation_mrc_path, n_validation_images, device
            )
            syn_val_images = generate_synthetic_validation_set(
                prior_loader, models, simulation_param, n_validation_images, device
            )
        except Exception as e:
            print(f"Warning: Could not create validation set: {e}. Training without validation.")
            validation_mrc_path = None
        print("---------------------------\n")

    loss = NPELoss(estimator)

    if freeze_embedding:
        print("\n" + "="*70)
        print("OPTIMIZER: TRAINING FLOW and THETA_EMBEDDING ONLY")
        print("="*70)
        print(f"Flow learning rate: {train_config['LEARNING_RATE']:.2e}")
        print(f"Theta embedding learning rate: {train_config['LEARNING_RATE']:.2e}")
        print("="*70 + "\n")

        optimizer = optim.AdamW([
            {
                'params': estimator.nle.parameters(),
                'lr': train_config["LEARNING_RATE"],
                'weight_decay': 0.001,
                'name': 'flow'
            },
            {
                'params': estimator.theta_embedding.parameters(),
                'lr': train_config["LEARNING_RATE"],
                'weight_decay': 0.001,
                'name': 'theta_embedding'
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

    # Store all validation metrics for later analysis
    validation_scores = {'ape_R': [], 'amll_R': [], 'ape_S': [], 'amll_S': []}

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # loader for real images
    real_train_loader = None
    if real_data_finetune_fraction > 0.0:
        print(f"\n--- Setting up real data loader for fine-tuning phase (starting epoch {split_epoch}) ---")
        if sample_indices:
          print(f"  With probabilitic model assignment")
        # define dataset from MRC
        real_train_dataset = RealImageMRCDataset(validation_mrc_path)
        # define loader
        real_train_loader = DataLoader(
            real_train_dataset,
            batch_size=train_config["BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        print("------------------------------------------------------------------------------------\n")

    print("Training neural network:")
    estimator.train()
    with tqdm(range(epochs), unit="epoch") as tq:
        for epoch in tq:
            losses = []

            if epoch >= split_epoch and real_data_finetune_fraction > 0.0:
                # PHASE 2: Fine-tuning on real data with pseudo-labels
                tq.set_description("Fine-tuning (Real Data)")

                # Create a frozen teacher
                if epoch == split_epoch:
                   print("\nFreezing conformational embedding parameters...")
                   for param in estimator.theta_embedding.parameters():
                        param.requires_grad = False

                print("\nCreating and freezing a static Teacher model for pseudo-labeling")
                estimator_teacher = copy.deepcopy(estimator)
                # Teacher is always in eval mode
                estimator_teacher.eval()
                # Explicitly disable gradient tracking for all teacher parameters
                for param in estimator_teacher.parameters():
                    param.requires_grad = False

                # Selecting high-confidence images
                print("\nSelecting high-confidence real images for fine-tuning...")
                all_image_apes = []
                with torch.no_grad():
                    for real_images_batch in tqdm(real_train_loader, desc="  Calculating APE_R for all images"):
                        batch_apes = calculate_apes(
                           estimator_teacher,
                           real_images_batch.to(device, non_blocking=True),
                           n_models
                        )
                        all_image_apes.append(batch_apes)

                all_image_apes = torch.cat(all_image_apes)

                # Determine the number of images to select
                N = 100*train_config["BATCH_SIZE"]
                print(f"  Sorting images by APE_R and selecting the top {N} most confident examples.")

                # Get the indices of the images with the lowest APE
                sorted_indices = torch.argsort(all_image_apes)
                top_N_indices = sorted_indices[:N]

                # Create a Subset using these specific indices
                finetune_dataset = Subset(real_train_dataset, top_N_indices.tolist())

                # Create the final loader from this curated subset, with shuffling
                fine_tune_loader = DataLoader(
                    finetune_dataset,
                    batch_size=train_config["BATCH_SIZE"],
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True
                )
                fine_tune_iter = iter(fine_tune_loader)
                print(f"✅ Created a new fine-tuning dataset with {len(finetune_dataset)} images.")

                for _ in range(100):
                    try:
                        real_images_batch = next(fine_tune_iter)
                    except StopIteration:
                        fine_tune_iter = iter(fine_tune_loader)
                        real_images_batch = next(fine_tune_iter)
                    
                    real_images_batch = real_images_batch.to(device, non_blocking=True)

                    # Find the class X that maximizes p(image | X)
                    with torch.no_grad():
                        log_probs = []
                        for i in range(n_models):
                            indices_i = torch.full((real_images_batch.shape[0], 1), float(i), device=device)
                            log_probs.append(estimator_teacher(real_images_batch, indices_i).unsqueeze(-1))
                        
                        log_probs_cat = torch.cat(log_probs, dim=-1)

                        # random assignment
                        if sample_indices:
                           probs = torch.softmax(log_probs_cat, dim=-1)
                           # Sample a class for each image based on the probability distribution
                           inferred_indices = torch.multinomial(probs, num_samples=1)
                        else:
                           # The inferred indices become our pseudo-labels
                           inferred_indices = torch.argmax(log_probs_cat, dim=-1).unsqueeze(-1)

                    # Calculate loss using real images and their pseudo-labels
                    losses.append(
                        step(
                           loss(real_images_batch, inferred_indices)
                        )
                    )
            else:
                # PHASE 1: Standard training on simulated data
                tq.set_description("Training (Simulated Data)")
                for parameters in islice(prior_loader, 100):
                    (
                        indices, quaternions, shift, defocus,
                        b_factor, amp, snr,
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


            # calculate mean loss across mini-batches
            losses = torch.stack(losses)
            mean_train_loss = losses.mean().item()
            # add to list
            mean_loss.append(mean_train_loss)
            # add to postfix
            postfix_dict = {'loss': mean_train_loss}
            # add current learning rate
            postfix_dict['lr'] = scheduler.get_last_lr()[0]

            # Validation metrics
            if validation_mrc_path and (epoch % saving_frequency == 0 or epoch == epochs - 1):
                # Set model to eval mode for validation
                estimator.eval()

                # Get all validation metrics
                metrics = calculate_raw_metrics(
                    estimator,
                    real_val_images,
                    syn_val_images,
                    n_models
                )
                # Append each metric to its corresponding list
                for key, value in metrics.items():
                    validation_scores[key].append(value)

                # Define variables for easy printout
                ape_R_score = metrics['ape_R']
                ape_S_score = metrics['ape_S']
                amll_R_score = metrics['amll_R']
                amll_S_score = metrics['amll_S']

                print(f"\nEpoch {epoch} | APE_R score: {ape_R_score:.4f} APE_S score: {ape_S_score:.4f} AMLL_R score: {amll_R_score:.4f} AMLL_S score: {amll_S_score:.4f}")
                
                # Set model back to train mode
                estimator.train()

                # define postfix_dict for this epoch
                postfix_dict['ape_R'] = ape_R_score
                postfix_dict['ape_S'] = ape_S_score
                postfix_dict['amll_R'] = amll_R_score
                postfix_dict['amll_S'] = amll_S_score

            # set postfix
            tq.set_postfix(postfix_dict)

            # save model checkpoint
            if epoch % saving_frequency == 0:
                torch.save(estimator.state_dict(), estimator_file + f"_epoch={epoch}")

            # scheduler step
            scheduler.step()

    # save final stuff
    torch.save(estimator.state_dict(), estimator_file)
    torch.save(torch.tensor(mean_loss), loss_file)

    # save validation scores - in case
    if validation_mrc_path:
        # Save the whole dictionary for detailed analysis
        torch.save(validation_scores, validation_log_file)
        print(f"\nValidation scores saved to {validation_log_file}")
