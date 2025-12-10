import torch 
import numpy as np 
from tqdm.auto import tqdm
from cryo_sbi import CryoEmSimulator
import cryo_sbi.utils.estimator_utils as est_utils


def center_models(models):
    """
    Remove center of mass from each model.
    
    Args:
        models: torch.Tensor of shape [num_models, 3, N]
                where 3 = (x, y, z) and N = number of atoms
    
    Returns:
        centered_models: torch.Tensor of same shape, centered at origin
    """
    # Compute center of mass for each model
    # Mean over atoms (dim=2) -> [num_models, 3]
    com = models.mean(dim=2, keepdim=True)  # [num_models, 3, 1]
    
    # Subtract center of mass
    centered_models = models - com
    
    return centered_models

def create_weighted_ensemble(models, w, precision=0.01, max_models=1000, verbose=True):
    """
    Create an ensemble of models by repeating them according to weights.
    Automatically calculates minimum number of repetitions to match weights within specified precision.
    
    Parameters:
        models: torch.Tensor of shape (n_models, ...)
        w: np.array or list of weights (length n_models)
        precision: float, desired precision for weight matching (default: 0.01 for 2 decimals)
        max_models: int, maximum allowed total models (safety limit)
        verbose: bool, whether to print summary
    
    Returns:
        models_ensemble: torch.Tensor of repeated models
        w_actual: np.array of actual weights achieved
        total_models: int, total number of models in ensemble
    """

    # Convert w to numpy array and normalize
    w = np.array(w, dtype=float)
    w = w / w.sum()
    
    # Get non-zero indices
    indices = np.where(w > 0)[0]
    active_weights = w[indices]
    n_active = len(indices)
    
    if n_active == 0:
        raise ValueError("All weights are zero!")
    
    # Special case: only one non-zero weight
    if n_active == 1:
        counts = np.array([1])
        total_models = 1
        w_actual = np.zeros_like(w)
        w_actual[indices[0]] = 1.0
    else:
        # Find minimum multiplier that achieves desired precision
        found = False
        for multiplier in range(1, max_models + 1):
            # Calculate counts
            counts = np.round(active_weights * multiplier).astype(int)
            
            # Skip if any count is 0 (would lose a model)
            if np.any(counts == 0):
                continue
            
            total_models = counts.sum()
            
            # Calculate actual weights from these counts
            actual_weights = counts / total_models
            
            # Check if precision is met for all active weights
            max_error = np.max(np.abs(actual_weights - active_weights))
            
            if max_error < precision:
                found = True
                break
        
        if not found:
            raise ValueError(f"Could not achieve precision {precision} within {max_models} models. "
                           f"Try increasing max_models or relaxing precision.")
        
        # Build full w_actual array
        w_actual = np.zeros_like(w)
        w_actual[indices] = actual_weights
    
    # Create models tensor with repetitions
    model_list = [models[idx].unsqueeze(0).repeat(count, *[1]*(models.dim()-1)) 
                  for idx, count in zip(indices, counts) if count > 0]
    
    # Concatenate to create ensemble
    models_ensemble = torch.cat(model_list, dim=0)
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"WEIGHTED ENSEMBLE CREATION (Precision: {precision})")
        print(f"{'='*60}")
        print(f"\nInput weights (w):")
        print(f"  Non-zero indices: {indices.tolist()}")
        print(f"  Target weights:   {[f'{w:.4f}' for w in active_weights]}")
        print(f"\nOptimal integer counts:")
        print(f"  Counts:           {counts.tolist()}")
        print(f"  Total models:     {total_models}")
        print(f"\nAchieved weights (w'):")
        print(f"  Actual weights:   {[f'{w:.4f}' for w in w_actual[indices]]}")
        print(f"{'='*60}\n")
    
    return models_ensemble, w_actual, total_models

def generate_population_weights(n_states=20, population_steps=11):
    """
    Generate population weights for mixtures of state 0 and each other state.
    Returns:
        weights: (n_points, n_states)
        pop_fractions: fraction of state 0 for each mixture
        state_pairs: list of tuples (0, i) for each mixture
    """
    fractions = np.linspace(0, 1, population_steps)
    weights_list = []
    pop_fractions = []
    state_pairs = []

    for i in range(1, n_states):
        for f in fractions:
            w = np.zeros(n_states)
            w[0] = f
            w[i] = 1 - f
            weights_list.append(w)
            pop_fractions.append(f)
            state_pairs.append((0, i))

    return np.array(weights_list), np.array(pop_fractions), state_pairs

def evaluate_likelihood_pairwise(
    estimator: torch.nn.Module,
    images: torch.Tensor,        # shape: [N_images, H, W]
    models: torch.Tensor,        # not needed, just for interface
    batch_size_pairs: int = 4096,  # Batch size for (image, model) pairs
    device: str = "cuda"
) -> torch.Tensor:
    """
    Evaluate log p(X_i | theta_j) for all pairs efficiently.
    Creates all pairs and processes in large batches.
    
    Returns:
        log_probs: shape [N_images, N_models]
    """
    estimator.eval()
    estimator.to(device)
    
    N_images = len(images)
    N_models = len(models)
    model_indices = torch.arange(N_models, dtype=torch.float32, device=device)
    
    # Total number of pairs
    total_pairs = N_images * N_models
    
    log_probs = torch.zeros(N_images, N_models, device=device)
    
    with torch.no_grad():
        print(f"Evaluating {total_pairs:,} pairs...")
        
        # Process all pairs in large batches
        for pair_start in tqdm(range(0, total_pairs, batch_size_pairs), desc="Pairs"):
            pair_end = min(pair_start + batch_size_pairs, total_pairs)
            batch_size = pair_end - pair_start
            
            # Compute which (image, model) each pair corresponds to
            pair_indices = torch.arange(pair_start, pair_end, device=device)
            img_idx = pair_indices // N_models  # [batch_size]
            mod_idx = pair_indices % N_models   # [batch_size]
            
            # Gather the corresponding images and model indices
            batch_images = images[img_idx.cpu()]  # [batch_size, H, W]
            batch_indices = model_indices[mod_idx].unsqueeze(-1)  # [batch_size, 1]
            
            # Single forward pass!
            log_p = estimator(batch_images.to(device), batch_indices)  # [batch_size]
            
            # Place results in the correct positions
            log_probs.view(-1)[pair_start:pair_end] = log_p
    
    return log_probs

class WeightOptimizer:
    """
    Numerically stable optimizer for:
    L = -sum_i log(sum_j w_j * p_ij) + theta * sum_j w_j * log(w_j / w_j^0)
    """
    
    def __init__(self, log_p, w0=None, theta=0.0, device='cpu'):
        """
        Args:
            log_p: tensor of shape (n_j, n_i) containing log(p_ij)
            w0: prior weights of shape (n_j,), normalized. If None, uses uniform weights.
            theta: regularization parameter (float), default 0.0
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.log_p = torch.tensor(log_p, dtype=torch.float64, device=device)
        self.n_j, self.n_i = self.log_p.shape
        
        # Initialize w0 as uniform if not provided - used only with theta>0
        if w0 is None:
            self.w0 = torch.ones(self.n_j, dtype=torch.float64, device=device) / self.n_j
            if(theta>0): print(f"Prior weights w0 not specified, using uniform: w0 = 1/{self.n_j}")
        else:
            self.w0 = torch.tensor(w0, dtype=torch.float64, device=device)
        
        self.theta = torch.tensor(theta, dtype=torch.float64, device=device)

    def compute_loss(self, w):
        """
        Compute loss given weights w (must be normalized, sum to 1, non-negative)
        Loss is normalized: average log-likelihood per sample and average KL per weight
        """
        eps = 1e-15
        
        # First term: -(1/n_i) * sum_i log(sum_j w_j * p_ij)
        log_w = torch.log(w + eps)  # shape (n_j,)
        log_terms = log_w.unsqueeze(1) + self.log_p  # shape (n_j, n_i)
        term1 = -torch.logsumexp(log_terms, dim=0).sum() / self.n_i  # average over samples
        
        # Second term: +theta * (1/n_j) * sum_j w_j * log(w_j / w_j^0)
        if self.theta > 0:
            log_w0 = torch.log(self.w0 + eps)
            term2 = self.theta * torch.sum(w * (log_w - log_w0)) / self.n_j  # average over weights
        else:
            term2 = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        return term1 + term2
        
    
    def optimize(self, lr=0.1, max_iter=10000, tol=1e-9, verbose=False):
        """
        Optimize weights using PyTorch Adam optimizers
        
        Args:
            lr: learning rate
            max_iter: maximum iterations
            tol: convergence tolerance
            verbose: print progress
        """
        # Use unconstrained parameterization: w = softmax(z)
        # Initialize with random normalized weights
        z_init = torch.randn(self.n_j, dtype=torch.float64, device=self.device)
        z = z_init.clone().detach().requires_grad_(True)
        
        # initialize Adam optimizer
        optimizer = torch.optim.Adam([z], lr=lr)
            
        losses = []
        for iteration in range(max_iter):
             optimizer.zero_grad()
                
             # Convert unconstrained z to normalized weights
             w = torch.softmax(z, dim=0)
                
             loss = self.compute_loss(w)
             loss.backward()
             optimizer.step()
                
             losses.append(loss.item())
                
             if verbose and iteration % 100 == 0:
                  print(f"Iter {iteration}: Loss = {loss.item():.6f}")
                
             # Check convergence
             if iteration > 10 and abs(losses[-1] - losses[-2]) < tol:
                 if verbose:
                     print(f"Converged at iteration {iteration}")
                 break
                    
        # Final weights
        with torch.no_grad():
            w_opt = torch.softmax(z, dim=0)
        
        return w_opt.cpu().numpy(), losses
    
def rmse(x_opt, w_actual):
    # Convert to numpy if needed
    if torch.is_tensor(x_opt):
        x_opt = x_opt.detach().cpu().numpy()
    if torch.is_tensor(w_actual):
        w_actual = w_actual.detach().cpu().numpy()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((x_opt - w_actual)**2))
    return rmse 


class PopulationOptimizer:
    def __init__(self, models, estimator, device, population_steps=11, num_sim=100000):
        """
        estimator: trained Cryo-EM estimator
        models: PyTorch tensor of models (n_models, 3, n_atoms)
        simulator: CryoEmSimulator instance
        device: torch device
        population_steps: how many fractions between 0-1 for state 0
        num_sim: number of images to simulate for each population
        """
        self.estimator = estimator
        self.models = models
        self.device = device
        self.population_steps = population_steps
        self.n_models = len(models)
        self.num_sim = num_sim
        
        # Generate population weights
        self.weights, self.pop_fractions, self.state_pairs = generate_population_weights(
            n_states=self.n_models,
            population_steps=self.population_steps
        )
        
        # Storage
        self.rmse_values = []
        self.pop_fraction_record = []

    def rmse(self, x_opt, w_actual):
        """Compute RMSE between optimized weights and target weights."""
        if torch.is_tensor(x_opt):
            x_opt = x_opt.detach().cpu().numpy()
        if torch.is_tensor(w_actual):
            w_actual = w_actual.detach().cpu().numpy()
        return np.sqrt(np.mean((x_opt - w_actual)**2))

    def run_for_all_populations(self, sim_config, batch_size, num_sim=None, verbose=False):
        """
        Loop over all generated population mixtures, simulate images, optimize weights,
        and compute RMSE relative to the target weights.
        """
        
        if num_sim is None:
            num_sim = self.num_sim  # use default from class

        for i, w in enumerate(self.weights):
            if verbose:
                print(f"Processing population {i+1}/{len(self.weights)}: {w}")

            # 1. Create weighted ensemble
            models_ensemble, w_actual, total_models = create_weighted_ensemble(
                models=self.models,
                w=w,
                verbose=False
            )
            # save weigthed ensemble - needed for image generation
            torch.save(models_ensemble, "models_ensemble.pt")
            
            # 2. Simulate images
            simulator = CryoEmSimulator(sim_config, device=self.device)
            images, parameters = simulator.simulate(
                num_sim=num_sim, 
                return_parameters=True,
                batch_size=batch_size
            )
            
            # 3. Compute likelihood matrix
            log_probs_matrix = evaluate_likelihood_pairwise(
                self.estimator,
                images,
                self.models,
                batch_size_pairs=20000,
                device=self.device
            ).T  # transpose to [n_models x n_images]
            
            # 4. Optimize weights
            opt = WeightOptimizer(log_probs_matrix, device=self.device)
            w_opt, _ = opt.optimize()
            
            # 5. Compute RMSE between optimized weights and target
            rmse_val = self.rmse(w_opt, w)
            
            # 6. Store results
            self.rmse_values.append(rmse_val)
            self.pop_fraction_record.append(w[0])  # fraction of closed state

        # Convert to NumPy arrays 
        self.rmse_values = np.array(self.rmse_values)
        self.pop_fraction_record = np.array(self.pop_fraction_record)
        return self.rmse_values, self.pop_fraction_record
