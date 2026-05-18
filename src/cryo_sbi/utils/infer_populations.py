import torch 
import numpy as np 
from tqdm.auto import tqdm
from cryo_sbi import CryoEmSimulator
import cryo_sbi.utils.estimator_utils as est_utils
from cryo_sbi.inference.models import build_models
import mrcfile
from typing import Optional, Tuple, List
try:
    import jax
    import jax.numpy as jnp
    from jax.nn import softmax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: jax/numpyro not installed. Bayesian inference populations disabled.")
    print("Install with: conda install conda-forge::numpyro")


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

def create_weighted_ensemble(models, w, precision=0.1, max_models=10000, verbose=True):
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

def generate_population_weights(n_states, population_steps):
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

    weights = np.array(weights_list)
    pop_fractions = np.array(pop_fractions)
    state_pairs = np.array(state_pairs)

    weights, idx = np.unique(weights, axis=0, return_index=True)
    pop_fractions = pop_fractions[idx]
    state_pairs = state_pairs[idx]

    return weights, pop_fractions, state_pairs

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
    
    log_probs = torch.zeros(N_images, N_models, device='cpu')
    
    with torch.no_grad():
        print(f"Evaluating {total_pairs:,} pairs in batches of {batch_size_pairs}...")
        
        # Process all pairs in large batches
        for pair_start in tqdm(range(0, total_pairs, batch_size_pairs), desc="Pairs"):
            pair_end = min(pair_start + batch_size_pairs, total_pairs)
            batch_size = pair_end - pair_start
            
            # Compute which (image, model) each pair corresponds to
            pair_indices = torch.arange(pair_start, pair_end)
            img_idx = pair_indices // N_models  # [batch_size]
            mod_idx = pair_indices % N_models   # [batch_size]
            
            # Gather the corresponding images and model indices
            #batch_images = images[img_idx].to(device) # [batch_size, H, W]
            batch_images =  images[img_idx.cpu()]
            batch_indices = model_indices[mod_idx].unsqueeze(-1) # [batch_size, 1]
            
            # Single forward pass!
            log_p = estimator(batch_images.to(device), batch_indices)  # [batch_size]
            del batch_images  # free GPU memory
            torch.cuda.empty_cache()
            
            # Place results in the correct positions
            log_probs.view(-1)[pair_start:pair_end] = log_p
    
    return log_probs

def sample_posterior_weights(
    log_probs_matrix: torch.Tensor,
    num_samples: int = 2000,
    num_warmup: int = 1000,
    num_chains: int = 2,
    device: str = "cpu"
) -> np.ndarray:
    """
    Samples the posterior distribution of weights w using NUTS (HMC).

    Args:
        log_probs_matrix: Tensor of shape [N_models, N_images]
        num_samples: Number of samples to draw from the posterior.
        num_warmup: Number of "burn-in" steps for the sampler to adapt.
        num_chains: Number of parallel chains to run.
        device: 'cpu' or 'cuda'.

    Returns:
        w_samples: An array of posterior samples for the weights,
                   shape [num_chains * num_samples, N_models].
    """
    # 1. Set NumPyro to use the correct platform (CPU or GPU)
    numpyro.set_platform(device)
    
    # 2. Convert the log-likelihood matrix to a JAX array
    # This is the only data that needs to be passed to the model
    log_p_jax = jnp.array(log_probs_matrix.cpu().numpy(), dtype=jnp.float32)
    n_models, n_images = log_p_jax.shape

    # 3. Define the probabilistic model in NumPyro
    def model(log_p_matrix):
        # Unconstrained parameters for the weights
        # We place a standard Normal prior on z. This induces a logistic-normal
        # prior on the weights w, which is a reasonable default.
        z = numpyro.sample('z', dist.Normal(0., 1.).expand([n_models]))
        
        # Transform to the simplex to get weights
        w = softmax(z)
        
        # For inspecting the weights during sampling
        numpyro.deterministic('w', w)

        # Log-likelihood calculation (the numerically stable way)
        # log L = sum_i log(sum_j w_j * p_ij)
        #       = sum_i logsumexp_j (log(w_j) + log(p_ij))
        log_w = jnp.log(w + 1e-15)
        log_terms = log_w[:, None] + log_p_matrix  # shape [n_models, n_images]
        
        log_likelihood_per_image = jax.scipy.special.logsumexp(log_terms, axis=0)
        
        total_log_likelihood = jnp.sum(log_likelihood_per_image)
        
        # The numpyro.factor statement adds a potential term to the
        # overall log probability of the model. This is how we provide our
        # custom log-likelihood.
        numpyro.factor("log_likelihood", total_log_likelihood)

    # 4. Run the NUTS sampler
    print("Initializing NUTS sampler...")
    rng_key = jax.random.PRNGKey(0)
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    
    print(f"Running MCMC with {num_chains} chain(s)...")
    mcmc.run(rng_key, log_p_matrix=log_p_jax)
    
    print("\nMCMC summary:")
    mcmc.print_summary()
    
    # 5. Extract the samples for the weights 'w'
    posterior_samples = mcmc.get_samples()
    w_samples = posterior_samples['w']
    
    return w_samples


class WeightOptimizer:
    """
    Numerically stable optimizer for:
    L = -sum_i log(sum_j w_j * p_ij) + theta * sum_j w_j * log(w_j / w_j^0)
    
    This version uses the Adam optimizer, suitable for mini-batching on large datasets.
    """
    
    def __init__(self, log_p: np.ndarray, w0: Optional[np.ndarray] = None, theta: float = 0.0, device: str = 'cpu'):
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

    def compute_loss(self, w: torch.Tensor) -> torch.Tensor:
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
        
    def optimize(self, lr: float = 0.1, max_iter: int = 10000, tol: float = 1e-9, verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize weights using PyTorch Adam optimizer.
        
        Args:
            lr: learning rate
            max_iter: maximum iterations
            tol: convergence tolerance
            verbose: print progress
        """
        # Use unconstrained parameterization: w = softmax(z)
        z_init = torch.randn(self.n_j, dtype=torch.float64, device=self.device)
        z = z_init.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=lr)
            
        losses = []
        for iteration in range(max_iter):
             optimizer.zero_grad()
                
             w = torch.softmax(z, dim=0)
             loss = self.compute_loss(w)
             loss.backward()
             optimizer.step()
                
             losses.append(loss.item())
                
             if verbose and iteration % 100 == 0:
                  print(f"Iter {iteration}: Loss = {loss.item():.8f}")
                
             if iteration > 10 and abs(losses[-1] - losses[-2]) < tol:
                 if verbose:
                     print(f"Converged at iteration {iteration}")
                 break
                    
        with torch.no_grad():
            w_opt = torch.softmax(z, dim=0)
        
        return w_opt.cpu().numpy(), losses


class WeightOptimizerLBFGS(WeightOptimizer):
    """
    Numerically stable optimizer for the same objective function, but using the L-BFGS optimizer.

    L-BFGS is a quasi-Newton method that often converges in far fewer iterations
    for smooth, convex-like problems. However, it requires computing the gradient
    on the **full dataset** at each step and is NOT suitable for mini-batching.
    It should only be used if the entire `log_p` matrix can be processed in memory.
    """
    
    def optimize(self, max_iter: int = 100, tol: float = 1e-9, verbose: bool = False, history_size: int = 100) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize weights using PyTorch L-BFGS optimizer.
        
        Args:
            max_iter: maximum number of optimization steps (L-BFGS updates).
            tol: convergence tolerance on the loss function value.
            verbose: print progress.
            history_size: the number of past gradients L-BFGS uses to approximate the Hessian matrix.
        """
        # Use unconstrained parameterization: w = softmax(z)
        z_init = torch.randn(self.n_j, dtype=torch.float64, device=self.device)
        z = z_init.clone().detach().requires_grad_(True)
        
        # L-BFGS does not use a learning rate. It uses a line search.
        optimizer = torch.optim.LBFGS(
            [z], 
            history_size=history_size, 
            max_iter=20, # Max iterations for the line search within one step
            line_search_fn="strong_wolfe"
        )
            
        losses = []
        for iteration in range(max_iter):
             
             # The L-BFGS optimizer requires a "closure" function that
             # re-evaluates the model and returns the loss.
             def closure():
                 optimizer.zero_grad()
                 w = torch.softmax(z, dim=0)
                 loss = self.compute_loss(w) # Note: computes loss on the FULL dataset
                 loss.backward()
                 return loss

             # optimizer.step performs the update.
             # It calls the closure multiple times to compute the loss and gradient.
             loss = optimizer.step(closure)
                
             losses.append(loss.item())
                
             if verbose:
                  print(f"Iter {iteration}: Loss = {loss.item():.8f}")
                
             # Check for convergence
             if iteration > 0 and abs(losses[-1] - losses[-2]) < tol:
                 if verbose:
                     print(f"Converged at iteration {iteration}")
                 break
                    
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
    def __init__(self, models, estimator, device, population_steps, num_sim, use_random = False):
        """
        models: PyTorch tensor of models (n_models, 3, n_atoms)
        estimator: trained Cryo-EM estimator
        device: torch device
        population_steps: how many fractions
        num_sim: number of images to simulate for each population
        """
        self.estimator = estimator
        self.models = models
        self.device = device
        self.population_steps = population_steps
        self.n_models = len(models)
        self.num_sim = num_sim
        
        # Generate population weights
        if use_random == True:
            self.weights = np.random.dirichlet(np.ones(self.n_models), size=1000)
        else:
            self.weights, self.pop_fractions, self.state_pairs = generate_population_weights(
                n_states=self.n_models,
                population_steps=self.population_steps
            )

        # Storage
        self.rmse_values = []
        self.actual_weights = []
        self.opt_weights = []
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
            images, _ = simulator.simulate(
                num_sim=num_sim, 
                return_parameters=True,
                batch_size=batch_size
            )
            
            # 3. Compute likelihood matrix
            log_probs_matrix = evaluate_likelihood_pairwise(
                self.estimator,
                images,
                self.models,
                batch_size_pairs=512,
                device=self.device
            ).T  # transpose to [n_models x n_images]
            
            # 4. Optimize weights
            opt = WeightOptimizer(log_probs_matrix, device=self.device)
            w_opt, _ = opt.optimize()
            
            # 5. Compute RMSE between optimized weights and target
            rmse_val = self.rmse(w_opt, w)
            
            # 6. Store results
            self.rmse_values.append(rmse_val) # rmse value
            self.actual_weights.append(w)  # actual weights
            self.opt_weights.append(w_opt)  # optimized weights
            self.pop_fraction_record.append(w[0])

        # Convert to NumPy arrays 
        self.rmse_values = np.array(self.rmse_values)
        self.actual_weights = np.array(self.actual_weights)
        self.opt_weights = np.array(self.opt_weights)
        self.pop_fraction_record = np.array(self.pop_fraction_record)
        return self.rmse_values, self.actual_weights, self.opt_weights, self.pop_fraction_record


def run_inference_real_data(args):
    """
    Executes the core inference workflow using the provided arguments.

    Args:
        args: An argparse.Namespace object containing the script parameters.
    """
    # 1. Setup Device
    if not torch.cuda.is_available() and "cuda" in args.device:
        print(f"CUDA not available. Switching device from '{args.device}' to 'cpu'.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Load Pre-trained SBI Estimator
    print(f"Loading SBI estimator from {args.estimator_file}")
    estimator = est_utils.load_estimator(
        args.train_config_file,
        args.image_config_file,
        build_models.build_nle_flow_model,
        args.estimator_file,
        device=device
    )
    estimator.eval()

    # 3. Load 3D Models
    print(f"Loading 3D models from {args.models_file}")
    models = torch.load(args.models_file).to(device)
    print(f"Loaded {models.shape[0]} models of size {models.shape[1:]}")

    # 4. Load Experimental 2D Images
    print(f"Reading experimental images from {args.image_stack}")
    with mrcfile.open(args.image_stack, mode='r') as mrc:
        images = mrc.data
    
    print(f"Image stack shape: {images.shape}")
    print(f"Number of particles: {images.shape[0]}")
    print(f"Particle size: {images.shape[1]} x {images.shape[2]}")
 
    # Convert to torch tensor, kept on CPU to be batched to GPU later
    images = torch.from_numpy(images).cpu()

    # 5. Evaluate Likelihood Matrix
    print("Evaluating pairwise likelihood matrix...")
    log_probs_matrix = evaluate_likelihood_pairwise(
        estimator,
        images,
        models,
        batch_size_pairs=args.batch_size,
        device=device
    )
    # Transpose to shape [N_models, N_images] for the optimizer
    log_probs_matrix = log_probs_matrix.T
    print(f"Likelihood matrix evaluation complete. Shape: {log_probs_matrix.shape}")

    # 5b. Optionally save the likelihood matrix
    log_likelihood_file = getattr(args, "log_likelihood_file", None)
    if log_likelihood_file is not None:
        print(f"Saving likelihood matrix to {log_likelihood_file}")
        torch.save(log_probs_matrix, log_likelihood_file)

    # 6. Optimize Weights (use two different optimizers)
    print("Initializing weight optimizer...")

    adam_optimizer  = WeightOptimizer(log_probs_matrix, device=device)
    lbfgs_optimizer = WeightOptimizerLBFGS(log_probs_matrix, device=device)

    print("Optimizing weights to maximize the posterior...")

    print("Optimizing with Adam")
    w_adam, losses_adam = adam_optimizer.optimize(lr=0.1, max_iter=1000, tol=1e-10, verbose=True)
    print(f"Adam converged in {len(losses_adam)} iterations.\n")
    np.set_printoptions(precision=4, suppress=True)
    print(f"\nFinal Adam Weights: {w_adam}")
    print(f"\nFinal Adam Loss:  {losses_adam[-1]:.8f}")

    print("\nOptimizing with L-BFGS")
    w_lbfgs, losses_lbfgs = lbfgs_optimizer.optimize(max_iter=1000, tol=1e-10, verbose=True)
    print(f"L-BFGS converged in {len(losses_lbfgs)} iterations\n")
    print(f"\nFinal L-BFGS Weights: {w_lbfgs}")
    print(f"Final L-BFGS Loss: {losses_lbfgs[-1]:.8f}")
    
    # 7. Save and Report Results
    if(losses_adam[-1]<losses_lbfgs[-1]):
       w_opt = w_adam
    else:
       w_opt = w_lbfgs

    print("\nOptimization complete.")
    print(f"Optimal weights:\n{w_opt}")
    
    print(f"Saving optimal weights to {args.output_file}")
    torch.save(w_opt, args.output_file)


def run_inference_real_data_bayes(args):
    """
    Executes the core inference workflow using the provided arguments.
    This version is modified to use MCMC sampling instead of optimization.
    """
    if not JAX_AVAILABLE:
       raise ImportError("Jax/Numpyro not installed. Install with: conda install conda-forge::numpyro")

    # 1. Setup Device
    if not torch.cuda.is_available() and "cuda" in args.device:
        print(f"CUDA not available. Switching device from '{args.device}' to 'cpu'.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Load Pre-trained SBI Estimator
    print(f"Loading SBI estimator from {args.estimator_file}")
    estimator = est_utils.load_estimator(
        args.train_config_file,
        args.image_config_file,
        build_models.build_nle_flow_model,
        args.estimator_file,
        device=device
    )
    estimator.eval()

    # 3. Load 3D Models
    print(f"Loading 3D models from {args.models_file}")
    models = torch.load(args.models_file).to(device)
    print(f"Loaded {models.shape[0]} models of size {models.shape[1:]}")

    # 4. Load Experimental 2D Images
    print(f"Reading experimental images from {args.image_stack}")
    with mrcfile.open(args.image_stack, mode='r') as mrc:
        images = mrc.data

    print(f"Image stack shape: {images.shape}")
    print(f"Number of particles: {images.shape[0]}")
    print(f"Particle size: {images.shape[1]} x {images.shape[2]}")

    # Convert to torch tensor, kept on CPU to be batched to GPU later
    images = torch.from_numpy(images).cpu()

    # 5. Evaluate Likelihood Matrix
    print("Evaluating pairwise likelihood matrix...")
    log_probs_matrix = evaluate_likelihood_pairwise(
        estimator,
        images,
        models,
        batch_size_pairs=args.batch_size,
        device=device
    )
    # Transpose to shape [N_models, N_images] for the optimizer
    log_probs_matrix = log_probs_matrix.T
    print(f"Likelihood matrix evaluation complete. Shape: {log_probs_matrix.shape}")

    # 6. Sample from the posterior using MCMC
    print("\nStarting Bayesian inference via MCMC sampling...")
    w_samples = sample_posterior_weights(
        log_probs_matrix,
        num_samples=2000,
        num_warmup=1000,
        num_chains=2
    )
    # w_samples will have shape [4000, N_models]

    # 7. Analyze, Save, and Report Results
    print("\nInference complete. Analyzing posterior samples...")
    
    # Calculate posterior mean as a point estimate
    w_mean = np.mean(w_samples, axis=0)
    
    # Calculate 95% highest posterior density interval (credible interval)
    w_lower = np.percentile(w_samples, 2.5, axis=0)
    w_upper = np.percentile(w_samples, 97.5, axis=0)
    
    np.set_printoptions(precision=4, suppress=True)
    print("\n--- Posterior Summary ---")
    print(f"{'Model':<8} {'Mean Weight':<15} {'95% Credible Interval':<25}")
    print("-" * 50)
    for i in range(models.shape[0]):
        print(f"{i:<8} {w_mean[i]:<15.4f} [{w_lower[i]:.4f}, {w_upper[i]:.4f}]")
    print("-" * 50)

    print(f"Saving posterior mean weights as a torch tensor to {args.output_file}")
    w_mean_tensor = torch.from_numpy(np.array(w_mean)).float()
    torch.save(w_mean_tensor, args.output_file)