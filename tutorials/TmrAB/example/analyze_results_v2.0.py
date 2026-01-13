import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.offsetbox import AnchoredText

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

# --- 1. SET THE GROUND TRUTH VALUES ---
# This should be a 1D tensor with 6 real values.
GROUND_TRUTH = torch.tensor([0.114, 0.108, 0.0, 0.0, 0.418, 0.246])
# Renormalize it 
GROUND_TRUTH = GROUND_TRUTH / GROUND_TRUTH.sum()

# --- NEW: CALCULATE THE AGGREGATED "BASIN" GROUND TRUTH ---
# Sums pairs of elements: (0,1), (2,3), (4,5)
GROUND_TRUTH_BASIN = GROUND_TRUTH.view(3, 2).sum(axis=1)


# --- 2. SET THE BASE DIRECTORY ---
# This is the directory that contains all your 60 test folders.
# Use '.' if the script is in the same directory as the test folders.
BASE_DIR = Path('.')

# --- 3. PLOT SETTINGS ---
# Define the layout of the mosaic plot grid.
# nrows * ncols should be >= the number of test directories.
PLOT_GRID_ROWS = 4 
PLOT_GRID_COLS = 6
OUTPUT_FILENAME = "results_mosaic.png"

# =============================================================================

def parse_dir_name(dir_path: Path) -> dict:
    """Extracts parameters from a directory name using a corrected regex."""
    params = {}
    name = dir_path.name
    
    patterns = {
        'SHIFT': r"SHIFT_([\d.]+)",
        'SNRMIN': r"SNRMIN_([\d.]+)",
        'SNRMAX': r"SNRMAX_([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            try:
                params[key] = float(match.group(1))
            except (ValueError, IndexError):
                params[key] = 'N/A'
        else:
            params[key] = 'N/A'
    
    test_num_match = re.match(r"(\d+)-", name)
    if test_num_match:
        params['test_num'] = int(test_num_match.group(1))
    else:
        params['test_num'] = -1
        
    return params

def parse_log_file(log_path: Path) -> dict:
    """Extracts key metrics from the log.check file with specific regex for each metric."""
    metrics = {
        "PCA 2D Separation": "N/A",
        "Spatial Overlap": "N/A",
        "Visual Overlap": "N/A",
    }
    
    if not log_path.exists():
        return metrics

    try:
        content = log_path.read_text()
        
        # Pattern for PCA Separation (e.g., "1.50σ")
        pca_match = re.search(r"PCA 2D Separation\s*:\s*([\d.]+)\s*σ", content, re.IGNORECASE)
        if pca_match:
            try:
                metrics["PCA 2D Separation"] = f"{float(pca_match.group(1)):.2f}σ"
            except (ValueError, IndexError):
                pass
        
        # Pattern for Spatial Overlap (e.g., "18.0%")
        spatial_match = re.search(r"Spatial Overlap\s*:\s*([\d.]+)\s*%", content, re.IGNORECASE)
        if spatial_match:
            try:
                metrics["Spatial Overlap"] = f"{float(spatial_match.group(1)):.1f}%"
            except (ValueError, IndexError):
                pass

        # Pattern for Visual Overlap (e.g., "84.8%")
        visual_match = re.search(r"Visual Overlap\s*:\s*([\d.]+)\s*%", content, re.IGNORECASE)
        if visual_match:
            try:
                metrics["Visual Overlap"] = f"{float(visual_match.group(1)):.1f}%"
            except (ValueError, IndexError):
                pass

    except Exception as e:
        print(f"Warning: Could not read or parse {log_path}. Reason: {e}")

    return metrics

def calculate_rmse(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    """Calculates the Root Mean Square Error."""
    return torch.sqrt(torch.mean((predicted - truth) ** 2)).item()

def collect_data(base_dir: Path, ground_truth: torch.Tensor, ground_truth_basin: torch.Tensor):
    """Gathers all data from subdirectories, sorted by Test ID."""
    all_results = []
    
    subdirs = sorted(
        [d for d in base_dir.glob('[0-9]*-*') if d.is_dir()], 
        key=lambda p: int(p.name.split('-')[0])
    )

    print(f"Found {len(subdirs)} experiment directories.")

    for dir_path in subdirs:
        results_pt_path = dir_path / "results.pt"
        log_check_path = dir_path / "log.check"
        
        if not results_pt_path.exists():
            print(f"WARNING: Skipping {dir_path.name}, 'results.pt' not found.")
            continue
            
        try:
            loaded_data = torch.load(results_pt_path, weights_only=False)
            # skip garbage model if present
            predicted_tensor = torch.as_tensor(loaded_data, dtype=ground_truth.dtype)[0:6]
            # renormalize
            predicted_tensor = predicted_tensor / torch.sum(predicted_tensor)

            if predicted_tensor.shape != ground_truth.shape:
                print(f"WARNING: Skipping {dir_path.name}, tensor shape mismatch.")
                continue

            params = parse_dir_name(dir_path)
            metrics = parse_log_file(log_check_path)
            
            # --- MODIFIED: CALCULATE BOTH RMSE VALUES ---
            rmse = calculate_rmse(predicted_tensor, ground_truth)
            
            # Aggregate the predicted tensor into basins
            predicted_tensor_basin = predicted_tensor.view(3, 2).sum(axis=1)
            rmse_basin = calculate_rmse(predicted_tensor_basin, ground_truth_basin)
            # --- END MODIFICATION ---

            all_results.append({
                'params': params,
                'predicted': predicted_tensor,
                'metrics': metrics,
                'rmse': rmse,
                'rmse_basin': rmse_basin  # <-- ADDED
            })
        except Exception as e:
            print(f"ERROR: Could not process directory {dir_path.name}. Reason: {e}")
            
    return all_results

def print_results_table(results_data: list):
    """Prints a Markdown table of the results, sorted by Test ID."""
    if not results_data:
        print("No data available to generate a table.")
        return
    
    print("\n--- Experiment Results Summary (Sorted by Test ID) ---\n")
    # --- MODIFIED: ADDED RMSE_BASIN COLUMN ---
    header =    "| Test # | SHIFT |  SNR Min  |  SNR Max  |  RMSE  | RMSE Basin | PCA Sep. | Spatial Ovlp. | Visual Ovlp. |"
    separator = "|:------:|:-----:|:---------:|:---------:|:------:|:----------:|:--------:|:-------------:|:------------:|"
    print(header)
    print(separator)

    for data in results_data:
        p = data['params']
        m = data['metrics']
        # --- MODIFIED: ADDED RMSE_BASIN TO THE PRINTED ROW ---
        print(f"| {p['test_num']:<6} | {p['SHIFT']:<5.1f} | {p['SNRMIN']:<7.3f} | {p['SNRMAX']:<7.3f} | "
              f"{data['rmse']:.4f} | {data['rmse_basin']:.4f}   | "
              f"{m['PCA 2D Separation']:<8} | {m['Spatial Overlap']:<13} | {m['Visual Overlap']:<12} |")
    print("\n----------------------------------------------------------\n")

def create_mosaic_plot(results_data: list, ground_truth: torch.Tensor):
    """Creates and saves the final mosaic plot, ordered by Test ID."""
    if not results_data:
        print("No data to plot. Exiting.")
        return

    num_plots = len(results_data)
    fig, axes = plt.subplots(
        nrows=PLOT_GRID_ROWS, 
        ncols=PLOT_GRID_COLS, 
        figsize=(PLOT_GRID_COLS * 5, PLOT_GRID_ROWS * 4),
        constrained_layout=True
    )
    axes = axes.flatten()

    bar_width = 0.35
    x_indices = np.arange(len(ground_truth))

    for i, data in enumerate(results_data):
        ax = axes[i]
        
        ax.bar(x_indices - bar_width/2, data['predicted'].numpy(), bar_width, label='Predicted', color='skyblue')
        ax.bar(x_indices + bar_width/2, ground_truth.numpy(), bar_width, label='Ground Truth', color='salmon', alpha=0.8)
        
        p = data['params']
        title = (f"#{p['test_num']} SHIFT={p['SHIFT']}, SNR=[{p['SNRMIN']},{p['SNRMAX']}]")
        ax.set_title(title, fontsize=10, wrap=True)
        
        m = data['metrics']
        info_text = (f"RMSE: {data['rmse']:.4f}\n"
                     f"PCA Sep: {m['PCA 2D Separation']}\n"
                     f"Spatial Ovlp: {m['Spatial Overlap']}\n"
                     f"Visual Ovlp: {m['Visual Overlap']}")
        
        anc_text = AnchoredText(info_text, loc='upper right', frameon=True, prop=dict(size=8))
        anc_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        anc_text.patch.set_facecolor('wheat')
        anc_text.patch.set_alpha(0.5)
        ax.add_artist(anc_text)

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'Val {j+1}' for j in range(len(ground_truth))])
        ax.set_ylabel("Value")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(fontsize=8, loc='upper center')
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    fig.suptitle("Experiment Results Analysis Mosaic", fontsize=24, weight='bold')
    
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully created mosaic plot: {OUTPUT_FILENAME}")


if __name__ == '__main__':
    # --- MODIFIED: PASS THE NEW BASIN TENSOR TO THE COLLECTION FUNCTION ---
    results_data = collect_data(BASE_DIR, GROUND_TRUTH, GROUND_TRUTH_BASIN)
    print_results_table(results_data)
    create_mosaic_plot(results_data, GROUND_TRUTH)
