# "plot_metrics.py"
import os
import matplotlib.pyplot as plt
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = sys.argv[1]
NUM_TESTS = 12

# A distinct color palette for up to 12 tests
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
    '#bcbd22', '#17becf', '#393b79', '#9c9ede'
]

def extract_losses(log_path):
    """Extracts 'Pred loss:' values sequentially from log.pretrain"""
    losses = []
    if not os.path.exists(log_path):
        return losses
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Pred loss:" in line:
                # Line looks like: "    Total loss:     1.234567"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        losses.append(float(parts[2]))
                    except ValueError:
                        pass
    return losses

def extract_validation_metrics(metrics_path):
    """
    Extracts sequentially NSR (%), Coverage, Dist (Med), and Dist (p90) from the log txt
    for both z_synthetic (stochastic) and mu_synthetic (deterministic).
    """
    nsrs = []
    cov_z, med_z, p90_z = [], [], []
    cov_mu, med_mu, p90_mu = [], [], []
    
    if not os.path.exists(metrics_path):
        return nsrs, cov_mu, med_mu, p90_mu, cov_z, med_z, p90_z
        
    with open(metrics_path, 'r') as f:
        lines = f.readlines()
        
        # Skip header lines
        for line in lines[2:]:
            if line.strip():  # ignore empty lines
                parts = line.split('|')
                if len(parts) >= 8:
                    try:
                        nsrs.append(float(parts[1].strip()))
                        
                        cov_z.append(float(parts[2].strip()))
                        med_z.append(float(parts[3].strip()))
                        p90_z.append(float(parts[4].strip()))

                        cov_mu.append(float(parts[5].strip()))
                        med_mu.append(float(parts[6].strip()))
                        p90_mu.append(float(parts[7].strip()))
                    except ValueError:
                        pass
                        
    return nsrs, cov_mu, med_mu, p90_mu, cov_z, med_z, p90_z

def main():
    # Set up the 2x3 subplots figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: mu_synthetic (deterministic)
    ax_cov_mu, ax_med_mu, ax_p90_mu = axes[0, 0], axes[0, 1], axes[0, 2]
    
    # Row 2: z_synthetic (stochastic)
    ax_cov_z, ax_med_z, ax_p90_z = axes[1, 0], axes[1, 1], axes[1, 2]
    
    for i in range(1, NUM_TESTS + 1):
        test_name = f"TEST-{i:02d}"
        test_dir = os.path.join(BASE_DIR, test_name)
        color = COLORS[i - 1]
        
        pretrain_log = os.path.join(test_dir, "log.pretrain")
        val_metrics_log = os.path.join(test_dir, "validation_metrics_log.txt")
        
        # Extract the ordered data
        losses = extract_losses(pretrain_log)
        nsrs, cov_mu, med_mu, p90_mu, cov_z, med_z, p90_z = extract_validation_metrics(val_metrics_log)
        
        # Zip up to shortest list length to prevent mismatch errors
        min_len = min(len(losses), len(nsrs), len(cov_mu), len(cov_z))
        
        if min_len == 0:
            print(f"Skipping {test_name}: Data missing.")
            continue
            
        plot_losses = losses[:min_len]
        plot_nsrs = nsrs[:min_len]
        
        plot_cov_mu = cov_mu[:min_len]
        plot_med_mu = med_mu[:min_len]
        plot_p90_mu = p90_mu[:min_len]
        
        plot_cov_z = cov_z[:min_len]
        plot_med_z = med_z[:min_len]
        plot_p90_z = p90_z[:min_len]
        
        # Create test label annotated with final Noise-to-Signal Ratio (NSR)
        final_nsr = plot_nsrs[-1]
        label_text = f"{test_name} (NSR: {final_nsr:.1f}%)"
        
        # Row 1: mu_synthetic (deterministic)
        ax_cov_mu.scatter(plot_losses, plot_cov_mu, color=color, label=label_text, alpha=0.8, s=40)
        ax_med_mu.scatter(plot_losses, plot_med_mu, color=color, label=label_text, alpha=0.8, s=40)
        ax_p90_mu.scatter(plot_losses, plot_p90_mu, color=color, label=label_text, alpha=0.8, s=40)

        # Row 2: z_synthetic (stochastic)
        ax_cov_z.scatter(plot_losses, plot_cov_z, color=color, label=label_text, alpha=0.8, s=40)
        ax_med_z.scatter(plot_losses, plot_med_z, color=color, label=label_text, alpha=0.8, s=40)
        ax_p90_z.scatter(plot_losses, plot_p90_z, color=color, label=label_text, alpha=0.8, s=40)
        
        print(f"Plotted {test_name} ({min_len} points, Final NSR: {final_nsr:.2f}%)")

    # ---------------------------------------------------------
    # Formatting Row 1: mu_synthetic (deterministic)
    # ---------------------------------------------------------
    ax_cov_mu.set_xlabel("Pred Loss")
    ax_cov_mu.set_ylabel("Coverage (%)")
    ax_cov_mu.set_title("Pred Loss vs. Coverage (mu_synth)")
    ax_cov_mu.grid(True, linestyle='--', alpha=0.6)
    
    ax_med_mu.set_xlabel("Pred Loss")
    ax_med_mu.set_ylabel("Distance (Median)")
    ax_med_mu.set_title("Pred Loss vs. Distance Median (mu_synth)")
    ax_med_mu.grid(True, linestyle='--', alpha=0.6)
    
    ax_p90_mu.set_xlabel("Pred Loss")
    ax_p90_mu.set_ylabel("Distance (p90)")
    ax_p90_mu.set_title("Pred Loss vs. Distance p90 (mu_synth)")
    ax_p90_mu.grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # Formatting Row 2: z_synthetic (stochastic)
    # ---------------------------------------------------------
    ax_cov_z.set_xlabel("Pred Loss")
    ax_cov_z.set_ylabel("Coverage (%)")
    ax_cov_z.set_title("Pred Loss vs. Coverage (z_synth)")
    ax_cov_z.grid(True, linestyle='--', alpha=0.6)
    
    ax_med_z.set_xlabel("Pred Loss")
    ax_med_z.set_ylabel("Distance (Median)")
    ax_med_z.set_title("Pred Loss vs. Distance Median (z_synth)")
    ax_med_z.grid(True, linestyle='--', alpha=0.6)
    
    ax_p90_z.set_xlabel("Pred Loss")
    ax_p90_z.set_ylabel("Distance (p90)")
    ax_p90_z.set_title("Pred Loss vs. Distance p90 (z_synth)")
    ax_p90_z.grid(True, linestyle='--', alpha=0.6)

    # Add a single shared legend at the bottom
    ax_med_z.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)

    # Adjust layout so legend and titles fit nicely
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    # Save and show
    output_filename = "validation_scatter_plots.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()
