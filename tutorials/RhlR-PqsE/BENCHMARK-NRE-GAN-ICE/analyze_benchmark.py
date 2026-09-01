#!/usr/bin/env python3
"""
analyze_benchmark.py

Reads benchmark results from TEST-XX/REP-YY folders and produces a 6-panel
figure summarizing the late-epoch behavior across all 16 tests and 6 replicates.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
ROOT_DIR = Path.cwd()
TEST_IDS = [f"TEST-{i:02d}" for i in range(1, 7)]
REPS = [f"REP-{i:02d}" for i in range(1, 7)]

# Number of late checkpoints to keep and their spacing.
# For 100 epochs this means 5 checkpoints spaced by 5 epochs.
NUM_LATE_EPOCHS = 5
DEFAULT_STEP = 5

TEST_COLORS = cm.tab20(np.linspace(0, 1, len(TEST_IDS)))


# ------------------------------------------------------------------------------
# PARSING HELPERS (identical to analyze_benchmark.py)
# ------------------------------------------------------------------------------
def parse_pretrain_log(log_path: Path):
    """
    Parse log.pretrain and return a flat dict exactly like the first script:
        {
          'epoch_080': {'Pred loss': ..., ...},
          'epoch_085': {...},
          ...
          'epoch_final': {...}
        }
    """
    if not log_path.exists():
        return {}

    with open(log_path, "r") as f:
        text = f.read()

    results = {}

    # --- Per-epoch sections ---
    epoch_pattern = re.compile(
        r"Stage 1 Epoch\s+(\d+):\s*\n"
        r"(.*?)(?=Stage 1 Epoch\s+\d+:|={10,}|Computing final embedding|\Z)",
        re.DOTALL,
    )
    for match in epoch_pattern.finditer(text):
        epoch_num = int(match.group(1))
        section = match.group(2)
        key = f"{epoch_num:03d}"

        # Store every epoch; the dynamic planner selects which ones to use.
        results[f"epoch_{key}"] = {
            "Pred loss": _extract_float(section, r"Pred loss:\s+([0-9.eE+-]+)"),
            "Cons loss": _extract_float(section, r"Cons loss:\s+([0-9.eE+-]+)"),
            "NRE loss": _extract_float(section, r"NRE loss:\s+([0-9.eE+-]+)"),
            "NSR": _extract_float(
                section, r"Noise-to-Signal Ratio:\s+([0-9.eE+-]+)%"
            ),
            "mu_cov": _extract_mu_coverage(section),
        }

    # --- Final metrics section ---
    final_section = text.split("Final metrics:")[-1].split("SAVING WEIGHTS")[0]
    final_val_section = text.split("Final Validation (Sim2Real Overlap):")[-1]

    results["epoch_final"] = {
        "Pred loss": _extract_float(final_section, r"Pred loss:\s+([0-9.eE+-]+)"),
        "Cons loss": _extract_float(final_section, r"Cons loss:\s+([0-9.eE+-]+)"),
        "NRE loss": _extract_float(final_section, r"NRE loss:\s+([0-9.eE+-]+)"),
        "NSR": _extract_float(
            final_val_section, r"Noise-to-Signal Ratio:\s+([0-9.eE+-]+)%"
        ),
        "mu_cov": _extract_mu_coverage(final_val_section),
    }

    return results


def _extract_float(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else np.nan


def _extract_mu_coverage(section):
    """
    Extract 'Coverage (% in manifold):' immediately after the
    'Using mu_synth (deterministic):' block.
    """
    parts = re.split(r"Using mu_synth \(deterministic\):", section)
    if len(parts) < 2:
        return np.nan
    m = re.search(
        r"Coverage \(% in manifold\):\s+([0-9.eE+-]+)", parts[1]
    )
    return float(m.group(1)) if m else np.nan


# ------------------------------------------------------------------------------
# DYNAMIC EPOCH PLAN
# ------------------------------------------------------------------------------
def _numeric_pretrain_epochs(pretrain: dict):
    """Return sorted numeric epochs present in the parsed pretrain log."""
    eps = []
    for k in pretrain.keys():
        m = re.fullmatch(r"epoch_(\d{3})", k)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def _numeric_inference_epochs(rep_dir: Path):
    """Return sorted numeric epochs present in inference log filenames."""
    eps = []
    for p in sorted(rep_dir.glob("log.inference-small.*")):
        if p.name.endswith(".pt"):
            continue
        m = re.fullmatch(r"log\.inference-small\.(\d+)", p.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(set(eps))


def get_epoch_plan(rep_dir: Path, pretrain: dict):
    """
    Discover the late-epoch plan for this replicate.

    Logic:
      1. Numbered inference logs reveal the checkpoint spacing.
      2. The unlabeled log.inference-small is the final checkpoint,
         i.e. one step after the last numbered log.
      3. Fallback to pretrain Stage 1 Epoch headers if inference logs are absent.
    """
    inf_eps = _numeric_inference_epochs(rep_dir)
    pretrain_eps = _numeric_pretrain_epochs(pretrain)
    has_final_log = (
        (rep_dir / "log.inference-small").exists()
        or (rep_dir / "log.inference-small.final").exists()
    )
    has_final_block = "epoch_final" in pretrain

    # Determine step
    if len(inf_eps) >= 2:
        step = inf_eps[-1] - inf_eps[-2]
    elif len(pretrain_eps) >= 2:
        step = pretrain_eps[-1] - pretrain_eps[-2]
    else:
        step = DEFAULT_STEP

    # Determine final epoch
    if inf_eps and has_final_log:
        final_epoch = inf_eps[-1] + step
    elif inf_eps:
        final_epoch = inf_eps[-1]
    elif pretrain_eps and has_final_block:
        final_epoch = pretrain_eps[-1] + step
    elif pretrain_eps:
        final_epoch = pretrain_eps[-1]
    else:
        return [], None

    numeric_targets = [
        final_epoch - i * step for i in range(NUM_LATE_EPOCHS - 1, 0, -1)
    ]
    targets = numeric_targets + ["final"]

    return targets, final_epoch


def read_weight(rep_dir: Path, epoch, final_epoch):
    """
    Read the first element of 'Selected optimal weights:' exactly like the
    first script:
      - numeric epoch E -> log.inference-small.{E:03d}
      - final           -> log.inference-small  (or .final)
    """
    if epoch == "final":
        for name in ("log.inference-small", "log.inference-small.final"):
            path = rep_dir / name
            if path.exists():
                return _extract_first_weight(path)
        return np.nan

    path = rep_dir / f"log.inference-small.{epoch:03d}"
    if path.exists():
        return _extract_first_weight(path)
    return np.nan


def _extract_first_weight(log_path: Path):
    """Return the first element of 'Selected optimal weights:' from a log."""
    if not log_path.exists():
        return np.nan
    with open(log_path, "r") as f:
        text = f.read()
    m = re.search(
        r"Selected optimal weights:\s*\n\s*\[\s*([0-9.eE+-]+)", text
    )
    return float(m.group(1)) if m else np.nan


# ------------------------------------------------------------------------------
# DATA COLLECTION
# ------------------------------------------------------------------------------
def collect_data():
    rows = []

    for test_idx, test_id in enumerate(TEST_IDS, start=1):
        test_dir = ROOT_DIR / test_id
        if not test_dir.exists():
            print(f"Warning: {test_dir} not found, skipping.")
            continue

        for rep in REPS:
            rep_dir = test_dir / rep
            if not rep_dir.exists():
                print(f"Warning: {rep_dir} not found, skipping.")
                continue

            pretrain = parse_pretrain_log(rep_dir / "log.pretrain")
            if not pretrain:
                print(f"Warning: could not parse pretrain log in {rep_dir}, skipping.")
                continue

            targets, final_epoch = get_epoch_plan(rep_dir, pretrain)
            print(f"{rep_dir}: final_epoch={final_epoch}, targets={targets}")

            for epoch in targets:
                if epoch == "final":
                    key = "epoch_final"
                    if key not in pretrain:
                        continue
                    metrics = pretrain[key]
                    epoch_num = final_epoch
                else:
                    key = f"epoch_{epoch:03d}"
                    if key not in pretrain:
                        continue
                    metrics = pretrain[key]
                    epoch_num = epoch

                weight = read_weight(rep_dir, epoch, final_epoch)

                rows.append(
                    {
                        "test_id": test_id,
                        "test_idx": test_idx,
                        "rep": rep,
                        "epoch_num": epoch_num,
                        "epoch_key": key,
                        "Pred loss": metrics.get("Pred loss", np.nan),
                        "Cons loss": metrics.get("Cons loss", np.nan),
                        "NRE loss": metrics.get("NRE loss", np.nan),
                        "NSR": metrics.get("NSR", np.nan),
                        "mu_cov": metrics.get("mu_cov", np.nan),
                        "first_weight": weight,
                    }
                )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# PLOTTING (identical to analyze_benchmark.py)
# ------------------------------------------------------------------------------
def jitter_x(x_center, n_points, width=0.25, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return x_center + rng.uniform(-width, width, size=n_points)


def plot_benchmark(df, output_path="benchmark_summary.png"):
    panels = [
        ("Pred loss", "Loss", "Pred loss"),
        ("Cons loss", "Loss", "Cons loss"),
        ("NRE loss", "Loss", "NRE loss"),
        ("Noise-to-Signal Ratio", "NSR (%)", "NSR"),
        ("mu_synth coverage %", "Coverage (%)", "mu_cov"),
        ("First optimal weight", "Weight", "first_weight"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    jitter_width = 0.22
    np.random.seed(42)

    for ax_idx, (title, ylabel, col) in enumerate(panels):
        ax = axes[ax_idx]

        for test_idx, test_id in enumerate(TEST_IDS, start=1):
            subset = df[(df["test_id"] == test_id) & (df[col].notna())]
            if subset.empty:
                continue

            y_vals = subset[col].values
            color = TEST_COLORS[test_idx - 1]

            x_jitter = jitter_x(test_idx, len(y_vals), width=jitter_width)

            ax.scatter(
                x_jitter,
                y_vals,
                color=color,
                alpha=0.6,
                s=30,
                edgecolors="none",
                zorder=2,
                label=test_id if ax_idx == 0 else "",
            )

            mean_val = np.mean(y_vals)
            median_val = np.median(y_vals)
            std_val = np.std(y_vals, ddof=1)

            ax.scatter(
                test_idx,
                mean_val,
                marker="o",
                color="white",
                s=80,
                zorder=5,
                edgecolors="black",
                linewidths=1.0,
            )

            ax.scatter(
                test_idx,
                median_val,
                marker="+",
                color="black",
                s=300,
                linewidths=1.0,
                zorder=4,
            )

            ax.errorbar(
                test_idx,
                mean_val,
                yerr=std_val,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=5,
                capthick=1,
                zorder=4,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("TEST ID", fontsize=10)
        ax.set_xticks(range(1, len(TEST_IDS) + 1))
        ax.set_xticklabels(
            [str(i) for i in range(1, len(TEST_IDS) + 1)], fontsize=7
        )
        ax.grid(True, alpha=0.3, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=8,
        fontsize=8,
        title="TEST",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Collecting benchmark data...")
    df = collect_data()

    csv_path = ROOT_DIR / "benchmark_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved data table to {csv_path}")

    print("Generating figure...")
    plot_benchmark(df, output_path="benchmark_summary.png")

    summary = (
        df.groupby("test_idx")
        .agg(
            n=("Pred loss", "size"),
            pred_mean=("Pred loss", "mean"),
            cons_mean=("Cons loss", "mean"),
            nre_mean=("NRE loss", "mean"),
            nsr_mean=("NSR", "mean"),
            mucov_mean=("mu_cov", "mean"),
            weight_mean=("first_weight", "mean"),
        )
        .reset_index()
    )
    print("\nPer-TEST summary (means across late-epoch points):")
    print(summary.to_string(index=False))
