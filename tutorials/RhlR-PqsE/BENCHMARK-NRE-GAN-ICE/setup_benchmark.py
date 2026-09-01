#!/usr/bin/env python3
import itertools
import os
import re
import shutil
from pathlib import Path


def setup_benchmark():
    root_dir = Path.cwd()
    scripts_dir = root_dir / "00-SCRIPTS"
    data_dir = root_dir / "00-DATA"
    template_script = scripts_dir / "go.sh"

    # Validation checks
    if not scripts_dir.exists() or not template_script.exists():
        raise FileNotFoundError(
            f"Template script not found at {template_script}"
        )
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")

    # Read the SLURM template
    with open(template_script, "r") as f:
        template_content = f.read()

    # Define the parameter space (2 x 2 x 2 x 2 = 16 combinations)
    betas = ["1.0e-4", "1.0e-5"]
    betas_cons = ["0.1", "0.5"]
    betas_ot = ["0.0"]

    combinations = list(
        itertools.product(betas, betas_cons, betas_ot)
    )

    replicates = ["REP-01", "REP-02", "REP-03", "REP-04", "REP-05", "REP-06"]
    summary_lines = []

    summary_lines.append("=" * 80)
    summary_lines.append("BENCHMARK TEST CONFIGURATIONS SUMMARY")
    summary_lines.append("=" * 80 + "\n")

    for idx, (beta, beta_cons, beta_ot) in enumerate(
        combinations, start=1
    ):
        test_name = f"TEST-{idx:02d}"
        test_dir = root_dir / test_name
        test_dir.mkdir(exist_ok=True)

        # 1. Add details to summary file
        summary_lines.append(f"Directory: {test_name}")
        summary_lines.append(f"  - Beta (--beta)                        : {beta}")
        summary_lines.append(f"  - Beta cons (--beta_cons)              : {beta_cons}")
        summary_lines.append(f"  - Beta ot (--beta_OT)                  : {beta_ot}")
        summary_lines.append("-" * 50)

        # 2. Create Replicate Folders & Copy Data
        for rep in replicates:
            rep_dir = test_dir / rep
            rep_dir.mkdir(exist_ok=True)

            # Copy all data from 00-DATA to REP-XX
            for item in data_dir.iterdir():
                dest = rep_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

            # 3. Generate modified run_all.slurm for this TEST
            script_content = template_content
            
            # Build the customized pretrain command
            pretrain_flags = [
                "python ${SCRIPTS_}/pretrain_image_embed_v15.py",
                "--image_config simulation_parameters.json",
                "--embedding SPATIAL_CRYO",
                "--batch_size 512",
                "--embedding_dim 16",
                "--simulation_batch_size 1024",
                "--epochs 100",
                "--lr 2e-4",
                f"--beta {beta}",
                f"--beta_cons {beta_cons}",
                "--beta_NRE 0.1",
                f"--beta_OT {beta_ot}",
                "--real_data_mrc ${DATA_}/mixed-0.5.mrc",
                "--val_size 3000",
                "> log.pretrain",
            ]
            
            new_pretrain_cmd = " ".join(pretrain_flags)
            
            # Replace the original pretrain command line in template
            script_content = re.sub(
                r"python \$\{SCRIPTS_\}/pretrain_image_embed_v7\.py.*",
                new_pretrain_cmd,
                script_content,
            )
            
            # Write out go.sh inside each replica
            run_script_path = rep_dir / "go.sh"
            with open(run_script_path, "w") as f:
                f.write(script_content)


    # 5. Write plain text summary file in root
    summary_file = root_dir / "benchmark_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))

    print("Setup completed successfully!")
    print(f"- Directory structure created for 16 tests (6 replicates each).")
    print(f"- Data from '00-DATA' copied to each replicate folder.")
    print(f"- Execution script ('go.sh') created in each TEST-XX/REP-YY folder.")
    print(f"- Summary written to: {summary_file.name}")


if __name__ == "__main__":
    setup_benchmark()
