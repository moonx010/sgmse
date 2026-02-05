#!/usr/bin/env python
"""Batch evaluation script for noise-conditioned models.

Usage:
    python scripts/eval_batch.py --phase enhance --gpus 0,1,2,3
    python scripts/eval_batch.py --phase metrics
    python scripts/eval_batch.py --phase all --gpus 0,1,2,3
"""

import os
import re
import subprocess
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ============================================
# CONFIGURATION - Modify this section
# ============================================

EXPERIMENTS = [
    # (name, checkpoint_path, cfg_scale, is_noise_cond)
    # ========== Scaled Training (Paper-level, 58k steps) ==========
    # SGMSE+ baseline (paper config, no noise conditioning)
    ("sgmse_scaled", "./logs/55jxu1gw/last.ckpt", 1.0, False),
    # CFG p=0.2 scaled
    ("cfg_p0.2_scaled", "./logs/50y8p2e4/last.ckpt", 1.0, True),

    # ========== PoC (50k steps, batch=4) ==========
    # SGMSE+ baseline (no noise conditioning)
    # ("sgmse_baseline", "./logs/ppwxy81n-sgmse-baseline/step=50000.ckpt", 1.0, False),
    # CFG p=0.2 (best PoC)
    # ("cfg_p0.2", "./logs/kvue4el4-nc-cfg-p0.2/step=50000.ckpt", 1.0, True),
]

DATASETS = [
    # (name_suffix, test_dir, clean_dir, noisy_dir)
    ("", "./data/voicebank-demand/test", "./data/voicebank-demand/test/clean", "./data/voicebank-demand/test/noisy"),
    ("_ood", "./data/test_mixtures/vb_esc50/snr_0dB", "./data/test_mixtures/vb_esc50/snr_0dB/clean", "./data/test_mixtures/vb_esc50/snr_0dB/noisy"),
]

N_STEPS = 30
OUTPUT_BASE = "./enhanced"
RESULTS_DIR = "./evaluation_results"

# ============================================
# END CONFIGURATION
# ============================================


def run_enhancement(gpu_id, exp_name, ckpt_path, cfg_scale, is_noise_cond, dataset_suffix, test_dir, clean_dir):
    """Run enhancement on a single GPU."""
    output_dir = f"{OUTPUT_BASE}_{exp_name}{dataset_suffix}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if is_noise_cond:
        # Noise-conditioned model
        cmd = [
            "python", "enhancement_noise_cond.py",
            "--test_dir", test_dir,
            "--enhanced_dir", output_dir,
            "--ckpt", ckpt_path,
            "--oracle_noise",
            "--clean_dir", clean_dir,
            "--N", str(N_STEPS),
            "--cfg_scale", str(cfg_scale),
            "--device", "cuda"
        ]
    else:
        # SGMSE+ baseline (no noise conditioning)
        cmd = [
            "python", "enhancement.py",
            "--test_dir", os.path.join(test_dir, "noisy"),
            "--enhanced_dir", output_dir,
            "--ckpt", ckpt_path,
            "--N", str(N_STEPS),
            "--device", "cuda"
        ]

    print(f"[GPU {gpu_id}] Starting: {exp_name}{dataset_suffix}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[GPU {gpu_id}] ERROR: {exp_name}{dataset_suffix}")
        print(result.stderr)
    else:
        print(f"[GPU {gpu_id}] Done: {exp_name}{dataset_suffix}")

    return result.returncode


def run_metrics(exp_name, cfg_scale, dataset_suffix, clean_dir, noisy_dir):
    """Run metrics calculation and return parsed results."""
    output_dir = f"{OUTPUT_BASE}_{exp_name}{dataset_suffix}"

    cmd = [
        "python", "calc_metrics.py",
        "--clean_dir", clean_dir,
        "--noisy_dir", noisy_dir,
        "--enhanced_dir", output_dir
    ]

    print(f"\n=== {exp_name}{dataset_suffix} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # Parse metrics from output
    metrics = {
        "experiment": exp_name,
        "dataset": "VB-DEMAND" if dataset_suffix == "" else "OOD (ESC-50)",
        "cfg_scale": cfg_scale,
    }

    # Parse PESQ, ESTOI, SI-SDR with std from stdout
    # Format: "PESQ: 1.59 ± 0.42"
    for line in result.stdout.split("\n"):
        if "PESQ:" in line:
            match = re.search(r"PESQ:\s*([\d.]+)\s*±\s*([\d.]+)", line)
            if match:
                metrics["PESQ"] = float(match.group(1))
                metrics["PESQ_std"] = float(match.group(2))
        elif "ESTOI:" in line:
            match = re.search(r"ESTOI:\s*([\d.]+)\s*±\s*([\d.]+)", line)
            if match:
                metrics["ESTOI"] = float(match.group(1))
                metrics["ESTOI_std"] = float(match.group(2))
        elif "SI-SDR:" in line:
            match = re.search(r"SI-SDR:\s*([\d.-]+)\s*±\s*([\d.]+)", line)
            if match:
                metrics["SI-SDR"] = float(match.group(1))
                metrics["SI-SDR_std"] = float(match.group(2))

    return metrics


def format_metric(r, key):
    """Format metric with ± std if available."""
    val = r.get(key)
    std = r.get(f"{key}_std")
    if val is None:
        return "-"
    if std is not None:
        if key == "SI-SDR":
            return f"{val:.1f} ± {std:.1f}"
        else:
            return f"{val:.2f} ± {std:.2f}"
    else:
        if key == "SI-SDR":
            return f"{val:.1f}"
        else:
            return f"{val:.2f}"


def save_results(all_results, output_path):
    """Save results to CSV and markdown files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as CSV
    csv_path = output_path + ".csv"
    with open(csv_path, "w") as f:
        headers = ["experiment", "dataset", "cfg_scale", "PESQ", "PESQ_std", "ESTOI", "ESTOI_std", "SI-SDR", "SI-SDR_std"]
        f.write(",".join(headers) + "\n")
        for r in all_results:
            row = [str(r.get(h, "")) for h in headers]
            f.write(",".join(row) + "\n")
    print(f"\nResults saved to: {csv_path}")

    # Save as Markdown table
    md_path = output_path + ".md"
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # In-distribution results
        f.write("## In-Distribution (VB-DEMAND)\n\n")
        f.write("| Experiment | CFG Scale | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |\n")
        f.write("|------------|-----------|--------|---------|----------|\n")
        for r in all_results:
            if r["dataset"] == "VB-DEMAND":
                f.write(f"| {r['experiment']} | {r['cfg_scale']} | {format_metric(r, 'PESQ')} | {format_metric(r, 'ESTOI')} | {format_metric(r, 'SI-SDR')} |\n")

        # OOD results
        f.write("\n## Out-of-Distribution (ESC-50 Noise, SNR 0dB)\n\n")
        f.write("| Experiment | CFG Scale | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |\n")
        f.write("|------------|-----------|--------|---------|----------|\n")
        for r in all_results:
            if r["dataset"] != "VB-DEMAND":
                f.write(f"| {r['experiment']} | {r['cfg_scale']} | {format_metric(r, 'PESQ')} | {format_metric(r, 'ESTOI')} | {format_metric(r, 'SI-SDR')} |\n")

    print(f"Results saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["enhance", "metrics", "all"], default="all")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--output", type=str, default=None, help="Output file path (without extension)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(RESULTS_DIR, f"eval_{timestamp}")

    # Build task list
    tasks = []
    for exp_name, ckpt_path, cfg_scale, is_noise_cond in EXPERIMENTS:
        for dataset_suffix, test_dir, clean_dir, noisy_dir in DATASETS:
            tasks.append({
                "exp_name": exp_name,
                "ckpt_path": ckpt_path,
                "cfg_scale": cfg_scale,
                "is_noise_cond": is_noise_cond,
                "dataset_suffix": dataset_suffix,
                "test_dir": test_dir,
                "clean_dir": clean_dir,
                "noisy_dir": noisy_dir,
            })

    # Enhancement phase
    if args.phase in ["enhance", "all"]:
        print(f"Running enhancement with GPUs: {gpu_ids}")
        print(f"Total tasks: {len(tasks)}")

        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for i, task in enumerate(tasks):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(
                    run_enhancement,
                    gpu_id,
                    task["exp_name"],
                    task["ckpt_path"],
                    task["cfg_scale"],
                    task["is_noise_cond"],
                    task["dataset_suffix"],
                    task["test_dir"],
                    task["clean_dir"]
                )
                futures.append(future)

            # Wait for all to complete
            for future in futures:
                future.result()

        print("\nEnhancement complete!")

    # Metrics phase
    if args.phase in ["metrics", "all"]:
        print("\nCalculating metrics...")

        all_results = []
        for task in tasks:
            metrics = run_metrics(
                task["exp_name"],
                task["cfg_scale"],
                task["dataset_suffix"],
                task["clean_dir"],
                task["noisy_dir"]
            )
            all_results.append(metrics)

        # Save consolidated results
        save_results(all_results, args.output)

        print("\nMetrics complete!")


if __name__ == "__main__":
    main()
