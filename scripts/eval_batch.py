#!/usr/bin/env python
"""Batch evaluation script for noise-conditioned models.

Usage:
    python scripts/eval_batch.py --phase enhance --gpus 0,1,2,3
    python scripts/eval_batch.py --phase metrics
"""

import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor

# ============================================
# CONFIGURATION - Modify this section
# ============================================

EXPERIMENTS = [
    # (name, checkpoint_path, cfg_scale)
    ("nc_ref0.25s", "./logs/zqtm721z-nc-ref-0.25s/step=50000.ckpt", 1.0),
    ("nc_ref0.5s", "./logs/7tucs4jy-nc-ref-0.5s/step=50000.ckpt", 1.0),
    ("nc_ref1.0s", "./logs/sr8toljy-nc-ref-1.0s/step=50000.ckpt", 1.0),
    ("nc_ref2.0s", "./logs/5xbd359r-nc-ref-2.0s/step=50000.ckpt", 1.0),
]

DATASETS = [
    # (name_suffix, test_dir, clean_dir, noisy_dir)
    ("", "./data/voicebank-demand/test", "./data/voicebank-demand/test/clean", "./data/voicebank-demand/test/noisy"),
    ("_ood", "./data/test_mixtures/vb_esc50/snr_0dB", "./data/test_mixtures/vb_esc50/snr_0dB/clean", "./data/test_mixtures/vb_esc50/snr_0dB/noisy"),
]

N_STEPS = 30
OUTPUT_BASE = "./enhanced"

# ============================================
# END CONFIGURATION
# ============================================


def run_enhancement(gpu_id, exp_name, ckpt_path, cfg_scale, dataset_suffix, test_dir, clean_dir):
    """Run enhancement on a single GPU."""
    output_dir = f"{OUTPUT_BASE}_{exp_name}{dataset_suffix}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

    print(f"[GPU {gpu_id}] Starting: {exp_name}{dataset_suffix}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[GPU {gpu_id}] ERROR: {exp_name}{dataset_suffix}")
        print(result.stderr)
    else:
        print(f"[GPU {gpu_id}] Done: {exp_name}{dataset_suffix}")

    return result.returncode


def run_metrics(exp_name, dataset_suffix, clean_dir, noisy_dir):
    """Run metrics calculation."""
    output_dir = f"{OUTPUT_BASE}_{exp_name}{dataset_suffix}"

    cmd = [
        "python", "calc_metrics.py",
        "--clean_dir", clean_dir,
        "--noisy_dir", noisy_dir,
        "--enhanced_dir", output_dir
    ]

    print(f"\n=== {exp_name}{dataset_suffix} ===")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["enhance", "metrics", "all"], default="all")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    # Build task list
    tasks = []
    for exp_name, ckpt_path, cfg_scale in EXPERIMENTS:
        for dataset_suffix, test_dir, clean_dir, noisy_dir in DATASETS:
            tasks.append({
                "exp_name": exp_name,
                "ckpt_path": ckpt_path,
                "cfg_scale": cfg_scale,
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

        for task in tasks:
            run_metrics(
                task["exp_name"],
                task["dataset_suffix"],
                task["clean_dir"],
                task["noisy_dir"]
            )

        print("\nMetrics complete!")


if __name__ == "__main__":
    main()
