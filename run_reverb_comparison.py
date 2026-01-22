#!/usr/bin/env python3
"""
Run reverb comparison experiment.

Compare VB-DEMAND (additive noise) vs WSJ0-REVERB (dereverberation) models
on Edinburgh Reverb test data.

Usage:
    # Step 1: Download checkpoints
    gdown 1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ -O checkpoints/vb_demand.ckpt
    gdown 1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD -O checkpoints/wsj0_reverb.ckpt

    # Step 2: Prepare Edinburgh reverb data
    python preprocessing/prepare_edinburgh_reverb.py \
        --edinburgh_dir data/edinburgh_reverb \
        --target_dir data/test_mixtures/edinburgh_reverb

    # Step 3: Run comparison
    python run_reverb_comparison.py --config reverb_comparison_config.yaml

    # Or run each model separately:
    python run_reverb_comparison.py --model vb_demand
    python run_reverb_comparison.py --model wsj0_reverb
"""

import os
import argparse
import yaml
import subprocess
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_enhancement(checkpoint, test_dir, enhanced_dir, sampler_config):
    """Run enhancement.py with given parameters."""
    cmd = [
        "python", "enhancement.py",
        "--test_dir", test_dir,
        "--enhanced_dir", enhanced_dir,
        "--ckpt", checkpoint,
        "--N", str(sampler_config['N']),
        "--corrector", sampler_config['corrector'],
        "--corrector_steps", str(sampler_config['corrector_steps']),
        "--snr", str(sampler_config['snr']),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_metrics(clean_dir, noisy_dir, enhanced_dir):
    """Run calc_metrics.py to evaluate."""
    cmd = [
        "python", "calc_metrics.py",
        "--clean_dir", clean_dir,
        "--noisy_dir", noisy_dir,
        "--enhanced_dir", enhanced_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run reverb comparison experiment")
    parser.add_argument("--config", type=str, default="reverb_comparison_config.yaml",
                       help="Path to config file")
    parser.add_argument("--model", type=str, choices=['vb_demand', 'wsj0_reverb', 'both'],
                       default='both', help="Which model to run")
    parser.add_argument("--skip_enhancement", action="store_true",
                       help="Skip enhancement, only run metrics")
    args = parser.parse_args()

    config = load_config(args.config)

    models_to_run = []
    if args.model == 'both':
        models_to_run = ['vb_demand', 'wsj0_reverb']
    else:
        models_to_run = [args.model]

    # Get test data info
    test_data = config['test_data']['edinburgh_reverb']
    clean_dir = test_data['clean_dir']
    noisy_dir = test_data['noisy_dir']

    for model_name in models_to_run:
        print("\n" + "="*60)
        print(f"Processing model: {model_name}")
        print("="*60)

        model_config = config['models'][model_name]
        checkpoint = model_config['checkpoint']
        sampler = model_config['sampler']
        enhanced_dir = test_data['enhanced_dirs'][model_name]

        # Check checkpoint exists
        if not os.path.exists(checkpoint):
            print(f"WARNING: Checkpoint not found: {checkpoint}")
            print(f"Download with: gdown <id> -O {checkpoint}")
            continue

        # Create output directory
        os.makedirs(enhanced_dir, exist_ok=True)

        # Run enhancement
        if not args.skip_enhancement:
            run_enhancement(checkpoint, noisy_dir, enhanced_dir, sampler)

        # Run metrics
        run_metrics(clean_dir, noisy_dir, enhanced_dir)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nResults saved in:")
    for model_name in models_to_run:
        enhanced_dir = test_data['enhanced_dirs'][model_name]
        print(f"  {model_name}: {enhanced_dir}/_results.csv")

    print("\nTo visualize comparison:")
    print("  python visualize_reverb_comparison.py")


if __name__ == "__main__":
    main()
