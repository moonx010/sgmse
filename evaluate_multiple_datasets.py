#!/usr/bin/env python3
"""
Evaluate a model checkpoint on multiple test datasets.

Usage:
    python evaluate_multiple_datasets.py --config datasets_config.yaml --ckpt model.ckpt
"""

import os
import yaml
import subprocess
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime


def run_enhancement(test_dir, enhanced_dir, ckpt, sampler_args):
    """Run enhancement.py on a test directory."""
    cmd = [
        "python", "enhancement.py",
        "--test_dir", test_dir,
        "--enhanced_dir", enhanced_dir,
        "--ckpt", ckpt,
    ]

    # Add sampler arguments
    for key, value in sampler_args.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*80}")
    print(f"Running enhancement on: {test_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    subprocess.run(cmd, check=True)


def run_metrics(clean_dir, noisy_dir, enhanced_dir):
    """Run calc_metrics.py to compute metrics."""
    cmd = [
        "python", "calc_metrics.py",
        "--clean_dir", clean_dir,
        "--noisy_dir", noisy_dir,
        "--enhanced_dir", enhanced_dir,
    ]

    print(f"\n{'='*80}")
    print(f"Computing metrics for: {enhanced_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    subprocess.run(cmd, check=True)


def collect_results(datasets_config, results_dir):
    """Collect all results into a single summary file."""
    summary_data = []

    for dataset in datasets_config:
        dataset_name = dataset['name']
        enhanced_dir = dataset['enhanced_dir']

        # Read results CSV
        results_csv = os.path.join(enhanced_dir, "_results.csv")
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)

            # Calculate mean and std
            summary = {
                'Dataset': dataset_name,
                'PESQ_mean': df['pesq'].mean(),
                'PESQ_std': df['pesq'].std(),
                'ESTOI_mean': df['estoi'].mean(),
                'ESTOI_std': df['estoi'].std(),
                'SI-SDR_mean': df['si_sdr'].mean(),
                'SI-SDR_std': df['si_sdr'].std(),
                'SI-SIR_mean': df['si_sir'].mean(),
                'SI-SIR_std': df['si_sir'].std(),
                'SI-SAR_mean': df['si_sar'].mean(),
                'SI-SAR_std': df['si_sar'].std(),
                'Num_files': len(df)
            }
            summary_data.append(summary)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(results_dir, f"summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_csv}")
    print(f"{'='*80}\n")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on multiple datasets")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file with dataset configurations")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--results_dir", type=str, default="evaluation_results",
                       help="Directory to save summary results")
    parser.add_argument("--skip_enhancement", action="store_true",
                       help="Skip enhancement step (use existing enhanced files)")
    parser.add_argument("--skip_metrics", action="store_true",
                       help="Skip metrics computation")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    datasets = config['datasets']
    sampler_args = config.get('sampler', {
        'N': 30,
        'corrector': 'ald',
        'corrector_steps': 1,
        'snr': 0.5,
        'sampler_type': 'pc'
    })

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Process each dataset
    for dataset in datasets:
        dataset_name = dataset['name']
        test_dir = dataset.get('test_dir') or dataset['noisy_dir']
        clean_dir = dataset['clean_dir']
        noisy_dir = dataset['noisy_dir']
        enhanced_dir = dataset['enhanced_dir']

        print(f"\n{'#'*80}")
        print(f"# Processing dataset: {dataset_name}")
        print(f"{'#'*80}\n")

        # Create enhanced directory
        os.makedirs(enhanced_dir, exist_ok=True)

        # Step 1: Enhancement
        if not args.skip_enhancement:
            run_enhancement(test_dir, enhanced_dir, args.ckpt, sampler_args)
        else:
            print(f"Skipping enhancement for {dataset_name}")

        # Step 2: Metrics
        if not args.skip_metrics:
            run_metrics(clean_dir, noisy_dir, enhanced_dir)
        else:
            print(f"Skipping metrics computation for {dataset_name}")

    # Collect and summarize results
    if not args.skip_metrics:
        summary_df = collect_results(datasets, args.results_dir)


if __name__ == "__main__":
    main()
