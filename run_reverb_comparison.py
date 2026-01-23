#!/usr/bin/env python3
"""
Run reverb comparison experiment.

Compare VB-DEMAND (additive noise) vs WSJ0-REVERB (dereverberation) models
on VOiCES Reverb test data.

Usage:
    # Step 1: Download checkpoints
    gdown 1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ -O checkpoints/sgmse_vb-dmd.ckpt
    gdown 1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD -O checkpoints/wsj0_reverb.ckpt

    # Step 2: Prepare VOiCES reverb data
    python preprocessing/prepare_voices_reverb.py \
        --voices_dir data/VOiCES_devkit \
        --target_dir data/test_mixtures/voices_reverb

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
from glob import glob
from os.path import join

import pandas as pd
import librosa
from soundfile import read
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

# Import pure si_sdr function (doesn't need noise, works for reverb)
from sgmse.util.other import si_sdr, mean_std


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


def calc_metrics_reverb(clean_dir, noisy_dir, enhanced_dir):
    """
    Calculate metrics for reverb experiments.

    Uses pure si_sdr(s, s_hat) instead of energy_ratios(s_hat, s, n).
    For reverb, n = y - x is meaningless since reverb is convolutive (y = x * h),
    not additive (y = x + n).
    """
    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": []}

    # Get noisy files
    noisy_files = []
    noisy_files += sorted(glob(join(noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(noisy_dir, '**', '*.wav')))

    print(f"\nCalculating metrics for {len(noisy_files)} files...")

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(noisy_dir, "")[1:]

        clean_path = join(clean_dir, filename)
        enhanced_path = join(enhanced_dir, filename)

        if not os.path.exists(clean_path):
            print(f"WARNING: Clean file not found: {clean_path}")
            continue
        if not os.path.exists(enhanced_path):
            print(f"WARNING: Enhanced file not found: {enhanced_path}")
            continue

        x, sr_x = read(clean_path)
        x_hat, sr_x_hat = read(enhanced_path)

        # Ensure same sample rate
        if sr_x != sr_x_hat:
            x_hat = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=sr_x)
            sr_x_hat = sr_x

        # Ensure same length
        min_len = min(len(x), len(x_hat))
        x = x[:min_len]
        x_hat = x_hat[:min_len]

        # Resample to 16kHz for PESQ if needed
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat

        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        # Use pure si_sdr (no noise needed, works for reverb)
        data["si_sdr"].append(si_sdr(x, x_hat))

    # Save results as DataFrame
    df = pd.DataFrame(data)

    # Print results
    print("\n" + "-"*50)
    print("RESULTS (using pure SI-SDR for reverb)")
    print("-"*50)
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))

    # Save average results to file
    log = open(join(enhanced_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())) + "\n")
    log.write("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())) + "\n")
    log.close()

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
    print(f"Results saved to: {enhanced_dir}/_results.csv")


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
    test_data = config['test_data']['voices_reverb']
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

        # Run metrics (using reverb-specific calculation with pure si_sdr)
        calc_metrics_reverb(clean_dir, noisy_dir, enhanced_dir)

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
