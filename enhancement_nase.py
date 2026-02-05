#!/usr/bin/env python
"""
Enhancement script for NASE-style model with CFG.

Usage:
    # Basic enhancement with oracle noise reference
    python enhancement_nase.py \
        --test_dir ./data/voicebank-demand/test \
        --enhanced_dir ./enhanced_nase \
        --ckpt ./logs/<exp_id>/last.ckpt \
        --oracle_noise

    # Enhancement with different guidance scales
    python enhancement_nase.py \
        --test_dir ./data/test_mixtures/vb_esc50/snr_0dB \
        --enhanced_dir ./enhanced_nase_w0.5 \
        --ckpt ./logs/<exp_id>/last.ckpt \
        --oracle_noise \
        --w 0.5

    # Unconditional enhancement (w=0, fallback mode)
    python enhancement_nase.py \
        --test_dir ./data/test_mixtures/vb_esc50/snr_0dB \
        --enhanced_dir ./enhanced_nase_uncond \
        --ckpt ./logs/<exp_id>/last.ckpt \
        --w 0.0
"""

import argparse
import os
from glob import glob
from tqdm import tqdm

import torch
import torchaudio
from soundfile import write as sf_write

from sgmse.model_nase import NASEScoreModel
from sgmse.util.other import pad_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test files (with noisy/ and optionally clean/ subdirs)")
    parser.add_argument("--enhanced_dir", type=str, required=True,
                        help="Directory to save enhanced files")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")

    # Noise reference options
    parser.add_argument("--oracle_noise", action='store_true',
                        help="Use oracle noise (clean - noisy) as reference")
    parser.add_argument("--clean_dir", type=str, default=None,
                        help="Directory containing clean files (for oracle noise)")
    parser.add_argument("--noise_dir", type=str, default=None,
                        help="Directory containing noise reference files")

    # CFG options
    parser.add_argument("--w", type=float, default=1.0,
                        help="CFG guidance scale (0=unconditional, 1=conditional)")

    # Sampling options
    parser.add_argument("--N", type=int, default=50,
                        help="Number of reverse diffusion steps")
    parser.add_argument("--corrector_steps", type=int, default=1,
                        help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5,
                        help="SNR for corrector")

    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.enhanced_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.ckpt}")
    model = NASEScoreModel.load_from_checkpoint(
        args.ckpt,
        map_location=args.device,
        strict=False
    )
    model.eval()
    model.to(args.device)

    # Get file lists
    noisy_dir = os.path.join(args.test_dir, "noisy") if os.path.isdir(os.path.join(args.test_dir, "noisy")) else args.test_dir
    clean_dir = args.clean_dir or os.path.join(args.test_dir, "clean")

    noisy_files = sorted(glob(os.path.join(noisy_dir, "*.wav")))
    print(f"Found {len(noisy_files)} noisy files")

    # Process files
    for noisy_file in tqdm(noisy_files, desc=f"Enhancing (w={args.w})"):
        # Load noisy audio
        y, sr = torchaudio.load(noisy_file)

        # Get noise reference
        noise_ref = None
        if args.oracle_noise and os.path.isdir(clean_dir):
            # Oracle noise: extract from noisy - clean
            clean_file = os.path.join(clean_dir, os.path.basename(noisy_file))
            if os.path.exists(clean_file):
                x, _ = torchaudio.load(clean_file)
                # Ensure same length
                min_len = min(x.shape[1], y.shape[1])
                noise_ref = y[:, :min_len] - x[:, :min_len]
            else:
                print(f"Warning: Clean file not found for {noisy_file}, using unconditional")
        elif args.noise_dir:
            # External noise reference
            noise_file = os.path.join(args.noise_dir, os.path.basename(noisy_file))
            if os.path.exists(noise_file):
                noise_ref, _ = torchaudio.load(noise_file)
            else:
                print(f"Warning: Noise reference not found for {noisy_file}, using unconditional")

        # Enhance
        with torch.no_grad():
            x_hat = model.enhance(
                y,
                noise_ref=noise_ref,
                N=args.N,
                w=args.w,
                corrector_steps=args.corrector_steps,
                snr=args.snr,
            )

        # Save enhanced audio
        out_file = os.path.join(args.enhanced_dir, os.path.basename(noisy_file))
        sf_write(out_file, x_hat, sr)

    print(f"Enhanced files saved to {args.enhanced_dir}")


if __name__ == "__main__":
    main()
