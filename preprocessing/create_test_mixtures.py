#!/usr/bin/env python3
"""
Create test mixtures for out-of-domain evaluation.

Combines clean speech (LibriSpeech, VoiceBank) with various noise sources
(DEMAND, ESC-50) at different SNR levels.

Usage:
    python preprocessing/create_test_mixtures.py \
        --clean_dir data/LibriSpeech/test-clean \
        --noise_dir data/DEMAND \
        --output_dir data/test_mixtures/libri_demand \
        --snr_levels 0 5 10 \
        --num_samples 50
"""

import os
import argparse
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa


def load_audio(file_path, sr=16000):
    """Load audio file and resample to target sample rate."""
    audio, orig_sr = sf.read(file_path)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def get_audio_files(directory, recursive=True):
    """Get all audio files from directory."""
    extensions = ['*.wav', '*.flac', '*.mp3', '*.ogg']
    files = []
    for ext in extensions:
        if recursive:
            files.extend(glob(os.path.join(directory, '**', ext), recursive=True))
        else:
            files.extend(glob(os.path.join(directory, ext)))
    return sorted(files)


def mix_with_snr(clean, noise, snr_db):
    """Mix clean speech with noise at specified SNR."""
    # Ensure noise is at least as long as clean
    if len(noise) < len(clean):
        # Tile noise to be long enough
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)

    # Random start position for noise
    if len(noise) > len(clean):
        start = np.random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]

    # Calculate powers
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10

    # Scale noise to achieve target SNR
    target_noise_power = clean_power * np.power(10, -snr_db / 10)
    scale = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * scale

    # Mix
    noisy = clean + scaled_noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0.99:
        noisy = noisy * 0.99 / max_val
        clean = clean * 0.99 / max_val

    return noisy, clean


def main():
    parser = argparse.ArgumentParser(
        description="Create test mixtures for evaluation"
    )
    parser.add_argument("--clean_dir", type=str, required=True,
                       help="Directory containing clean speech files")
    parser.add_argument("--noise_dir", type=str, required=True,
                       help="Directory containing noise files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for mixtures")
    parser.add_argument("--snr_levels", type=float, nargs='+', default=[0, 5, 10],
                       help="SNR levels in dB (default: 0 5 10)")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples per SNR level (default: 50)")
    parser.add_argument("--sr", type=int, default=16000,
                       help="Target sample rate (default: 16000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get file lists
    print(f"Loading clean speech files from: {args.clean_dir}")
    clean_files = get_audio_files(args.clean_dir)
    print(f"Found {len(clean_files)} clean speech files")

    print(f"Loading noise files from: {args.noise_dir}")
    noise_files = get_audio_files(args.noise_dir)
    print(f"Found {len(noise_files)} noise files")

    if len(clean_files) == 0:
        raise ValueError(f"No audio files found in {args.clean_dir}")
    if len(noise_files) == 0:
        raise ValueError(f"No audio files found in {args.noise_dir}")

    # Load all noise files
    print("Loading noise files into memory...")
    noises = []
    noise_names = []
    for nf in tqdm(noise_files):
        try:
            audio, _ = load_audio(nf, sr=args.sr)
            if len(audio) > args.sr:  # At least 1 second
                noises.append(audio)
                noise_names.append(os.path.basename(nf))
        except Exception as e:
            print(f"Warning: Could not load {nf}: {e}")

    print(f"Loaded {len(noises)} valid noise files")

    # Create output directories for each SNR level
    for snr in args.snr_levels:
        snr_str = f"snr_{int(snr)}dB" if snr >= 0 else f"snr_neg{int(abs(snr))}dB"
        clean_out = Path(args.output_dir) / snr_str / "clean"
        noisy_out = Path(args.output_dir) / snr_str / "noisy"
        clean_out.mkdir(parents=True, exist_ok=True)
        noisy_out.mkdir(parents=True, exist_ok=True)

    # Select random subset of clean files
    selected_clean = np.random.choice(
        clean_files,
        size=min(args.num_samples, len(clean_files)),
        replace=False
    )

    # Create mixtures
    print(f"\nCreating mixtures for SNR levels: {args.snr_levels}")

    for snr in args.snr_levels:
        snr_str = f"snr_{int(snr)}dB" if snr >= 0 else f"snr_neg{int(abs(snr))}dB"
        clean_out = Path(args.output_dir) / snr_str / "clean"
        noisy_out = Path(args.output_dir) / snr_str / "noisy"

        print(f"\nProcessing SNR = {snr} dB")

        for i, clean_file in enumerate(tqdm(selected_clean)):
            try:
                # Load clean speech
                clean_audio, _ = load_audio(clean_file, sr=args.sr)

                if len(clean_audio) < args.sr // 2:  # Skip very short files
                    continue

                # Select random noise
                noise_idx = np.random.randint(len(noises))
                noise_audio = noises[noise_idx]

                # Mix
                noisy_audio, clean_audio_normalized = mix_with_snr(
                    clean_audio, noise_audio, snr
                )

                # Generate filename
                clean_basename = os.path.splitext(os.path.basename(clean_file))[0]
                noise_basename = os.path.splitext(noise_names[noise_idx])[0]
                output_name = f"{clean_basename}_{noise_basename}_{int(snr)}dB.wav"

                # Save
                sf.write(clean_out / output_name, clean_audio_normalized, args.sr)
                sf.write(noisy_out / output_name, noisy_audio, args.sr)

            except Exception as e:
                print(f"Warning: Error processing {clean_file}: {e}")

    print(f"\nDone! Mixtures saved to: {args.output_dir}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for snr in args.snr_levels:
        snr_str = f"snr_{int(snr)}dB" if snr >= 0 else f"snr_neg{int(abs(snr))}dB"
        noisy_dir = Path(args.output_dir) / snr_str / "noisy"
        n_files = len(list(noisy_dir.glob("*.wav")))
        print(f"  SNR {snr:+3.0f} dB: {n_files} files")
    print("="*60)


if __name__ == "__main__":
    main()
