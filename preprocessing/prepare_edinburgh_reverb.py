#!/usr/bin/env python3
"""
Prepare Edinburgh Reverberant Speech Database for evaluation.

This script converts the Edinburgh dataset structure to the standard
clean/noisy format used by this project.

Dataset: https://datashare.ed.ac.uk/handle/10283/2031

Usage:
    python preprocessing/prepare_edinburgh_reverb.py \
        --edinburgh_dir data/edinburgh_reverb \
        --target_dir data/test_mixtures/edinburgh_reverb
"""

import os
import argparse
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa


def find_dataset_structure(edinburgh_dir):
    """Detect the structure of the Edinburgh dataset."""
    # Check for common folder patterns
    possible_structures = []

    # Pattern 1: clean/ and reverb/ folders
    if os.path.exists(os.path.join(edinburgh_dir, 'clean')):
        possible_structures.append('clean_reverb')

    # Pattern 2: anechoic/ and reverberant/ folders
    if os.path.exists(os.path.join(edinburgh_dir, 'anechoic')):
        possible_structures.append('anechoic_reverberant')

    # Pattern 3: Files with naming convention (e.g., *_clean.wav, *_reverb.wav)
    all_wavs = glob(os.path.join(edinburgh_dir, '**', '*.wav'), recursive=True)
    if any('clean' in f.lower() for f in all_wavs):
        possible_structures.append('naming_convention')

    # List all directories for manual inspection
    subdirs = [d for d in os.listdir(edinburgh_dir)
               if os.path.isdir(os.path.join(edinburgh_dir, d))]

    return possible_structures, subdirs, all_wavs[:10]  # Return sample files


def get_file_pairs(edinburgh_dir, structure_info):
    """
    Get pairs of (clean, reverb) files based on detected structure.
    Returns list of tuples: [(clean_path, reverb_path, output_name), ...]
    """
    pairs = []
    subdirs = structure_info['subdirs']

    print(f"Detected subdirectories: {subdirs}")

    # Try to find clean and reverb folders
    clean_folder = None
    reverb_folder = None

    for subdir in subdirs:
        subdir_lower = subdir.lower()
        if 'clean' in subdir_lower or 'anechoic' in subdir_lower or 'dry' in subdir_lower:
            clean_folder = subdir
        elif 'reverb' in subdir_lower or 'wet' in subdir_lower:
            reverb_folder = subdir

    if clean_folder and reverb_folder:
        print(f"Found clean folder: {clean_folder}")
        print(f"Found reverb folder: {reverb_folder}")

        clean_dir = os.path.join(edinburgh_dir, clean_folder)
        reverb_dir = os.path.join(edinburgh_dir, reverb_folder)

        # Get all clean files
        clean_files = sorted(glob(os.path.join(clean_dir, '**', '*.wav'), recursive=True))

        for clean_path in clean_files:
            # Try to find matching reverb file
            rel_path = os.path.relpath(clean_path, clean_dir)
            reverb_path = os.path.join(reverb_dir, rel_path)

            if os.path.exists(reverb_path):
                output_name = rel_path.replace('/', '_').replace('\\', '_')
                pairs.append((clean_path, reverb_path, output_name))
            else:
                # Try finding by basename
                basename = os.path.basename(clean_path)
                reverb_matches = glob(os.path.join(reverb_dir, '**', basename), recursive=True)
                if reverb_matches:
                    output_name = basename
                    pairs.append((clean_path, reverb_matches[0], output_name))
    else:
        print("Could not automatically detect clean/reverb folder structure.")
        print(f"Available subdirectories: {subdirs}")
        print("\nPlease specify the structure manually or check the dataset.")

        # List some sample files for inspection
        all_wavs = glob(os.path.join(edinburgh_dir, '**', '*.wav'), recursive=True)[:20]
        print(f"\nSample files found:")
        for f in all_wavs:
            print(f"  {f}")

    return pairs


def prepare_dataset(edinburgh_dir, target_dir, target_sr=16000):
    """Prepare the Edinburgh dataset for evaluation."""

    # Detect structure
    structures, subdirs, sample_files = find_dataset_structure(edinburgh_dir)
    print(f"Detected possible structures: {structures}")
    print(f"Subdirectories: {subdirs}")
    print(f"Sample files: {sample_files}")

    structure_info = {
        'structures': structures,
        'subdirs': subdirs,
        'sample_files': sample_files
    }

    # Get file pairs
    pairs = get_file_pairs(edinburgh_dir, structure_info)

    if not pairs:
        print("\nNo file pairs found. Exiting.")
        return

    print(f"\nFound {len(pairs)} file pairs")

    # Create output directories
    clean_dir = os.path.join(target_dir, 'clean')
    noisy_dir = os.path.join(target_dir, 'noisy')  # reverb goes here as "noisy"

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Process files
    print(f"\nProcessing files (target sr: {target_sr} Hz)...")
    for clean_path, reverb_path, output_name in tqdm(pairs):
        try:
            # Load files
            clean, sr_clean = sf.read(clean_path)
            reverb, sr_reverb = sf.read(reverb_path)

            # Resample if needed
            if sr_clean != target_sr:
                clean = librosa.resample(clean, orig_sr=sr_clean, target_sr=target_sr)
            if sr_reverb != target_sr:
                reverb = librosa.resample(reverb, orig_sr=sr_reverb, target_sr=target_sr)

            # Ensure same length (reverb might be longer due to tail)
            min_len = min(len(clean), len(reverb))
            clean = clean[:min_len]
            reverb = reverb[:min_len]

            # Ensure .wav extension
            if not output_name.endswith('.wav'):
                output_name = output_name + '.wav'

            # Save
            sf.write(os.path.join(clean_dir, output_name), clean, target_sr)
            sf.write(os.path.join(noisy_dir, output_name), reverb, target_sr)

        except Exception as e:
            print(f"Error processing {clean_path}: {e}")

    print(f"\nDone! Files saved to:")
    print(f"  Clean: {clean_dir}")
    print(f"  Reverb (as noisy): {noisy_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Edinburgh Reverberant Speech Database for evaluation"
    )
    parser.add_argument(
        "--edinburgh_dir", type=str, required=True,
        help="Path to extracted Edinburgh dataset directory"
    )
    parser.add_argument(
        "--target_dir", type=str, required=True,
        help="Target directory for processed files"
    )
    parser.add_argument(
        "--target_sr", type=int, default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        "--inspect_only", action="store_true",
        help="Only inspect dataset structure without processing"
    )
    args = parser.parse_args()

    if args.inspect_only:
        structures, subdirs, sample_files = find_dataset_structure(args.edinburgh_dir)
        print(f"Detected structures: {structures}")
        print(f"Subdirectories: {subdirs}")
        print(f"Sample files: {sample_files}")
    else:
        prepare_dataset(args.edinburgh_dir, args.target_dir, args.target_sr)


if __name__ == "__main__":
    main()
