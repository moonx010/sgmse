#!/usr/bin/env python
"""Calculate total duration of audio files in a directory."""

import argparse
import soundfile as sf
from glob import glob
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Calculate total audio duration')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--recursive', action='store_true', help='Search recursively')
    args = parser.parse_args()

    pattern = f"{args.dir}/**/*.wav" if args.recursive else f"{args.dir}/*.wav"
    files = glob(pattern, recursive=args.recursive)

    if not files:
        print(f"No .wav files found in {args.dir}")
        return

    total_sec = 0
    for f in tqdm(files, desc="Calculating duration"):
        try:
            info = sf.info(f)
            total_sec += info.duration
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    print(f"\nTotal files: {len(files)}")
    print(f"Total duration: {total_sec:.1f} sec = {total_sec/60:.1f} min = {total_sec/3600:.2f} hours")


if __name__ == '__main__':
    main()
