#!/usr/bin/env python3
"""
Split ESC-50 dataset by category.

Creates symlinks or copies files into category-based subdirectories.

Usage:
    python preprocessing/split_esc50_by_category.py \
        --esc50_dir data/ESC-50-master \
        --output_dir data/ESC-50-categories
"""

import os
import argparse
import shutil
import pandas as pd
from pathlib import Path

# ESC-50 category mapping (target number -> major category)
CATEGORY_MAP = {
    # Animals (0-9)
    0: 'animals', 1: 'animals', 2: 'animals', 3: 'animals', 4: 'animals',
    5: 'animals', 6: 'animals', 7: 'animals', 8: 'animals', 9: 'animals',
    # Natural (10-19)
    10: 'natural', 11: 'natural', 12: 'natural', 13: 'natural', 14: 'natural',
    15: 'natural', 16: 'natural', 17: 'natural', 18: 'natural', 19: 'natural',
    # Human non-speech (20-29)
    20: 'human', 21: 'human', 22: 'human', 23: 'human', 24: 'human',
    25: 'human', 26: 'human', 27: 'human', 28: 'human', 29: 'human',
    # Domestic (30-39)
    30: 'domestic', 31: 'domestic', 32: 'domestic', 33: 'domestic', 34: 'domestic',
    35: 'domestic', 36: 'domestic', 37: 'domestic', 38: 'domestic', 39: 'domestic',
    # Urban (40-49)
    40: 'urban', 41: 'urban', 42: 'urban', 43: 'urban', 44: 'urban',
    45: 'urban', 46: 'urban', 47: 'urban', 48: 'urban', 49: 'urban',
}

CATEGORY_DESCRIPTIONS = {
    'animals': 'Animals (dog, cat, birds, insects, etc.)',
    'natural': 'Natural (rain, thunder, wind, fire, etc.)',
    'human': 'Human non-speech (coughing, laughing, footsteps, etc.)',
    'domestic': 'Domestic (keyboard, vacuum, door, clock, etc.)',
    'urban': 'Urban (siren, car horn, helicopter, train, etc.)',
}


def main():
    parser = argparse.ArgumentParser(
        description="Split ESC-50 dataset by category"
    )
    parser.add_argument("--esc50_dir", type=str, required=True,
                       help="Path to ESC-50-master directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for category-split files")
    parser.add_argument("--copy", action="store_true",
                       help="Copy files instead of creating symlinks")
    args = parser.parse_args()

    # Read metadata
    meta_path = os.path.join(args.esc50_dir, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    audio_dir = os.path.join(args.esc50_dir, "audio")

    # Create output directories
    for category in CATEGORY_MAP.values():
        cat_dir = Path(args.output_dir) / category
        cat_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    print(f"Processing {len(df)} files...")
    category_counts = {cat: 0 for cat in set(CATEGORY_MAP.values())}

    for _, row in df.iterrows():
        filename = row['filename']
        target = row['target']
        category = CATEGORY_MAP.get(target)

        if category is None:
            print(f"Warning: Unknown target {target} for {filename}")
            continue

        src_path = os.path.join(audio_dir, filename)
        dst_path = os.path.join(args.output_dir, category, filename)

        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue

        if args.copy:
            shutil.copy2(src_path, dst_path)
        else:
            # Create relative symlink
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(os.path.abspath(src_path), dst_path)

        category_counts[category] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("ESC-50 CATEGORY SPLIT SUMMARY")
    print("=" * 60)
    for category, count in sorted(category_counts.items()):
        desc = CATEGORY_DESCRIPTIONS.get(category, '')
        print(f"  {category:12} : {count:4} files  - {desc}")
    print("=" * 60)
    print(f"Total: {sum(category_counts.values())} files")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
