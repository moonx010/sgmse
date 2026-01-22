#!/usr/bin/env python3
"""
Prepare VOiCES dataset for reverb evaluation.

VOiCES provides real recorded reverberant speech with matching clean sources.
We use the 'none' distractor condition (no added noise, pure reverb).

Dataset structure:
- source-16k/test/sp{ID}/  -> clean files
- distant-16k/speech/test/rm{1-4}/none/sp{ID}/  -> reverb files

Matching: sp{speaker}-ch{chapter}-sg{segment}

Usage:
    python preprocessing/prepare_voices_reverb.py \
        --voices_dir data/VOiCES_devkit \
        --target_dir data/test_mixtures/voices_reverb \
        --room rm1 \
        --mic_position clo
"""

import os
import re
import argparse
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import soundfile as sf


def parse_clean_filename(filename):
    """Parse clean filename to extract matching key."""
    # Lab41-SRI-VOiCES-src-sp0303-ch123507-sg0017.wav
    match = re.search(r'sp(\d+)-ch(\d+)-sg(\d+)', filename)
    if match:
        return f"sp{match.group(1)}-ch{match.group(2)}-sg{match.group(3)}"
    return None


def parse_reverb_filename(filename):
    """Parse reverb filename to extract matching key and metadata."""
    # Lab41-SRI-VOiCES-rm4-babb-sp4899-ch032639-sg0029-mc01-stu-clo-dg160.wav
    match = re.search(r'sp(\d+)-ch(\d+)-sg(\d+)-mc(\d+)-(\w+)-(\w+)-dg(\d+)', filename)
    if match:
        return {
            'key': f"sp{match.group(1)}-ch{match.group(2)}-sg{match.group(3)}",
            'mic_id': match.group(4),
            'mic_type': match.group(5),  # stu (studio) or lav (lavalier)
            'position': match.group(6),  # clo (close), mid, far, etc.
            'angle': match.group(7)
        }
    return None


def find_matching_pairs(voices_dir, room='rm1', distractor='none', mic_position='clo'):
    """Find matching clean-reverb pairs."""

    # Get all clean files
    clean_dir = os.path.join(voices_dir, 'source-16k', 'test')
    clean_files = glob(os.path.join(clean_dir, '**', '*.wav'), recursive=True)

    # Build clean file index
    clean_index = {}
    for f in clean_files:
        key = parse_clean_filename(os.path.basename(f))
        if key:
            clean_index[key] = f

    print(f"Found {len(clean_index)} clean files")

    # Get reverb files for specified room and distractor condition
    reverb_dir = os.path.join(voices_dir, 'distant-16k', 'speech', 'test', room, distractor)

    if not os.path.exists(reverb_dir):
        print(f"ERROR: Reverb directory not found: {reverb_dir}")
        print(f"Available rooms: {os.listdir(os.path.join(voices_dir, 'distant-16k', 'speech', 'test'))}")
        return []

    reverb_files = glob(os.path.join(reverb_dir, '**', '*.wav'), recursive=True)
    print(f"Found {len(reverb_files)} reverb files in {room}/{distractor}")

    # Find matching pairs
    pairs = []
    position_stats = defaultdict(int)

    for reverb_path in reverb_files:
        parsed = parse_reverb_filename(os.path.basename(reverb_path))
        if not parsed:
            continue

        position_stats[parsed['position']] += 1

        # Filter by mic position if specified
        if mic_position and parsed['position'] != mic_position:
            continue

        key = parsed['key']
        if key in clean_index:
            pairs.append({
                'clean': clean_index[key],
                'reverb': reverb_path,
                'key': key,
                'room': room,
                'mic_position': parsed['position'],
                'mic_type': parsed['mic_type'],
                'angle': parsed['angle']
            })

    print(f"\nMic position distribution: {dict(position_stats)}")
    print(f"Found {len(pairs)} matching pairs for position '{mic_position}'")

    return pairs


def prepare_dataset(voices_dir, target_dir, room='rm1', distractor='none',
                   mic_position='clo', max_files=None):
    """Prepare VOiCES dataset for evaluation."""

    pairs = find_matching_pairs(voices_dir, room, distractor, mic_position)

    if not pairs:
        print("No matching pairs found!")
        return

    if max_files:
        pairs = pairs[:max_files]
        print(f"Using first {max_files} pairs")

    # Create output directories
    clean_out = os.path.join(target_dir, 'clean')
    noisy_out = os.path.join(target_dir, 'noisy')  # reverb as "noisy"

    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)

    # Process pairs
    print(f"\nProcessing {len(pairs)} pairs...")
    for pair in tqdm(pairs):
        # Read files
        clean, sr_clean = sf.read(pair['clean'])
        reverb, sr_reverb = sf.read(pair['reverb'])

        assert sr_clean == sr_reverb == 16000, f"Expected 16kHz, got {sr_clean}, {sr_reverb}"

        # Trim to same length (reverb might be slightly longer)
        min_len = min(len(clean), len(reverb))
        clean = clean[:min_len]
        reverb = reverb[:min_len]

        # Output filename includes metadata
        out_name = f"{pair['key']}_{pair['room']}_{pair['mic_position']}.wav"

        sf.write(os.path.join(clean_out, out_name), clean, 16000)
        sf.write(os.path.join(noisy_out, out_name), reverb, 16000)

    print(f"\nDone! Files saved to:")
    print(f"  Clean: {clean_out}")
    print(f"  Reverb: {noisy_out}")
    print(f"  Total pairs: {len(pairs)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VOiCES dataset for reverb evaluation")
    parser.add_argument("--voices_dir", type=str, required=True,
                       help="Path to VOiCES_devkit directory")
    parser.add_argument("--target_dir", type=str, required=True,
                       help="Target directory for processed files")
    parser.add_argument("--room", type=str, default="rm1",
                       choices=['rm1', 'rm2', 'rm3', 'rm4'],
                       help="Room to use (default: rm1)")
    parser.add_argument("--distractor", type=str, default="none",
                       choices=['none', 'babb', 'musi'],
                       help="Distractor condition (default: none = pure reverb)")
    parser.add_argument("--mic_position", type=str, default="clo",
                       help="Mic position filter: clo (close), mid, far, etc. Use 'all' for all positions")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--list_only", action="store_true",
                       help="Only list available options without processing")
    args = parser.parse_args()

    if args.list_only:
        # List available rooms and conditions
        speech_dir = os.path.join(args.voices_dir, 'distant-16k', 'speech', 'test')
        print(f"Available rooms: {os.listdir(speech_dir)}")
        for room in os.listdir(speech_dir):
            room_path = os.path.join(speech_dir, room)
            if os.path.isdir(room_path):
                print(f"  {room}: {os.listdir(room_path)}")
        return

    mic_pos = None if args.mic_position == 'all' else args.mic_position
    prepare_dataset(args.voices_dir, args.target_dir, args.room,
                   args.distractor, mic_pos, args.max_files)


if __name__ == "__main__":
    main()
