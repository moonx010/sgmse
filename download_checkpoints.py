#!/usr/bin/env python3
"""
Download pretrained SGMSE model checkpoints.

Usage:
    python download_checkpoints.py --help
    python download_checkpoints.py --all
    python download_checkpoints.py --model vb-dmd
"""

import os
import argparse
import subprocess
from pathlib import Path


CHECKPOINTS = {
    # Speech Enhancement Models
    "vb-dmd": {
        "name": "VoiceBank-DEMAND",
        "type": "gdown",
        "id": "1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ",
        "filename": "sgmse_vb-dmd.ckpt",
        "description": "SGMSE+ trained on VoiceBank-DEMAND dataset (16kHz)",
        "paper": "[2] TASLP 2023",
    },
    "wsj0-chime3": {
        "name": "WSJ0-CHiME3",
        "type": "gdown",
        "id": "16K4DUdpmLhDNC7pJhBBc08pkSIn_yMPi",
        "filename": "sgmse_wsj0-chime3.ckpt",
        "description": "SGMSE+ trained on WSJ0-CHiME3 dataset (16kHz)",
        "paper": "[2] TASLP 2023",
    },

    # Dereverberation Models
    "wsj0-reverb": {
        "name": "WSJ0-REVERB",
        "type": "gdown",
        "id": "1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD",
        "filename": "sgmse_wsj0-reverb.ckpt",
        "description": "SGMSE+ trained on WSJ0-REVERB dataset (16kHz). Use with --N 50 --snr 0.33",
        "paper": "[2] TASLP 2023",
    },

    # 48 kHz Models
    "ears-wham": {
        "name": "EARS-WHAM",
        "type": "gdown",
        "id": "1t_DLLk8iPH6nj8M5wGeOP3jFPaz3i7K5",
        "filename": "sgmse_ears-wham_48k.ckpt",
        "description": "SGMSE+ trained on EARS-WHAM dataset (48kHz)",
        "paper": "[3] Interspeech 2024",
    },
    "ears-reverb": {
        "name": "EARS-Reverb",
        "type": "gdown",
        "id": "1PunXuLbuyGkknQCn_y-RCV2dTZBhyE3V",
        "filename": "sgmse_ears-reverb_48k.ckpt",
        "description": "SGMSE+ trained on EARS-Reverb dataset (48kHz)",
        "paper": "[3] Interspeech 2024",
    },

    # Training Objectives Models (M1-M8)
    "m1": {
        "name": "Training-Obj-M1",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m1.ckpt",
        "filename": "m1.ckpt",
        "description": "Training objectives model M1",
        "paper": "[4] ICASSP 2025",
    },
    "m2": {
        "name": "Training-Obj-M2",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m2.ckpt",
        "filename": "m2.ckpt",
        "description": "Training objectives model M2",
        "paper": "[4] ICASSP 2025",
    },
    "m3": {
        "name": "Training-Obj-M3",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m3.ckpt",
        "filename": "m3.ckpt",
        "description": "Training objectives model M3",
        "paper": "[4] ICASSP 2025",
    },
    "m4": {
        "name": "Training-Obj-M4",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m4.ckpt",
        "filename": "m4.ckpt",
        "description": "Training objectives model M4",
        "paper": "[4] ICASSP 2025",
    },
    "m5": {
        "name": "Training-Obj-M5",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m5.ckpt",
        "filename": "m5.ckpt",
        "description": "Training objectives model M5",
        "paper": "[4] ICASSP 2025",
    },
    "m6": {
        "name": "Training-Obj-M6",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m6.ckpt",
        "filename": "m6.ckpt",
        "description": "Training objectives model M6",
        "paper": "[4] ICASSP 2025",
    },
    "m7": {
        "name": "Training-Obj-M7",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m7.ckpt",
        "filename": "m7.ckpt",
        "description": "Training objectives model M7",
        "paper": "[4] ICASSP 2025",
    },
    "m8": {
        "name": "Training-Obj-M8",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m8.ckpt",
        "filename": "m8.ckpt",
        "description": "Training objectives model M8",
        "paper": "[4] ICASSP 2025",
    },

    # Schrödinger Bridge Model
    "sb-vbdmd-earswham": {
        "name": "Schrodinger-Bridge",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/lipdiffuser/checkpoints/SB_VB-DMD_EARS-WHAM_900k.pt",
        "filename": "sb_vbdmd_earswham.pt",
        "description": "Schrödinger Bridge trained on VB-DMD + EARS-WHAM (16kHz)",
        "paper": "[4] ICASSP 2025",
    },

    # ReverbFX Models
    "reverbfx-sgmse-artificial": {
        "name": "ReverbFX-SGMSE-Artificial",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/itg2025-reverbfx/checkpoints/sgmse_artificial_rir_350k.ckpt",
        "filename": "reverbfx_sgmse_artificial_rir.ckpt",
        "description": "SGMSE+ trained on Singing-ReverbFX with artificial RIR",
        "paper": "ITG 2025 ReverbFX",
    },
    "reverbfx-sgmse-natural": {
        "name": "ReverbFX-SGMSE-Natural",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/itg2025-reverbfx/checkpoints/sgmse_natural_rir_350k.ckpt",
        "filename": "reverbfx_sgmse_natural_rir.ckpt",
        "description": "SGMSE+ trained on Singing-ReverbFX with natural RIR",
        "paper": "ITG 2025 ReverbFX",
    },
    "reverbfx-sb-artificial": {
        "name": "ReverbFX-SB-Artificial",
        "type": "wget",
        "url": "https://www2.informatik.uni-hamburg.de/sp/audio/publications/itg2025-reverbfx/checkpoints/sb_artificial_rir_350k.ckpt",
        "filename": "reverbfx_sb_artificial_rir.ckpt",
        "description": "Schrödinger Bridge trained on Singing-ReverbFX with artificial RIR",
        "paper": "ITG 2025 ReverbFX",
    },
}


# Model groups for easy selection
MODEL_GROUPS = {
    "enhancement": ["vb-dmd", "wsj0-chime3", "ears-wham"],
    "dereverberation": ["wsj0-reverb", "ears-reverb"],
    "48khz": ["ears-wham", "ears-reverb"],
    "training-objectives": [f"m{i}" for i in range(1, 9)],
    "schrodinger-bridge": ["sb-vbdmd-earswham"],
    "reverbfx": ["reverbfx-sgmse-artificial", "reverbfx-sgmse-natural", "reverbfx-sb-artificial"],
}


def download_with_gdown(file_id, output_path):
    """Download file using gdown."""
    cmd = ["gdown", file_id, "-O", output_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_with_wget(url, output_path):
    """Download file using wget."""
    cmd = ["wget", url, "-O", output_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_checkpoint(key, output_dir="checkpoints", force=False):
    """Download a specific checkpoint."""
    if key not in CHECKPOINTS:
        print(f"Error: Unknown checkpoint '{key}'")
        print(f"Available checkpoints: {', '.join(CHECKPOINTS.keys())}")
        return False

    ckpt = CHECKPOINTS[key]
    output_path = os.path.join(output_dir, ckpt["filename"])

    # Check if already exists
    if os.path.exists(output_path) and not force:
        print(f"✓ {ckpt['name']} already exists at {output_path}")
        return True

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Downloading: {ckpt['name']}")
    print(f"Description: {ckpt['description']}")
    print(f"Paper: {ckpt['paper']}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    try:
        if ckpt["type"] == "gdown":
            download_with_gdown(ckpt["id"], output_path)
        elif ckpt["type"] == "wget":
            download_with_wget(ckpt["url"], output_path)
        else:
            print(f"Error: Unknown download type '{ckpt['type']}'")
            return False

        print(f"\n✓ Successfully downloaded: {output_path}\n")
        return True

    except Exception as e:
        print(f"\n✗ Failed to download {ckpt['name']}: {e}\n")
        return False


def list_checkpoints():
    """List all available checkpoints."""
    print("\n" + "="*80)
    print("AVAILABLE PRETRAINED CHECKPOINTS")
    print("="*80 + "\n")

    for group_name, keys in MODEL_GROUPS.items():
        print(f"\n## {group_name.upper().replace('-', ' ')}")
        print("-" * 80)
        for key in keys:
            if key in CHECKPOINTS:
                ckpt = CHECKPOINTS[key]
                print(f"  {key:25} - {ckpt['description']}")
                print(f"  {' '*25}   Paper: {ckpt['paper']}")

    print("\n" + "="*80)
    print("USAGE:")
    print("  python download_checkpoints.py --model vb-dmd")
    print("  python download_checkpoints.py --group enhancement")
    print("  python download_checkpoints.py --all")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained SGMSE model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model", type=str,
                       help="Download a specific model (e.g., vb-dmd, wsj0-chime3)")
    parser.add_argument("--group", type=str, choices=MODEL_GROUPS.keys(),
                       help="Download a group of models")
    parser.add_argument("--all", action="store_true",
                       help="Download all available checkpoints")
    parser.add_argument("--list", action="store_true",
                       help="List all available checkpoints")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints (default: checkpoints/)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if file exists")

    args = parser.parse_args()

    # List checkpoints
    if args.list:
        list_checkpoints()
        return

    # Check if gdown is installed
    try:
        subprocess.run(["gdown", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'gdown' is not installed. Install it with: pip install gdown")
        return

    # Check if wget is installed
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'wget' is not installed. Install it with: brew install wget (macOS) or apt install wget (Linux)")
        return

    # Download checkpoints
    if args.all:
        print("Downloading ALL checkpoints...")
        for key in CHECKPOINTS.keys():
            download_checkpoint(key, args.output_dir, args.force)

    elif args.group:
        keys = MODEL_GROUPS[args.group]
        print(f"Downloading {args.group} checkpoints: {', '.join(keys)}")
        for key in keys:
            download_checkpoint(key, args.output_dir, args.force)

    elif args.model:
        download_checkpoint(args.model, args.output_dir, args.force)

    else:
        parser.print_help()
        print("\nNo action specified. Use --list to see available checkpoints.\n")


if __name__ == "__main__":
    main()
