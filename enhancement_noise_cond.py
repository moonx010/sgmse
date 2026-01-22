"""
Enhancement script for noise-conditioned SGMSE+.

This script performs speech enhancement using the noise-conditioned model.
It accepts a noise reference signal that conditions the enhancement process.

Usage:
    # With explicit noise reference file
    python enhancement_noise_cond.py --test_dir /path/to/noisy \
        --enhanced_dir /path/to/enhanced --ckpt /path/to/checkpoint.ckpt \
        --noise_ref /path/to/noise_reference.wav

    # Using oracle noise (requires clean files)
    python enhancement_noise_cond.py --test_dir /path/to/test \
        --enhanced_dir /path/to/enhanced --ckpt /path/to/checkpoint.ckpt \
        --oracle_noise --clean_dir /path/to/test/clean
"""

import glob
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname, basename
from argparse import ArgumentParser
from librosa import resample
import numpy as np

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.model_cond import NoiseCondScoreModel
from sgmse.util.other import pad_spec


def extract_noise_reference(y, x, ref_duration_sec=0.5, sr=16000, hop_length=128):
    """
    Extract a noise reference segment from oracle noise (y - x).

    Args:
        y: Noisy waveform [1, T]
        x: Clean waveform [1, T]
        ref_duration_sec: Duration of reference in seconds
        sr: Sample rate
        hop_length: STFT hop length

    Returns:
        r: Noise reference waveform [1, T_ref]
    """
    n = y - x
    ref_len = int(ref_duration_sec * sr)

    if n.size(-1) > ref_len:
        # Random crop
        start = np.random.randint(0, n.size(-1) - ref_len)
        r = n[..., start:start + ref_len]
    else:
        r = n

    return r


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help='Directory containing the test data (noisy subfolder)')
    parser.add_argument("--enhanced_dir", type=str, required=True,
                        help='Directory to save enhanced audio')
    parser.add_argument("--ckpt", type=str, required=True,
                        help='Path to model checkpoint')

    # Noise reference options
    parser.add_argument("--noise_ref", type=str, default=None,
                        help='Path to noise reference file (used for all files)')
    parser.add_argument("--noise_ref_dir", type=str, default=None,
                        help='Directory containing per-file noise references')
    parser.add_argument("--oracle_noise", action='store_true',
                        help='Use oracle noise (y - x) as reference')
    parser.add_argument("--clean_dir", type=str, default=None,
                        help='Directory containing clean files (for oracle noise)')
    parser.add_argument("--ref_duration", type=float, default=0.5,
                        help='Duration of noise reference in seconds')

    # Sampler options
    parser.add_argument("--sampler_type", type=str, default="pc",
                        help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str,
                        choices=("ald", "langevin", "none"), default="ald",
                        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1,
                        help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5,
                        help="SNR value for Langevin dynamics")
    parser.add_argument("--N", type=int, default=30,
                        help="Number of reverse steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = no guidance)")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03,
                        help="The minimum process time")
    args = parser.parse_args()

    # Validate noise reference options
    if not args.oracle_noise and args.noise_ref is None and args.noise_ref_dir is None:
        print("Warning: No noise reference specified. Using zero embedding (unconditional).")

    if args.oracle_noise and args.clean_dir is None:
        raise ValueError("--clean_dir required when using --oracle_noise")

    # Load model
    model = NoiseCondScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()

    # Get list of noisy files
    noisy_dir = join(args.test_dir, 'noisy') if not args.test_dir.endswith('noisy') else args.test_dir
    noisy_files = []
    noisy_files += sorted(glob.glob(join(noisy_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(noisy_dir, '**', '*.wav')))
    noisy_files += sorted(glob.glob(join(noisy_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(noisy_dir, '**', '*.flac')))

    # Determine target sample rate
    target_sr = model.sr if hasattr(model, 'sr') else 16000
    pad_mode = "reflection"

    # Load shared noise reference if provided
    shared_noise_ref = None
    if args.noise_ref is not None:
        shared_noise_ref, ref_sr = load(args.noise_ref)
        if ref_sr != target_sr:
            shared_noise_ref = torch.tensor(
                resample(shared_noise_ref.numpy(), orig_sr=ref_sr, target_sr=target_sr)
            )

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(noisy_dir, "").lstrip("/")

        # Load noisy wav
        y, sr = load(noisy_file)
        if sr != target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))

        T_orig = y.size(1)

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Get noise reference
        noise_ref = None

        if args.oracle_noise:
            # Load clean file and compute oracle noise
            clean_file = join(args.clean_dir, filename)
            x, sr_x = load(clean_file)
            if sr_x != target_sr:
                x = torch.tensor(resample(x.numpy(), orig_sr=sr_x, target_sr=target_sr))
            x = x / norm_factor  # Use same normalization

            noise_ref = extract_noise_reference(
                y, x, ref_duration_sec=args.ref_duration,
                sr=target_sr, hop_length=model.data_module.hop_length
            )

        elif args.noise_ref_dir is not None:
            # Load per-file noise reference
            ref_file = join(args.noise_ref_dir, filename)
            if glob.glob(ref_file):
                noise_ref, ref_sr = load(ref_file)
                if ref_sr != target_sr:
                    noise_ref = torch.tensor(
                        resample(noise_ref.numpy(), orig_sr=ref_sr, target_sr=target_sr)
                    )
                noise_ref = noise_ref / norm_factor

        elif shared_noise_ref is not None:
            noise_ref = shared_noise_ref / norm_factor

        # Enhance
        x_hat = model.enhance(
            y.to(args.device),
            noise_ref=noise_ref.to(args.device) if noise_ref is not None else None,
            sampler_type=args.sampler_type,
            corrector=args.corrector,
            N=args.N,
            corrector_steps=args.corrector_steps,
            snr=args.snr,
            cfg_scale=args.cfg_scale
        )

        # Renormalize
        x_hat = x_hat * norm_factor.item()

        # Write enhanced wav file
        makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat, target_sr)
