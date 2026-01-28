"""
Data module with noise reference support for noise-conditioned SGMSE.

Extends the base SpecsDataModule to return (clean, noisy, noise_reference) tuples
where the noise reference is extracted from the oracle noise signal (y - x).
"""

from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F

from .data_module import get_window, SpecsDataModule


class NoiseCondSpecs(Dataset):
    """
    Dataset that returns (clean, noisy, noise_reference) tuples.

    The noise reference is extracted by:
    1. Computing oracle noise: n = y - x (in waveform domain)
    2. Random cropping a segment of specified length
    3. Converting to STFT spectrogram

    Optionally returns raw waveform for CLAP encoder.
    """

    def __init__(
        self,
        data_dir,
        subset,
        dummy,
        shuffle_spec,
        num_frames,
        ref_num_frames=32,  # ~0.5s at default settings
        format='default',
        normalize="noisy",
        spec_transform=None,
        stft_kwargs=None,
        return_ref_waveform=False,  # For CLAP encoder
        **ignored_kwargs
    ):
        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav")))
        elif format == "reverb":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "**", "*.wav")))
        else:
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.ref_num_frames = ref_num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.return_ref_waveform = return_ref_waveform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), \
            "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, \
            "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)

        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start:start + target_len]
            y = y[..., start:start + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
            y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode='constant')

        # Compute oracle noise in waveform domain
        n = y - x  # [1, T]

        # Extract noise reference segment
        ref_target_len = (self.ref_num_frames - 1) * self.hop_length
        n_len = n.size(-1)

        if n_len > ref_target_len:
            # Random crop for reference
            ref_start = int(np.random.uniform(0, n_len - ref_target_len))
            r = n[..., ref_start:ref_start + ref_target_len]
        else:
            # Use full noise if shorter than target
            r = n

        # normalize w.r.t to the noisy or the clean signal or not at all
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac
        r = r / normfac  # Normalize reference with same factor

        # Convert to STFT
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        R = torch.stft(r, **self.stft_kwargs)

        # Apply spectrogram transform
        X, Y, R = self.spec_transform(X), self.spec_transform(Y), self.spec_transform(R)

        if self.return_ref_waveform:
            # Return raw waveform for CLAP encoder
            return X, Y, R, r.squeeze(0)  # r: [T]
        return X, Y, R

    def __len__(self):
        if self.dummy:
            return int(len(self.clean_files) / 200)
        else:
            return len(self.clean_files)


class NoiseCondSpecsDataModule(SpecsDataModule):
    """
    DataModule for noise-conditioned SGMSE training.

    Extends SpecsDataModule to return (clean, noisy, noise_ref) tuples.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser = SpecsDataModule.add_argparse_args(parser)
        parser.add_argument("--ref_num_frames", type=int, default=32,
                            help="Number of STFT frames for noise reference. 32 (~0.5s) by default.")
        parser.add_argument("--ref_length_sec", type=float, default=None,
                            help="Alternative: specify reference length in seconds (overrides ref_num_frames)")
        parser.add_argument("--return_ref_waveform", action='store_true',
                            help="Return raw waveform for noise reference (for CLAP encoder)")
        return parser

    def __init__(
        self,
        base_dir,
        format='default',
        batch_size=8,
        n_fft=510,
        hop_length=128,
        num_frames=256,
        window='hann',
        num_workers=4,
        dummy=False,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        gpu=True,
        normalize='noisy',
        transform_type="exponent",
        ref_num_frames=32,
        ref_length_sec=None,
        return_ref_waveform=False,
        **kwargs
    ):
        super().__init__(
            base_dir=base_dir,
            format=format,
            batch_size=batch_size,
            n_fft=n_fft,
            hop_length=hop_length,
            num_frames=num_frames,
            window=window,
            num_workers=num_workers,
            dummy=dummy,
            spec_factor=spec_factor,
            spec_abs_exponent=spec_abs_exponent,
            gpu=gpu,
            normalize=normalize,
            transform_type=transform_type,
            **kwargs
        )

        # Compute ref_num_frames from seconds if specified
        if ref_length_sec is not None:
            # ref_length_sec * sample_rate / hop_length
            # Assuming 16kHz sample rate by default
            sample_rate = kwargs.get('sr', 16000)
            ref_num_frames = int(ref_length_sec * sample_rate / hop_length) + 1

        self.ref_num_frames = ref_num_frames
        self.return_ref_waveform = return_ref_waveform

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs,
            num_frames=self.num_frames,
            ref_num_frames=self.ref_num_frames,
            spec_transform=self.spec_fwd,
            return_ref_waveform=self.return_ref_waveform,
            **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = NoiseCondSpecs(
                data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize, **specs_kwargs
            )
            self.valid_set = NoiseCondSpecs(
                data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs
            )
        if stage == 'test' or stage is None:
            self.test_set = NoiseCondSpecs(
                data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs
            )
