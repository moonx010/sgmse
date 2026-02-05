"""
BEATs-based Noise Encoder following NASE architecture.

Uses pre-trained BEATs model for noise embedding extraction with optional
noise classification (NC) supervision for multi-task learning.

Reference: NASE (Interspeech 2023) - https://arxiv.org/abs/2307.08029
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEATsNoiseEncoder(nn.Module):
    """
    BEATs-based noise encoder following NASE architecture.

    Key features:
    - Pre-trained BEATs encoder (12 Transformer layers, 768-dim)
    - Optional noise classification head for multi-task learning
    - Frozen BEATs weights (only projection trained)

    Args:
        embed_dim: Output embedding dimension (default: 768 to match BEATs)
        num_classes: Number of noise classes for NC task (default: 10 for DEMAND)
        freeze_beats: Whether to freeze BEATs weights
        use_nc_head: Whether to include noise classification head
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 10,
        freeze_beats: bool = True,
        use_nc_head: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_nc_head = use_nc_head

        # Load BEATs model
        try:
            from BEATs import BEATs, BEATsConfig
            self.beats_available = True

            # Load pretrained BEATs
            # Note: User needs to download BEATs checkpoint
            checkpoint_path = "./pretrained/BEATs_iter3_plus_AS2M.pt"
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                cfg = BEATsConfig(checkpoint['cfg'])
                self.beats = BEATs(cfg)
                self.beats.load_state_dict(checkpoint['model'])
                print(f"[BEATsNoiseEncoder] Loaded BEATs from {checkpoint_path}")
            except FileNotFoundError:
                print(f"[BEATsNoiseEncoder] BEATs checkpoint not found at {checkpoint_path}")
                print("[BEATsNoiseEncoder] Using random initialization (for testing only)")
                cfg = BEATsConfig({
                    'encoder_layers': 12,
                    'encoder_embed_dim': 768,
                    'encoder_ffn_embed_dim': 3072,
                    'encoder_attention_heads': 12,
                    'input_patch_size': 16,
                    'fbank_config': {'sample_rate': 16000, 'n_fft': 512, 'win_length': 400, 'hop_length': 160, 'n_mels': 128}
                })
                self.beats = BEATs(cfg)

            if freeze_beats:
                for param in self.beats.parameters():
                    param.requires_grad = False
                print("[BEATsNoiseEncoder] BEATs weights frozen")

            beats_dim = 768

        except ImportError:
            print("[BEATsNoiseEncoder] BEATs not available, using fallback CNN encoder")
            self.beats_available = False
            self.beats = self._build_fallback_encoder()
            beats_dim = 768

        # Projection layer (if embed_dim != beats_dim)
        if embed_dim != beats_dim:
            self.proj = nn.Sequential(
                nn.Linear(beats_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.proj = nn.Identity()

        # Noise classification head for multi-task learning
        if use_nc_head:
            self.nc_head = nn.Sequential(
                nn.Linear(beats_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes),
            )
        else:
            self.nc_head = None

    def _build_fallback_encoder(self):
        """Build a fallback CNN encoder when BEATs is not available."""
        return nn.Sequential(
            # Process mel spectrogram
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )

    def forward(self, x, return_nc_logits=False):
        """
        Extract noise embedding from audio.

        Args:
            x: Audio waveform [B, T] or [B, 1, T] at 16kHz
            return_nc_logits: Whether to also return NC logits

        Returns:
            z: Noise embedding [B, embed_dim]
            nc_logits: (optional) NC logits [B, num_classes]
        """
        # Ensure correct shape
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, 1, T] -> [B, T]

        if self.beats_available:
            # BEATs expects [B, T] waveform at 16kHz
            # Pad to minimum length if needed
            if x.shape[1] < 16000:  # Min 1 second
                x = F.pad(x, (0, 16000 - x.shape[1]))

            # Extract features using BEATs
            with torch.set_grad_enabled(not self.beats.training or any(p.requires_grad for p in self.beats.parameters())):
                # BEATs forward returns (features, padding_mask)
                features, _ = self.beats.extract_features(x, padding_mask=None)
                # features: [B, T', 768] where T' is number of patches

                # Global average pooling over time
                beats_emb = features.mean(dim=1)  # [B, 768]
        else:
            # Fallback: compute mel spectrogram and use CNN
            mel = self._compute_mel(x)  # [B, 1, n_mels, T']
            beats_emb = self.beats(mel)  # [B, 768]

        # Project to embed_dim
        z = self.proj(beats_emb)  # [B, embed_dim]

        if return_nc_logits and self.nc_head is not None:
            nc_logits = self.nc_head(beats_emb)  # [B, num_classes]
            return z, nc_logits
        else:
            return z

    def _compute_mel(self, x):
        """Compute mel spectrogram for fallback encoder."""
        import torchaudio.transforms as T

        # Create mel spectrogram transform
        if not hasattr(self, '_mel_transform'):
            self._mel_transform = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                n_mels=128,
            )

        # Move transform to same device
        self._mel_transform = self._mel_transform.to(x.device)

        # Compute mel spectrogram
        mel = self._mel_transform(x)  # [B, n_mels, T']
        mel = torch.log(mel + 1e-6)  # Log scale
        mel = mel.unsqueeze(1)  # [B, 1, n_mels, T']

        return mel


class SimpleNoiseEncoder(nn.Module):
    """
    Simple CNN-based noise encoder for quick experiments.

    This is a lightweight alternative when BEATs is not available or
    for ablation studies comparing encoder architectures.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 10,
        use_nc_head: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_nc_head = use_nc_head

        # Simple spectrogram encoder
        self.encoder = nn.Sequential(
            # Input: [B, 2, F, T] (real + imag)
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Noise classification head
        if use_nc_head:
            self.nc_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes),
            )
        else:
            self.nc_head = None

    def forward(self, x, return_nc_logits=False):
        """
        Args:
            x: Complex spectrogram [B, 1, F, T] or [B, 2, F, T]
            return_nc_logits: Whether to return NC logits

        Returns:
            z: Embedding [B, embed_dim]
            nc_logits: (optional) [B, num_classes]
        """
        # Handle input format
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=1)
        elif x.shape[1] == 1:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        # Encode
        h = self.encoder(x)  # [B, 512]
        z = self.proj(h)  # [B, embed_dim]

        if return_nc_logits and self.nc_head is not None:
            nc_logits = self.nc_head(h)
            return z, nc_logits
        else:
            return z
