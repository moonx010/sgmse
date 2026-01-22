"""
Noise Encoder for extracting noise environment embeddings from reference signals.

This module implements a lightweight encoder that takes a short noise-only
reference signal and produces a fixed-dimensional embedding vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseEncoder(nn.Module):
    """
    Encodes a short noise reference signal into a latent embedding.

    The encoder processes the complex STFT spectrogram of the noise reference,
    applying temporal pooling to handle variable-length inputs and producing
    a fixed-dimensional embedding that captures the noise characteristics.

    Args:
        in_channels: Number of input channels (2 for real/imag parts)
        nf: Base number of filters
        embed_dim: Output embedding dimension
        num_layers: Number of convolutional layers
    """

    def __init__(
        self,
        in_channels: int = 2,
        nf: int = 64,
        embed_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Build encoder layers
        layers = []
        ch_in = in_channels
        ch_out = nf

        for i in range(num_layers):
            layers.append(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.GroupNorm(num_groups=min(ch_out // 4, 32), num_channels=ch_out))
            layers.append(nn.SiLU())
            ch_in = ch_out
            if i < num_layers - 1:
                ch_out = min(ch_out * 2, 512)

        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 1))  # Pool to [B, C, 4, 1]

        # Final projection to embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(ch_in * 4, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Encode a noise reference signal.

        Args:
            r: Complex spectrogram of noise reference [B, 1, F, T] or [B, F, T]
               Can be complex tensor or already split into real/imag channels [B, 2, F, T]

        Returns:
            z_r: Noise embedding [B, embed_dim]
        """
        # Handle different input formats
        if r.dim() == 3:
            r = r.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]

        if torch.is_complex(r):
            # Split complex into real/imag channels
            x = torch.cat([r.real, r.imag], dim=1)  # [B, 2, F, T]
        elif r.shape[1] == 1:
            # Single channel complex stored as real
            x = torch.cat([r, torch.zeros_like(r)], dim=1)
        else:
            # Already in [B, 2, F, T] format
            x = r

        # Encode
        h = self.conv_layers(x)  # [B, C, H', W']
        h = self.adaptive_pool(h)  # [B, C, 4, 1]
        h = h.flatten(start_dim=1)  # [B, C*4]
        z_r = self.fc(h)  # [B, embed_dim]

        return z_r


class NoiseEncoderLight(nn.Module):
    """
    Lightweight noise encoder using only 1D convolutions along frequency axis.

    This variant is more parameter-efficient and suitable for PoC experiments.
    """

    def __init__(
        self,
        n_freq: int = 256,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Process frequency dimension with 1D convs
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),  # 2 for real/imag
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Complex spectrogram [B, 1, F, T] or [B, F, T]

        Returns:
            z_r: Noise embedding [B, embed_dim]
        """
        if r.dim() == 3:
            r = r.unsqueeze(1)

        if torch.is_complex(r):
            x = torch.cat([r.real, r.imag], dim=1)  # [B, 2, F, T]
        else:
            x = r

        # Average over time first
        x = x.mean(dim=-1)  # [B, 2, F]

        # Encode frequency structure
        h = self.freq_encoder(x)  # [B, 256, 1]
        h = h.squeeze(-1)  # [B, 256]
        z_r = self.fc(h)  # [B, embed_dim]

        return z_r
