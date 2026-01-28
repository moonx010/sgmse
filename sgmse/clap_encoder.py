"""
CLAP-based Noise Encoder for noise-conditioned SGMSE.

Uses pre-trained CLAP (Contrastive Language-Audio Pretraining) model
to extract generalized audio representations from noise reference signals.
"""

import torch
import torch.nn as nn

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("Warning: laion-clap not installed. CLAPNoiseEncoder will not be available.")


class CLAPNoiseEncoder(nn.Module):
    """
    Noise encoder using pre-trained CLAP audio encoder.

    CLAP is trained on large-scale audio-text pairs and provides
    generalized audio representations that should transfer better
    to unseen noise types compared to training from scratch.

    NOTE: Currently only supports freeze_clap=True mode.
    Fine-tuning CLAP requires different gradient handling and is not yet implemented.

    Args:
        embed_dim: Output embedding dimension (default: 512)
        freeze_clap: Whether to freeze CLAP weights (must be True for now)
        clap_model: CLAP model variant (default: '630k-audioset-best')
    """

    def __init__(
        self,
        embed_dim: int = 512,
        freeze_clap: bool = True,
        clap_model: str = '630k-audioset-best',
    ):
        super().__init__()

        if not CLAP_AVAILABLE:
            raise ImportError(
                "laion-clap is required for CLAPNoiseEncoder. "
                "Install with: pip install laion-clap"
            )

        if not freeze_clap:
            raise NotImplementedError(
                "CLAP fine-tuning is not yet supported. "
                "Please use --freeze_clap flag. "
                "Fine-tuning requires gradient-compatible CLAP forward pass."
            )

        self.embed_dim = embed_dim
        self.freeze_clap = freeze_clap

        # Load pre-trained CLAP model
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.clap.load_ckpt()  # Load pre-trained weights

        # CLAP outputs 512-dim embeddings
        self.clap_dim = 512

        # Always freeze CLAP for now
        for param in self.clap.parameters():
            param.requires_grad = False
        self.clap.eval()

        # Projection layer to match desired embedding dimension
        if embed_dim != self.clap_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.clap_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.proj = nn.Identity()

        # CLAP expects 48kHz audio, our data is 16kHz
        self.input_sr = 16000
        self.clap_sr = 48000

    def _resample(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample from input sample rate to CLAP's expected 48kHz."""
        import torchaudio.transforms as T

        # Create resampler on the same device
        resampler = T.Resample(
            orig_freq=self.input_sr,
            new_freq=self.clap_sr
        ).to(waveform.device)

        return resampler(waveform)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Encode a noise reference waveform using CLAP.

        Args:
            r: Raw waveform tensor [B, T] at 16kHz
               or [B, 1, T] (will be squeezed)

        Returns:
            z_r: Noise embedding [B, embed_dim]
        """
        # Handle input shape
        if r.dim() == 3:
            r = r.squeeze(1)  # [B, 1, T] -> [B, T]

        # Resample to 48kHz for CLAP
        r_48k = self._resample(r)  # [B, T']

        # Get CLAP embeddings (always with no_grad for CLAP part)
        with torch.no_grad():
            emb = self._get_clap_embedding(r_48k)

        # Project to desired dimension (this part is trainable)
        z_r = self.proj(emb)

        return z_r

    def _get_clap_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Get CLAP audio embedding.

        Args:
            audio: [B, T] tensor at 48kHz

        Returns:
            embedding: [B, 512] CLAP embedding
        """
        device = audio.device

        # Normalize audio to [-1, 1] range
        audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        # CLAP's get_audio_embedding_from_data expects numpy array when use_tensor=False
        audio_np = audio.cpu().numpy()

        # Get embeddings (use_tensor=False returns numpy, then convert)
        embed_np = self.clap.get_audio_embedding_from_data(
            x=audio_np,
            use_tensor=False
        )

        # Convert to tensor
        embed = torch.from_numpy(embed_np).to(device)

        return embed


class CLAPNoiseEncoderSimple(nn.Module):
    """
    Simplified CLAP encoder that processes audio in numpy format.

    This version is more compatible with CLAP's expected input format
    but may be slower due to CPU-GPU transfers.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        freeze_clap: bool = True,
    ):
        super().__init__()

        if not CLAP_AVAILABLE:
            raise ImportError("laion-clap is required")

        self.embed_dim = embed_dim
        self.freeze_clap = freeze_clap

        # Load CLAP
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.clap.load_ckpt()

        if freeze_clap:
            for param in self.clap.parameters():
                param.requires_grad = False

        # Projection
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.input_sr = 16000
        self.clap_sr = 48000

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Raw waveform [B, T] at 16kHz

        Returns:
            z_r: [B, embed_dim]
        """
        import torchaudio.functional as F_audio

        if r.dim() == 3:
            r = r.squeeze(1)

        device = r.device
        batch_size = r.shape[0]

        # Resample to 48kHz
        r_48k = F_audio.resample(r, self.input_sr, self.clap_sr)

        # Process through CLAP
        # CLAP works best with numpy input
        r_np = r_48k.cpu().numpy()

        with torch.no_grad():
            emb = self.clap.get_audio_embedding_from_data(
                x=r_np,
                use_tensor=True
            )

        emb = emb.to(device)
        z_r = self.proj(emb)

        return z_r
