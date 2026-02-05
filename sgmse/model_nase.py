"""
NASE-style Score Model with CFG for OOD Robustness.

Implements NASE architecture (BEATs encoder + NC loss + input addition)
combined with Classifier-Free Guidance for graceful degradation on OOD noise.

Key differences from original NASE:
- CFG dropout during training (p_uncond)
- Adaptive guidance scale at inference

Reference:
- NASE (Interspeech 2023): https://arxiv.org/abs/2307.08029
- CFG (NeurIPS 2022): https://arxiv.org/abs/2207.12598
"""

import time
from math import ceil
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.beats_encoder import BEATsNoiseEncoder, SimpleNoiseEncoder
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi


# DEMAND noise type labels (for NC loss)
DEMAND_NOISE_TYPES = [
    'DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK',
    'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE', 'PCAFETERIA',
    'PRESTO', 'PSTATION', 'SCAFE', 'SPSQUARE', 'STRAFFIC', 'TBUS', 'TCAR', 'TMETRO'
]


class NASEScoreModel(pl.LightningModule):
    """
    NASE-style score model with CFG for OOD robustness.

    Architecture:
    - BEATs/Simple noise encoder
    - NC loss for encoder supervision (multi-task learning)
    - Input addition for noise embedding injection
    - CFG dropout for OOD graceful degradation

    The model learns:
        s_θ(x_t, y, z, t) ≈ ∇_{x_t} log p_t(x_t | y, z)

    where z = E_φ(n) is the noise embedding.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--t_eps", type=float, default=0.03)
        parser.add_argument("--num_eval_files", type=int, default=20)
        parser.add_argument("--loss_type", type=str, default="score_matching")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2")
        parser.add_argument("--sr", type=int, default=16000)

        # Noise encoder
        parser.add_argument("--noise_encoder_type", type=str, default="simple",
                            choices=["beats", "simple"],
                            help="Type of noise encoder")
        parser.add_argument("--noise_embed_dim", type=int, default=768,
                            help="Noise embedding dimension")
        parser.add_argument("--freeze_encoder", action='store_true',
                            help="Freeze noise encoder weights")

        # NC loss (multi-task learning)
        parser.add_argument("--use_nc_loss", action='store_true',
                            help="Use noise classification loss")
        parser.add_argument("--nc_loss_weight", type=float, default=0.3,
                            help="Weight for NC loss (lambda in NASE)")
        parser.add_argument("--num_noise_classes", type=int, default=10,
                            help="Number of noise classes")

        # CFG
        parser.add_argument("--p_uncond", type=float, default=0.2,
                            help="CFG dropout probability")

        return parser

    def __init__(
        self,
        backbone,
        sde,
        lr=1e-4,
        ema_decay=0.999,
        t_eps=0.03,
        num_eval_files=20,
        loss_type='score_matching',
        loss_weighting='sigma^2',
        sr=16000,
        noise_encoder_type='simple',
        noise_embed_dim=768,
        freeze_encoder=False,
        use_nc_loss=True,
        nc_loss_weight=0.3,
        num_noise_classes=10,
        p_uncond=0.2,
        data_module_cls=None,
        **kwargs
    ):
        super().__init__()

        # Backbone
        self.backbone_name = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(noise_embed_dim=noise_embed_dim, **kwargs)

        # Noise encoder
        self.noise_encoder_type = noise_encoder_type
        if noise_encoder_type == 'beats':
            self.noise_encoder = BEATsNoiseEncoder(
                embed_dim=noise_embed_dim,
                num_classes=num_noise_classes,
                freeze_beats=freeze_encoder,
                use_nc_head=use_nc_loss,
            )
        else:
            self.noise_encoder = SimpleNoiseEncoder(
                embed_dim=noise_embed_dim,
                num_classes=num_noise_classes,
                use_nc_head=use_nc_loss,
            )

        if freeze_encoder:
            for param in self.noise_encoder.parameters():
                param.requires_grad = False

        # SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)

        # Hyperparameters
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.num_eval_files = num_eval_files
        self.sr = sr
        self.noise_embed_dim = noise_embed_dim
        self.use_nc_loss = use_nc_loss
        self.nc_loss_weight = nc_loss_weight
        self.num_noise_classes = num_noise_classes
        self.p_uncond = p_uncond

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                self.ema.store(self.parameters())
                self.ema.copy_to(self.parameters())
            else:
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _score_loss(self, score, z, t):
        """Compute score matching loss."""
        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z))
            else:
                raise ValueError(f"Invalid loss weighting: {self.loss_weighting}")
            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        return loss

    def _step(self, batch, batch_idx):
        """
        Training/validation step.

        Args:
            batch: (x, y, noise_ref, noise_label) tuple
                   - x: clean spectrogram
                   - y: noisy spectrogram
                   - noise_ref: noise reference (waveform or spectrogram)
                   - noise_label: noise type label (optional, for NC loss)
        """
        if len(batch) == 4:
            x, y, noise_ref, noise_label = batch
        elif len(batch) == 3:
            x, y, noise_ref = batch
            noise_label = None
        else:
            raise ValueError(f"Expected 3 or 4 elements in batch, got {len(batch)}")

        # Encode noise reference
        if self.use_nc_loss and noise_label is not None:
            z, nc_logits = self.noise_encoder(noise_ref, return_nc_logits=True)
        else:
            z = self.noise_encoder(noise_ref, return_nc_logits=False)
            nc_logits = None

        # CFG dropout: randomly zero out noise embedding
        if self.training and self.p_uncond > 0:
            mask = torch.rand(z.shape[0], device=z.device) < self.p_uncond
            z = torch.where(mask[:, None], torch.zeros_like(z), z)

        # Sample time
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps

        # Get marginal distribution
        mean, std = self.sde.marginal_prob(x, y, t)

        # Sample x_t
        noise = torch.randn_like(x)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * noise

        # Forward pass
        score = self(x_t, y, t, z)

        # Score matching loss
        loss_score = self._score_loss(score, noise, t)

        # NC loss (multi-task learning)
        loss_nc = torch.tensor(0.0, device=x.device)
        if self.use_nc_loss and nc_logits is not None and noise_label is not None:
            loss_nc = F.cross_entropy(nc_logits, noise_label)

        # Total loss
        loss = loss_score + self.nc_loss_weight * loss_nc

        return loss, loss_score, loss_nc

    def training_step(self, batch, batch_idx):
        loss, loss_score, loss_nc = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_loss_score', loss_score, on_step=True, on_epoch=True, sync_dist=True)
        if self.use_nc_loss:
            self.log('train_loss_nc', loss_nc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate SE performance
        if batch_idx == 0 and self.num_eval_files != 0:
            self._evaluate_enhancement()

        loss, loss_score, loss_nc = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_loss_score', loss_score, on_step=False, on_epoch=True, sync_dist=True)
        if self.use_nc_loss:
            self.log('valid_loss_nc', loss_nc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def _evaluate_enhancement(self):
        """Evaluate speech enhancement on validation set."""
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except RuntimeError:
            rank = 0
            world_size = 1

        eval_files_per_gpu = self.num_eval_files // max(world_size, 1)
        clean_files = self.data_module.valid_set.clean_files[:self.num_eval_files]
        noisy_files = self.data_module.valid_set.noisy_files[:self.num_eval_files]

        if world_size > 1:
            if rank == world_size - 1:
                clean_files = clean_files[rank * eval_files_per_gpu:]
                noisy_files = noisy_files[rank * eval_files_per_gpu:]
            else:
                clean_files = clean_files[rank * eval_files_per_gpu:(rank + 1) * eval_files_per_gpu]
                noisy_files = noisy_files[rank * eval_files_per_gpu:(rank + 1) * eval_files_per_gpu]

        pesq_sum, si_sdr_sum, estoi_sum = 0, 0, 0

        for (clean_file, noisy_file) in zip(clean_files, noisy_files):
            x, sr_x = load(clean_file)
            x = x.squeeze().numpy()
            y, sr_y = load(noisy_file)

            if sr_x != 16000:
                x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
            else:
                x_16k = x

            x_hat = self.enhance(y)
            if self.sr != 16000:
                x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
            else:
                x_hat_16k = x_hat

            pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb')
            si_sdr_sum += si_sdr(x, x_hat)
            estoi_sum += stoi(x, x_hat, self.sr, extended=True)

        n = len(clean_files)
        self.log('pesq', pesq_sum / n, on_step=False, on_epoch=True, sync_dist=True)
        self.log('si_sdr', si_sdr_sum / n, on_step=False, on_epoch=True, sync_dist=True)
        self.log('estoi', estoi_sum / n, on_step=False, on_epoch=True, sync_dist=True)

    def forward(self, x_t, y, t, z=None):
        """
        Forward pass.

        Args:
            x_t: Corrupted signal [B, 1, F, T]
            y: Noisy observation [B, 1, F, T]
            t: Time step [B]
            z: Noise embedding [B, embed_dim]

        Returns:
            score: Estimated score [B, 1, F, T]
        """
        return self.dnn(x_t, y, t, z_r=z)

    def enhance(self, y, noise_ref=None, N=50, w=1.0, predictor="reverse_diffusion",
                corrector="ald", corrector_steps=1, snr=0.5, timeit=False, **kwargs):
        """
        Enhance noisy speech with CFG.

        Args:
            y: Noisy waveform [1, T] or [T]
            noise_ref: Noise reference waveform (optional)
            N: Number of diffusion steps
            w: CFG guidance scale (0=unconditional, 1=conditional, >1=amplified)
            predictor: Predictor type
            corrector: Corrector type
            corrector_steps: Number of corrector steps
            snr: SNR for corrector
            timeit: Return timing info

        Returns:
            x_hat: Enhanced waveform as numpy array
        """
        start = time.time()

        if y.dim() == 1:
            y = y.unsqueeze(0)

        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        # Convert to spectrogram
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        # Get noise embedding
        if noise_ref is not None:
            if noise_ref.dim() == 1:
                noise_ref = noise_ref.unsqueeze(0)
            noise_ref = noise_ref / norm_factor
            z = self.noise_encoder(noise_ref.cuda())
        else:
            z = torch.zeros(1, self.noise_embed_dim, device=Y.device)

        # CFG score function
        if w != 1.0 and self.p_uncond > 0:
            z_uncond = torch.zeros_like(z)

            def score_fn(x_t, y, t):
                s_cond = self(x_t, y, t, z)
                s_uncond = self(x_t, y, t, z_uncond)
                return s_uncond + w * (s_cond - s_uncond)
        else:
            def score_fn(x_t, y, t):
                return self(x_t, y, t, z)

        # Create sampler
        sde = self.sde.copy()
        sde.N = N
        sampler = sampling.get_pc_sampler(
            predictor, corrector, sde=sde,
            score_fn=score_fn, y=Y.cuda(), eps=self.t_eps,
            corrector_steps=corrector_steps, snr=snr,
            intermediate=False, **kwargs
        )

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()

        end = time.time()
        if timeit:
            rtf = (end - start) / (len(x_hat) / self.sr)
            return x_hat, nfe, rtf
        return x_hat

    # Data module methods
    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
