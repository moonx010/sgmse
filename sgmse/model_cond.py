"""
Noise-conditioned Score Model for SGMSE+.

Extends the base ScoreModel to incorporate noise reference conditioning.
The model learns p(x | y, z_r) where z_r is an embedding of the noise reference.
"""

import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.noise_encoder import NoiseEncoder, NoiseEncoderLight
try:
    from sgmse.clap_encoder import CLAPNoiseEncoder, CLAPNoiseEncoderSimple
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss


class NoiseCondScoreModel(pl.LightningModule):
    """
    Score-based model with noise reference conditioning.

    This model extends SGMSE+ by learning the conditional score
    s_θ(x_t, y, z_r, t) ≈ ∇_{x_t} log p_t(x_t | y, z_r)

    where z_r = E_φ(r) is the noise embedding from encoder E_φ.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999,
                            help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03,
                            help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20,
                            help="Number of files for evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="score_matching",
                            help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2",
                            help="The weighting of the loss function.")
        parser.add_argument("--sr", type=int, default=16000,
                            help="The sample rate of the audio files.")
        # Noise encoder arguments
        parser.add_argument("--noise_encoder_type", type=str, default="default",
                            choices=["default", "light", "clap", "clap_simple"],
                            help="Type of noise encoder to use.")
        parser.add_argument("--noise_embed_dim", type=int, default=512,
                            help="Dimension of noise embedding.")
        parser.add_argument("--noise_encoder_nf", type=int, default=64,
                            help="Base number of filters in noise encoder.")
        parser.add_argument("--train_noise_encoder", type=bool, default=True,
                            help="Whether to train the noise encoder jointly.")
        parser.add_argument("--freeze_clap", action='store_true',
                            help="Freeze CLAP weights (only train projection layer)")
        # Classifier-free guidance
        parser.add_argument("--cond_drop_prob", type=float, default=0.0,
                            help="Probability of dropping noise conditioning during training (for CFG).")
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
        noise_encoder_type='default',
        noise_embed_dim=512,
        noise_encoder_nf=64,
        train_noise_encoder=True,
        cond_drop_prob=0.0,
        freeze_clap=True,
        data_module_cls=None,
        **kwargs
    ):
        super().__init__()

        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(noise_embed_dim=noise_embed_dim, **kwargs)

        # Initialize Noise Encoder
        self.noise_encoder_type = noise_encoder_type
        self.use_clap = noise_encoder_type in ['clap', 'clap_simple']

        if noise_encoder_type == 'default':
            self.noise_encoder = NoiseEncoder(
                in_channels=2,
                nf=noise_encoder_nf,
                embed_dim=noise_embed_dim,
            )
        elif noise_encoder_type == 'light':
            self.noise_encoder = NoiseEncoderLight(
                embed_dim=noise_embed_dim,
            )
        elif noise_encoder_type == 'clap':
            if not CLAP_AVAILABLE:
                raise ImportError("CLAP encoder requires laion-clap. Install with: pip install laion-clap")
            self.noise_encoder = CLAPNoiseEncoder(
                embed_dim=noise_embed_dim,
                freeze_clap=freeze_clap,
            )
        elif noise_encoder_type == 'clap_simple':
            if not CLAP_AVAILABLE:
                raise ImportError("CLAP encoder requires laion-clap. Install with: pip install laion-clap")
            self.noise_encoder = CLAPNoiseEncoderSimple(
                embed_dim=noise_embed_dim,
                freeze_clap=freeze_clap,
            )

        self.train_noise_encoder = train_noise_encoder
        if not train_noise_encoder:
            for param in self.noise_encoder.parameters():
                param.requires_grad = False

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)

        # Store hyperparams
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.num_eval_files = num_eval_files
        self.sr = sr
        self.noise_embed_dim = noise_embed_dim
        self.cond_drop_prob = cond_drop_prob

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

    def _loss(self, score, z, t):
        """
        Compute score matching loss.

        Args:
            score: Predicted score s_θ(x_t, y, z_r, t)
            z: Sampled noise
            t: Time steps

        Returns:
            loss: Scalar loss value
        """
        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            if self.loss_weighting == "sigma^2":
                # DSM loss: ||σ * s_θ + z||^2
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
            batch: (x, y, r) or (x, y, r_spec, r_wav) tuple
                   - Standard: clean, noisy, and reference spectrograms
                   - CLAP: includes raw waveform for noise reference
            batch_idx: Batch index
        """
        if len(batch) == 4:
            # CLAP mode: (x, y, r_spec, r_wav)
            x, y, r_spec, r_wav = batch
            r = r_wav if self.use_clap else r_spec
        else:
            # Standard mode: (x, y, r)
            x, y, r = batch

        # Encode noise reference
        z_r = self.noise_encoder(r)  # [B, embed_dim]

        # Classifier-free guidance: randomly drop conditioning
        if self.training and self.cond_drop_prob > 0:
            mask = torch.rand(z_r.shape[0], device=z_r.device) < self.cond_drop_prob
            z_r = torch.where(mask[:, None], torch.zeros_like(z_r), z_r)

        # Sample time
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps

        # Get marginal distribution parameters
        mean, std = self.sde.marginal_prob(x, y, t)

        # Sample x_t
        z = torch.randn_like(x)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z

        # Forward pass
        score = self(x_t, y, t, z_r)

        # Compute loss
        loss = self._loss(score, z, t)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
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

            pesq_sum = 0
            si_sdr_sum = 0
            estoi_sum = 0

            for (clean_file, noisy_file) in zip(clean_files, noisy_files):
                x, sr_x = load(clean_file)
                x = x.squeeze().numpy()
                y, sr_y = load(noisy_file)
                assert sr_x == sr_y, "Sample rates do not match!"

                if sr_x != 16000:
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
                else:
                    x_16k = x

                # For validation, use the same noisy signal as reference
                # (oracle noise from y - x estimated after enhancement)
                x_hat = self.enhance(y)
                if self.sr != 16000:
                    x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
                else:
                    x_hat_16k = x_hat

                pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb')
                si_sdr_sum += si_sdr(x, x_hat)
                estoi_sum += stoi(x, x_hat, self.sr, extended=True)

            pesq_avg = pesq_sum / len(clean_files)
            si_sdr_avg = si_sdr_sum / len(clean_files)
            estoi_avg = estoi_sum / len(clean_files)

            self.log('pesq', pesq_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi', estoi_avg, on_step=False, on_epoch=True, sync_dist=True)

        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def forward(self, x_t, y, t, z_r=None):
        """
        Forward pass of the noise-conditioned score network.

        Args:
            x_t: Corrupted signal [B, 1, F, T] (complex)
            y: Noisy observation [B, 1, F, T] (complex)
            t: Time step [B]
            z_r: Noise embedding [B, embed_dim]

        Returns:
            score: Estimated score [B, 1, F, T] (complex)
        """
        score = self.dnn(x_t, y, t, z_r=z_r)
        return score

    def get_pc_sampler(self, predictor_name, corrector_name, y, z_r, N=None,
                       minibatch=None, **kwargs):
        """Get predictor-corrector sampler with noise conditioning."""
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}

        # Create a wrapper score function that includes z_r
        def score_fn(x_t, y, t):
            return self(x_t, y, t, z_r)

        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name, corrector_name, sde=sde,
                score_fn=score_fn, y=y, **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch:(i + 1) * minibatch]
                    z_r_mini = z_r[i * minibatch:(i + 1) * minibatch] if z_r is not None else None

                    def score_fn_mini(x_t, y, t):
                        return self(x_t, y, t, z_r_mini)

                    sampler = sampling.get_pc_sampler(
                        predictor_name, corrector_name, sde=sde,
                        score_fn=score_fn_mini, y=y_mini, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

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

    def enhance(self, y, noise_ref=None, sampler_type="pc", predictor="reverse_diffusion",
                corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
                cfg_scale=1.0, **kwargs):
        """
        One-call speech enhancement with noise conditioning.

        Args:
            y: Noisy waveform tensor [1, T] or [T]
            noise_ref: Noise reference waveform [1, T_ref] or None
                       If None, extracts reference from y (assumes silence detection)
            sampler_type: "pc" for predictor-corrector
            predictor: Predictor method name
            corrector: Corrector method name
            N: Number of reverse diffusion steps
            corrector_steps: Number of corrector steps
            snr: SNR for corrector
            timeit: Whether to return timing info
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            x_hat: Enhanced waveform as numpy array
        """
        start = time.time()

        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(0)

        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        # Convert noisy signal to spectrogram
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        # Handle noise reference
        if noise_ref is not None:
            if noise_ref.dim() == 1:
                noise_ref = noise_ref.unsqueeze(0)
            noise_ref = noise_ref / norm_factor

            if self.use_clap:
                # CLAP encoder expects raw waveform
                z_r = self.noise_encoder(noise_ref.cuda())
            else:
                # Standard encoder expects spectrogram
                R = torch.unsqueeze(self._forward_transform(self._stft(noise_ref.cuda())), 0)
                z_r = self.noise_encoder(R)
        else:
            # Use zero embedding if no reference provided
            z_r = torch.zeros(1, self.noise_embed_dim, device=Y.device)

        # Classifier-free guidance
        if cfg_scale != 1.0 and self.cond_drop_prob > 0:
            z_r_uncond = torch.zeros_like(z_r)

            def score_fn(x_t, y, t):
                # Conditional score
                score_cond = self(x_t, y, t, z_r)
                # Unconditional score
                score_uncond = self(x_t, y, t, z_r_uncond)
                # CFG interpolation
                return score_uncond + cfg_scale * (score_cond - score_uncond)
        else:
            def score_fn(x_t, y, t):
                return self(x_t, y, t, z_r)

        # Create sampler with custom score function
        if self.sde.__class__.__name__ == 'OUVESDE':
            sde = self.sde.copy()
            sde.N = N
            sampler = sampling.get_pc_sampler(
                predictor, corrector, sde=sde,
                score_fn=score_fn, y=Y.cuda(), eps=self.t_eps,
                corrector_steps=corrector_steps, snr=snr,
                intermediate=False, **kwargs
            )
        else:
            raise ValueError(f"Invalid SDE type: {self.sde.__class__.__name__}")

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()

        end = time.time()
        if timeit:
            rtf = (end - start) / (len(x_hat) / self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
