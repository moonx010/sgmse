# SGMSE Project Reference Guide

## Project Overview

**SGMSE (Score-based Generative Models for Speech Enhancement)** is a PyTorch implementation of diffusion-based generative models for speech enhancement and dereverberation. The project uses score-based generative models operating in the complex STFT domain.

### Key Papers
- [1] Interspeech 2022: Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain
- [2] IEEE/ACM TASLP 2023: Speech Enhancement and Dereverberation with Diffusion-Based Generative Models
- [3] Interspeech 2024: EARS dataset paper (48 kHz models)
- [4] ICASSP 2025: Investigating Training Objectives for Generative Speech Enhancement

## Core Architecture

### Diffusion Process
The model uses a forward diffusion process that gradually adds noise to clean speech spectrograms, and a reverse process that learns to iteratively generate clean speech from corrupted signals.

### Main Components
1. **SDE (Stochastic Differential Equations)**: Defines the diffusion process
2. **Backbone Networks**: Neural network architectures for score estimation
3. **Sampling**: Methods for generating enhanced speech (predictors + correctors)
4. **Data Module**: PyTorch Lightning data handling

## Project Structure

```
sgmse/
├── train.py                    # Main training script
├── enhancement.py              # Inference/enhancement script
├── calc_metrics.py             # Evaluation metrics computation
├── requirements.txt            # Dependencies
├── sgmse/
│   ├── model.py               # ScoreModel (main Lightning module)
│   ├── sdes.py                # SDE implementations (OUVE, VE, VP, SBVE, etc.)
│   ├── data_module.py         # SpecsDataModule for data loading
│   ├── backbones/
│   │   ├── ncsnpp.py          # NCSN++ backbone (paper [2] default)
│   │   ├── ncsnpp_v2.py       # NCSN++ v2 (paper [4])
│   │   ├── ncsnpp_48k.py      # 48 kHz model backbone (paper [3])
│   │   ├── dcunet.py          # DCUNet backbone (paper [1])
│   │   └── shared.py          # BackboneRegistry
│   ├── sampling/
│   │   ├── predictors.py      # Predictor methods (reverse diffusion)
│   │   └── correctors.py      # Corrector methods (Langevin dynamics)
│   └── util/
│       ├── inference.py       # Evaluation utilities
│       ├── other.py           # Helper functions
│       └── tensors.py         # Tensor operations
└── preprocessing/
    ├── create_wsj0_chime3.py  # WSJ0-CHiME3 dataset prep
    ├── create_wsj0_qut.py     # WSJ0-QUT dataset prep
    └── create_wsj0_reverb.py  # WSJ0-REVERB dataset prep
```

## Key Files and Their Roles

### Training Pipeline
- **train.py**: Main entry point for training
  - Uses PyTorch Lightning
  - Supports W&B logging (default) or CSVLogger
  - Dynamic argument parsing based on SDE and backbone choices

- **sgmse/model.py**: `ScoreModel` class
  - Inherits from `pl.LightningModule`
  - Handles training step, validation, optimizer setup
  - Implements various loss types: score_matching, data_prediction, etc.
  - Uses EMA (Exponential Moving Average) for model weights

- **sgmse/data_module.py**: `SpecsDataModule` class
  - Loads paired clean/noisy audio files
  - Converts to spectrograms (complex STFT)
  - Handles train/valid/test splits

### Inference Pipeline
- **enhancement.py**: Enhancement script
  - Loads checkpoint
  - Performs iterative sampling to denoise/dereverberate
  - Saves enhanced .wav files

- **calc_metrics.py**: Computes metrics
  - PESQ (Perceptual Evaluation of Speech Quality)
  - STOI (Short-Time Objective Intelligibility)
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

### Core Modules

#### SDE Classes (sgmse/sdes.py)
- Registry pattern: `SDERegistry`
- Available SDEs:
  - **OUVE** (default): Optimal Unconditional Variance Exploding
  - **VE**: Variance Exploding
  - **VP**: Variance Preserving
  - **SBVE**: Schrödinger Bridge Variance Exploding
  - **SubVP**: Sub-VP SDE

- Key methods:
  - `sde()`: Returns drift and diffusion coefficients
  - `marginal_prob()`: Computes marginal probability parameters
  - `prior_sampling()`: Samples from prior distribution

#### Backbone Networks (sgmse/backbones/)
- Registry pattern: `BackboneRegistry`
- Available backbones:
  - **ncsnpp**: NCSN++ (Noise Conditional Score Network++) - Paper [2]
  - **ncsnpp_v2**: NCSN++ v2 - Paper [4]
  - **ncsnpp_48k**: 48 kHz version - Paper [3]
  - **dcunet**: DCUNet - Paper [1]

#### Sampling Methods (sgmse/sampling/)
- **Predictors**: Reverse diffusion steps
  - ReverseDiffusionPredictor
  - EulerMaruyamaPredictor
  - NonePredictor

- **Correctors**: Refinement via Langevin dynamics
  - LangevinCorrector
  - NoneCorrector

## Training Configuration

### Default Settings (Paper [2])
```bash
python train.py --base_dir <your_base_dir>
```
- Backbone: `ncsnpp`
- SDE: `ouve`
- Learning rate: 1e-4
- EMA decay: 0.999

### Paper-Specific Configurations

**Paper [1] (Interspeech 2022)**
```bash
python train.py --base_dir <dir> --backbone dcunet --n_fft 512
```

**Paper [2] (TASLP 2023) - Default**
```bash
python train.py --base_dir <dir> --backbone ncsnpp
```

**Paper [3] (48 kHz Models)**
```bash
python train.py --base_dir <dir> \
    --backbone ncsnpp_48k \
    --n_fft 1534 \
    --hop_length 384 \
    --spec_factor 0.065 \
    --spec_abs_exponent 0.667 \
    --sigma-min 0.1 \
    --sigma-max 1.0 \
    --theta 2.0
```

**Paper [4] (Training Objectives)**
```bash
python train.py --base_dir <dir> --backbone ncsnpp_v2
```

**Schrödinger Bridge**
```bash
python train.py --base_dir <dir> \
    --backbone ncsnpp_v2 \
    --sde sbve \
    --loss_type data_prediction \
    --pesq_weight 5e-4
```

## Data Structure

Expected directory structure:
```
base_dir/
├── train/
│   ├── clean/
│   │   ├── file1.wav
│   │   └── file2.wav
│   └── noisy/
│       ├── file1.wav
│       └── file2.wav
├── valid/
│   ├── clean/
│   └── noisy/
└── test/ (optional)
    ├── clean/
    └── noisy/
```

- Only `.wav` files supported
- Clean and noisy subdirectories must have matching filenames

## Pretrained Checkpoints

### Speech Enhancement
- **VoiceBank-DEMAND**: `gdown 1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ`
- **WSJ0-CHiME3**: `gdown 16K4DUdpmLhDNC7pJhBBc08pkSIn_yMPi`
- **EARS-WHAM (48kHz)**: `gdown 1t_DLLk8iPH6nj8M5wGeOP3jFPaz3i7K5`

### Dereverberation
- **WSJ0-REVERB**: `gdown 1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD`
  - Use with: `--N 50 --snr 0.33`
- **EARS-Reverb (48kHz)**: `gdown 1PunXuLbuyGkknQCn_y-RCV2dTZBhyE3V`

### Training Objectives Models (M1-M8)
```bash
wget https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2025_gense/checkpoints/m{1..8}.ckpt
```

### Schrödinger Bridge
- Download from: `https://www2.informatik.uni-hamburg.de/sp/audio/publications/lipdiffuser/checkpoints/SB_VB-DMD_EARS-WHAM_900k.pt`

### ReverbFX Checkpoints
- SGMSE+ (artificial RIR)
- SGMSE+ (natural RIR)
- SB (artificial RIR)

## Evaluation Workflow

1. **Generate enhanced audio**:
```bash
python enhancement.py \
    --test_dir <test_dir> \
    --enhanced_dir <enhanced_dir> \
    --ckpt <checkpoint_path>
```

2. **Calculate metrics**:
```bash
python calc_metrics.py \
    --test_dir <test_dir> \
    --enhanced_dir <enhanced_dir>
```

## Key Hyperparameters

### Model
- `--lr`: Learning rate (default: 1e-4)
- `--ema_decay`: EMA decay constant (default: 0.999)
- `--t_eps`: Minimum process time (default: 0.03)
- `--loss_type`: Loss function type (score_matching, data_prediction)
- `--loss_weighting`: Loss weighting scheme (sigma^2, etc.)

### Spectrogram
- `--n_fft`: FFT size (default: 510 for 16kHz, 1534 for 48kHz)
- `--hop_length`: Hop length (default: 128 for 16kHz, 384 for 48kHz)
- `--spec_factor`: Spectrogram scaling factor (default: 0.15)
- `--spec_abs_exponent`: Magnitude compression exponent (default: 0.5)

### Training
- `--batch_size`: Batch size per GPU
- `--devices`: Number of GPUs (auto by default)
- `--num_eval_files`: Files for validation (20 by default)

### Sampling
- `--N`: Number of reverse diffusion steps (default: 30)
- `--corrector_steps`: Langevin corrector steps (default: 1)
- `--snr`: Signal-to-noise ratio for corrector (default: 0.5)

## Dependencies

Key packages:
- PyTorch + TorchAudio
- PyTorch Lightning
- WandB (optional logging)
- Librosa, SciPy
- PESQ, PySTOI (metrics)
- torch-ema, torchsde

Install: `pip install -r requirements.txt`

## Development Notes

### Code Organization
- Uses **registry pattern** for extensibility (BackboneRegistry, SDERegistry)
- PyTorch Lightning for training infrastructure
- Complex STFT domain processing (real + imaginary parts)

### Key Design Patterns
1. **Registry Pattern**: Easy addition of new backbones and SDEs
2. **Lightning Module**: Standardized training/validation/testing
3. **EMA**: Exponential moving average for stable inference
4. **Predictor-Corrector**: Two-stage sampling for quality

### Common Modifications
- Adding new backbone: Inherit from `BackboneRegistry` and register
- Adding new SDE: Inherit from `SDE` class and register with `SDERegistry`
- Changing loss: Modify `--loss_type` and implement in `model.py`

### Logging
- Default: WandB (requires account + login)
- Alternative: Pass `--nolog` for local CSV logging in `lightning_logs/`

## Related Projects

- **StoRM**: Stochastic Regeneration Model (follow-up)
- **SGMSE-BBED**: Prior mismatch reduction
- **Interactive Demo**: Jupyter notebook demo available

## Citation Info

When using this code, cite the relevant papers (see README.md for BibTeX entries).

---

## Research Documents

This project includes ongoing research on noise-conditioned speech enhancement. The following documents provide context for paper writing and experiments.

### Document Reference

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `docs/PERFORMANCE_DEBUG.md` | **[CRITICAL]** 성능 문제 분석 및 디버깅 | 성능 이슈 해결 시 최우선 |
| `docs/PAPER_WRITING_CONTEXT.md` | Paper writing guide and status | When working on paper (paper.tex) |
| `docs/EXPERIMENT_REPORT.md` | Detailed experimental results and analysis | When reviewing/updating results |
| `docs/NOISE_COND_IMPROVEMENTS.md` | Technical proposals and implementation details | When understanding methods |
| `docs/paper.tex` | Main Interspeech 2026 paper | When editing paper content |

### ⚠️ Current Issue: Performance Gap (2026-02-05)

**문제 상황**:
- Scaled training 결과가 기대에 미치지 못함
- CFG 모델이 baseline보다 오히려 성능 저하
- 논문 대비 baseline 성능도 낮음 (PESQ 2.9 vs 1.88)

**Scaled Training 결과 (N=30)**:
| Model | In-dist PESQ | OOD SI-SDR |
|-------|--------------|------------|
| SGMSE+ baseline | 1.88 | -0.2 |
| CFG p=0.2 | 1.75 | -0.6 |

**vs PoC 결과 (N=30)**:
| Model | In-dist PESQ | OOD SI-SDR |
|-------|--------------|------------|
| SGMSE+ baseline | 1.92 | -0.6 |
| CFG p=0.2 | 1.86 | **+0.8** |

**조사 중인 원인**:
1. Evaluation 설정 (N=30 vs N=50)
2. Noise encoder/conditioning 효과 부재
3. Scaled training에서의 CFG 동작 차이

**다음 단계**: `docs/PERFORMANCE_DEBUG.md` 참조

---

### Current Research: Noise-Conditioned SGMSE+

**Goal**: Improve OOD (Out-of-Distribution) noise generalization using CFG (Classifier-Free Guidance).

**Key Files**:
- Training: `train_noise_cond.py`
- Enhancement: `enhancement_noise_cond.py`
- Model: `sgmse/model_cond.py`
- Backbone: `sgmse/backbones/ncsnpp_v2_cond.py`

**Training Command**:
```bash
# PoC (1 GPU)
python train_noise_cond.py --base_dir ./data/voicebank-demand --backbone ncsnpp_v2_cond --cond_drop_prob 0.2 --devices 1 --batch_size 4 --max_steps 50000

# Scaled (4 GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_noise_cond.py --base_dir ./data/voicebank-demand --backbone ncsnpp_v2_cond --cond_drop_prob 0.2 --devices 4 --batch_size 8 --max_steps 58000
```

**Checkpoints**:
- SGMSE+ scaled: `./logs/55jxu1gw/last.ckpt`
- CFG p=0.2 scaled: `./logs/50y8p2e4/last.ckpt`
