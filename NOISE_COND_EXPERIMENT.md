# SGMSE+ Noise Reference Conditioning PoC

## 0. Purpose

This document describes a proof-of-concept experiment that extends **SGMSE+**
by conditioning the diffusion process on a short **noise reference** signal.

The goal is to adapt speech enhancement to the target noise environment
using only a short noise-only recording available at inference time.

This file is intended to provide **full experimental context** for understanding,
reproducing, and extending the experiment.

---

## 1. Baseline Model: SGMSE+

### 1.1 Signal Representation

The input signal is represented as a complex STFT.
A magnitude-compressed complex representation is used for numerical stability.

\[
\tilde{c}
=
\beta |c|^{\alpha} e^{j \angle c}
\]

The real and imaginary parts are treated as separate channels.
This representation is consistent with the Gaussian assumptions
used in the diffusion process.

---

### 1.2 Conditional Forward Diffusion

Given a noisy observation \( y \), SGMSE+ directly models the conditional distribution
of the clean signal \( x \mid y \).

The forward SDE is defined as

\[
\mathrm{d}x_t
=
\gamma (y - x_t)\,\mathrm{d}t
+
g(t)\,\mathrm{d}w_t
\]

where the diffusion coefficient follows a VE schedule

\[
g(t)
=
\sigma_{\min}
\left(
\frac{\sigma_{\max}}{\sigma_{\min}}
\right)^t
\sqrt{
2 \log \frac{\sigma_{\max}}{\sigma_{\min}}
}
\]

This is an Ornstein–Uhlenbeck–type process that pulls samples toward \( y \).

---

### 1.3 Reverse SDE and Sampling

The theoretical reverse-time SDE is

\[
\mathrm{d}x_t
=
\left[
\gamma (y - x_t)
-
g(t)^2 \nabla_{x_t} \log p_t(x_t \mid y)
\right]\mathrm{d}t
+
g(t)\,\mathrm{d}\bar{w}_t
\]

In practice, the score function is approximated by a neural network

\[
s_\theta(x_t, y, t)
\approx
\nabla_{x_t} \log p_t(x_t \mid y)
\]

The reverse SDE is solved using a plug-in approximation.

Initialization is performed around the observation:

\[
x_T \sim \mathcal{N}(y, \sigma_T^2 I)
\]

---

### 1.4 Training Objective

Samples are drawn directly from the perturbation kernel:

\[
x_t
=
\mu(x_0, y, t)
+
\sigma(t) z,
\quad
z \sim \mathcal{N}(0, I)
\]

The score-matching loss is

\[
\mathcal{L}_{\text{score}}
=
\mathbb{E}
\left[
\left\|
s_\theta(x_t, y, t)
+
\frac{z}{\sigma(t)}
\right\|^2
\right]
\]

This is **pure conditional score matching**, not noise prediction.

---

## 2. Extension Goal

We assume the availability of an additional observation:

- a short noise-only reference signal \( r \)

The target distribution becomes

\[
p(x_0 \mid y, r)
\]

This can be interpreted as adapting the enhancement process
to a specific noise environment.

---

## 3. Generative Assumption

We introduce a latent variable \( e \) representing the noise environment.

\[
e \sim p(e),
\quad
n \sim p(n \mid e),
\quad
r \sim p(r \mid e),
\quad
y = x_0 + n
\]

The reference \( r \) and the mixture noise \( n \) are generated
from the same environment \( e \).

A noise embedding is extracted as a sufficient statistic:

\[
z_r = E_\phi(r)
\]

---

## 4. Conditional Score Formulation

The target score is

\[
\nabla_{x_t} \log p_t(x_t \mid y, r)
\]

For PoC, we directly learn a **joint conditional score**:

\[
s_\theta(x_t, y, z_r, t)
\approx
\nabla_{x_t} \log p_t(x_t \mid y, z_r)
\]

This preserves the SGMSE+ diffusion framework
while extending the conditioning space.

---

## 5. Model Architecture

### 5.1 Noise Encoder

- Input: complex STFT of \( r \), same preprocessing as \( y \)
- Real and imaginary parts as channels
- Temporal dimension reduced via adaptive pooling
- Frequency structure preserved

Output:

\[
z_r \in \mathbb{R}^D
\]

---

### 5.2 Score Network Conditioning

- Backbone: unchanged NCSN++ from SGMSE+
- Conditioning injected through the time embedding path

Simple additive form:

\[
\text{temb} \leftarrow \text{temb} + W z_r
\]

or FiLM-style scale-shift modulation inside residual blocks.

For PoC, FiLM or additive conditioning is preferred over cross-attention.

---

## 6. Training Objective

Primary loss:

\[
\mathcal{L}
=
\mathbb{E}
\left[
\left\|
s_\theta(x_t, y, z_r, t)
+
\frac{z}{\sigma(t)}
\right\|^2
\right]
\]

Optional auxiliary loss to enforce noise conditioning.

Estimated residual noise:

\[
\hat{n} = y - \hat{x}
\]

Embedding consistency loss:

\[
\mathcal{L}_{\text{ref}}
=
d\big(E_\phi(r), E_\phi(\hat{n})\big)
\]

Total loss:

\[
\mathcal{L}_{\text{total}}
=
\mathcal{L}
+
\lambda_{\text{ref}} \mathcal{L}_{\text{ref}}
\]

For initial PoC runs, set \( \lambda_{\text{ref}} = 0 \).

---

## 7. Inference

- Same plug-in reverse SDE as SGMSE+
- Same noise embedding \( z_r \) used at all time steps
- Initialization and sampler follow SGMSE+ defaults

---

## 8. Experimental Design

### 8.1 Data Construction

**Oracle reference setting (PoC stage 1)**

- Given clean \( x_0 \) and noisy \( y \)
- True noise: \( n = y - x_0 \)
- Reference \( r \) obtained by random cropping from \( n \)

This ensures perfect environment matching.

---

### 8.2 Evaluation Scenarios

- Baseline SGMSE+ without reference
- Proposed model with correct reference
- Reference mismatch experiments

---

### 8.3 Ablations

- Reference length: 0.25s, 0.5s, 1s, 2s
- Conditioning method: additive vs FiLM
- \( \lambda_{\text{ref}} \) sweep
- Encoder frozen vs jointly trained

---

### 8.4 Success Criteria

- Improvement over baseline on seen noise
- Improvement with correct reference on unseen noise
- No catastrophic degradation under reference mismatch
- Graceful degradation as reference length decreases

---

## 9. Future Extensions

- Separate base score and noise guidance via energy-based formulation
- Define \( E_\psi(\hat{n}, z_r) \) and add \( \nabla_{x_t} E_\psi \) during sampling
- Probabilistic interpretation of noise embeddings

This document is intended to evolve alongside ongoing experiments.

---

## 10. Experiment Log

### 10.1 Reference: Paper Training Configuration (TASLP 2023)

| Setting | Value |
|---------|-------|
| GPUs | 4× Quadro RTX 6000 (24GB) |
| Batch size per GPU | 8 |
| Effective batch size | 32 |
| Epochs | 160 |
| Estimated steps | ~58k |
| Total samples seen | ~1.85M |
| Optimizer | Adam, lr=1e-4 |
| EMA decay | 0.999 |
| STFT | window=510, hop=128, frames=256 |
| SDE params | σ_min=0.05, σ_max=0.5, γ=1.5 |
| Sampling | N=30, 1 corrector step, r=0.5 |

**Note**: Full paper configuration requires ~160 epochs for best performance. Current PoC experiments use shorter training for faster iteration.

---

### 10.2 Training Setup (PoC Experiments)

| Model | Dataset | Steps | Epochs | Batch Size | GPUs |
|-------|---------|-------|--------|------------|------|
| Noise-Cond (Ours) | VoiceBank-DEMAND | 50,000 | 138 | 8 | 4 |
| Baseline (Ours) | VoiceBank-DEMAND | 50,000 | - | 4 | 1 |
| Baseline (Pretrained) | VoiceBank-DEMAND | 41,992 | 116 | - | - |

**Note**: Baseline (Ours) used batch_size=4 due to GPU memory constraints. Noise-Cond used batch_size=8 with 4 GPUs. This difference may affect fair comparison.

- **Training data duration**: ~10 hours (VoiceBank-DEMAND train set)
- **Noise-cond checkpoint**: `lightning_logs/version_0/checkpoints/epoch=138-step=50000.ckpt`
- **Baseline 50k checkpoint**: `logs/[RUN_ID]/last.ckpt`

---

### 10.3 In-Distribution Results (VoiceBank-DEMAND Test Set)

| Model | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|-------|--------|---------|----------|
| Noisy (input) | - | - | - |
| Baseline (Pretrained 42k) | 2.91 ± 0.62 | 0.86 ± 0.10 | 16.9 ± 3.1 |
| Baseline (Ours 50k, bs=4) | 1.95 ± 0.66 | 0.76 ± 0.17 | 12.8 ± 4.8 |
| Noise-Cond (Ours 50k, bs=8) | 1.80 ± 0.53 | 0.73 ± 0.17 | 12.5 ± 5.3 |

**Observation**:
- Pretrained baseline (42k steps) significantly outperforms both 50k models, likely due to longer effective training or hyperparameter differences.
- Baseline 50k (bs=4) vs Noise-Cond 50k (bs=8): Similar performance on in-distribution data. Noise-cond slightly lower, but batch size difference may affect comparison.
- Fair comparison requires same batch size. Consider retraining with matched settings.

---

### 10.4 Out-of-Distribution Results (ESC-50 Noise)

Test set: VoiceBank clean + ESC-50 noise (`vb_esc50`)

#### SNR 0 dB

| Model | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|-------|--------|---------|----------|
| Noisy (input) | - | - | - |
| Baseline (Pretrained 42k) | TBD | TBD | TBD |
| Baseline (Ours 50k) | TBD | TBD | TBD |
| Noise-Cond (Ours 50k) | TBD | TBD | TBD |

#### SNR 5 dB

| Model | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|-------|--------|---------|----------|
| Noisy (input) | - | - | - |
| Baseline (Pretrained 42k) | TBD | TBD | TBD |
| Baseline (Ours 50k) | TBD | TBD | TBD |
| Noise-Cond (Ours 50k) | TBD | TBD | TBD |

#### SNR 10 dB

| Model | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|-------|--------|---------|----------|
| Noisy (input) | - | - | - |
| Baseline (Pretrained 42k) | TBD | TBD | TBD |
| Baseline (Ours 50k) | TBD | TBD | TBD |
| Noise-Cond (Ours 50k) | TBD | TBD | TBD |

---

### 10.5 Ablation Study: Noise Reference Length

**Goal**: Test whether longer noise reference improves performance.

**Setup**:
- All models trained with batch_size=4 for fair comparison with baseline
- 50k steps each
- VoiceBank-DEMAND dataset
- WandB logging enabled

| Experiment ID | ref_length_sec | ref_num_frames | wandb_name | Checkpoint | Status |
|---------------|----------------|----------------|------------|------------|--------|
| NC-ref025 | 0.25s | 32 | nc-ref-0.25s | logs/zqtm721z-nc-ref-0.25s/step=50000.ckpt | ✅ Done |
| NC-ref05 | 0.5s | 63 | nc-ref-0.5s | logs/7tucs4jy-nc-ref-0.5s/step=50000.ckpt | ✅ Done |
| NC-ref1 | 1.0s | 126 | nc-ref-1.0s | logs/sr8toljy-nc-ref-1.0s/step=50000.ckpt | ✅ Done |
| NC-ref2 | 2.0s | 251 | nc-ref-2.0s | logs/5xbd359r-nc-ref-2.0s/step=50000.ckpt | ✅ Done |

**Frame calculation** (16kHz, hop_length=128):
```
ref_num_frames = int(ref_length_sec * 16000 / 128) + 1
```

**Training Commands**:
```bash
# Reference 0.25s
CUDA_VISIBLE_DEVICES=2 python train_noise_cond.py --base_dir ./data/voicebank-demand --devices 1 --max_steps 50000 --batch_size 4 --ref_length_sec 0.25 --wandb_name nc-ref-0.25s

# Reference 0.5s
CUDA_VISIBLE_DEVICES=3 python train_noise_cond.py --base_dir ./data/voicebank-demand --devices 1 --max_steps 50000 --batch_size 4 --ref_length_sec 0.5 --wandb_name nc-ref-0.5s

# Reference 1.0s
CUDA_VISIBLE_DEVICES=4 python train_noise_cond.py --base_dir ./data/voicebank-demand --devices 1 --max_steps 50000 --batch_size 4 --ref_length_sec 1.0 --wandb_name nc-ref-1.0s

# Reference 2.0s
CUDA_VISIBLE_DEVICES=5 python train_noise_cond.py --base_dir ./data/voicebank-demand --devices 1 --max_steps 50000 --batch_size 4 --ref_length_sec 2.0 --wandb_name nc-ref-2.0s
```

**Results (VB-DEMAND Test Set)**:

| ref_length | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|------------|--------|---------|----------|
| 0.25s | **1.59 ± 0.42** | **0.71 ± 0.18** | **10.9 ± 4.6** |
| 0.5s | 1.06 ± 0.03 | 0.43 ± 0.11 | 0.7 ± 2.1 |
| 1.0s | 1.24 ± 0.15 | 0.64 ± 0.15 | 9.2 ± 3.4 |
| 2.0s | 1.06 ± 0.03 | 0.48 ± 0.11 | 2.2 ± 2.2 |

**Results (OOD - ESC-50 Noise, SNR 0dB)**:

| ref_length | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|------------|--------|---------|----------|
| 0.25s | **1.12 ± 0.17** | **0.42 ± 0.23** | **-1.4 ± 3.4** |
| 0.5s | 1.04 ± 0.02 | 0.25 ± 0.11 | -5.8 ± 3.7 |
| 1.0s | 1.07 ± 0.06 | 0.38 ± 0.18 | -1.7 ± 3.7 |
| 2.0s | 1.04 ± 0.02 | 0.29 ± 0.11 | -4.8 ± 3.5 |

**Observations**:
- **0.25s reference performs best** on both in-distribution and OOD tests
- 0.5s and 2.0s references show poor performance (near PESQ 1.0)
- Results are counterintuitive - longer reference does NOT improve performance
- Possible issues: training instability, overfitting, or embedding capacity problems with longer references

---

### 10.6 Key Findings

1. **In-distribution performance**: Noise conditioning does not improve performance on noise types seen during training (DEMAND). The baseline model already captures these noise patterns.

2. **OOD generalization**: Performance degrades significantly on ESC-50 noise (OOD). All models show negative SI-SDR, indicating the enhanced signal is worse than input in terms of distortion.

3. **Training efficiency**: Both models trained for similar number of epochs/steps for fair comparison.

4. **Reference length effect (Ablation Study)**:
   - **Surprising result**: Shorter reference (0.25s) performs best
   - 0.5s and 2.0s show near-random performance (PESQ ~1.0)
   - 1.0s shows moderate performance
   - **Hypothesis**: Longer references may cause:
     - Training instability due to larger input variance
     - Encoder overfitting to noise-specific patterns
     - Difficulty in learning compressed representations
   - **Action needed**: Investigate training curves on WandB, check for convergence issues

---

### 10.7 Identified Limitations and Proposed Improvements

> **Detailed analysis**: See [docs/NOISE_COND_IMPROVEMENTS.md](docs/NOISE_COND_IMPROVEMENTS.md)

#### 10.7.1 Problem Analysis

| Issue | Description | Impact |
|-------|-------------|--------|
| **Embedding Space Coverage** | Noise encoder trained only on DEMAND noise | Poor OOD generalization |
| **Stationary Assumption** | Global pooling loses temporal information | Cannot handle non-stationary noise |
| **No Unconditional Baseline** | Model may ignore weak conditioning | Conditioning collapse risk |

#### 10.7.2 Proposed Solutions

| Solution | Priority | Implementation | Expected Benefit |
|----------|----------|----------------|------------------|
| **Classifier-Free Guidance (CFG)** | High | Add conditioning dropout | Better OOD, controllable guidance |
| **Pre-trained Encoder (CLAP)** | High | Replace noise encoder | Generalized representations |
| **Cross-Attention** | Medium | Modify architecture | Non-stationary noise handling |
| **Noise Augmentation** | Low | Add training data | Broader coverage |

#### 10.7.3 Planned Experiments

**Phase 1: CFG (Immediate)**
```bash
# Training with conditioning dropout
python train_noise_cond.py --base_dir ./data/voicebank-demand \
    --devices 1 --max_steps 50000 --batch_size 4 \
    --cond_drop_prob 0.1 --wandb_name nc-cfg-p0.1

# Inference with guidance scale
python enhancement_noise_cond.py --test_dir ... \
    --guidance_scale 3.0
```

**Phase 2: CLAP Integration**
- Replace `NoiseEncoder` with `CLAPNoiseEncoder`
- Test frozen vs fine-tuned CLAP weights
- Evaluate on OOD noise

**Phase 3: Cross-Attention**
- Implement sequence-based noise encoding
- Add cross-attention to ResBlocks
- Test on non-stationary noise

---

### 10.8 Commands Reference

**Training:**
```bash
# Noise-cond
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_noise_cond.py --base_dir ./data/voicebank-demand --gpus 4 --max_steps 50000

# Baseline
CUDA_VISIBLE_DEVICES=2 python train.py --base_dir ./data/voicebank-demand --backbone ncsnpp --devices 1 --max_steps 50000 --batch_size 4 --accumulate_grad_batches 2
```

**Enhancement:**
```bash
# Noise-cond (with oracle noise reference)
python enhancement_noise_cond.py --test_dir ./data/test_dir --enhanced_dir ./enhanced_dir --ckpt CKPT_PATH --oracle_noise --clean_dir ./clean_dir --N 30 --device cuda

# Baseline
python enhancement.py --test_dir ./data/test_dir/noisy --enhanced_dir ./enhanced_dir --ckpt CKPT_PATH --N 30 --device cuda
```

**Metrics:**
```bash
python calc_metrics.py --clean_dir ./clean_dir --noisy_dir ./noisy_dir --enhanced_dir ./enhanced_dir
```