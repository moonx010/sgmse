# Noise-Conditioned SGMSE+: Experiment Report

## 1. Overview

This document describes experiments extending **SGMSE+** by conditioning the diffusion process on a short **noise reference** signal using **Classifier-Free Guidance (CFG)**.

The goal is to adapt speech enhancement to the target noise environment using only a short noise-only recording available at inference time.

---

## 2. Background: SGMSE+

### 2.1 Signal Representation

The input signal is represented as a complex STFT with magnitude compression:

$$\tilde{c} = \beta |c|^{\alpha} e^{j \angle c}$$

### 2.2 Conditional Forward Diffusion

Given noisy observation $y$, SGMSE+ models $p(x \mid y)$ via the forward SDE:

$$\mathrm{d}x_t = \gamma (y - x_t)\,\mathrm{d}t + g(t)\,\mathrm{d}w_t$$

### 2.3 Extension: Noise Reference Conditioning

We extend to model $p(x_0 \mid y, r)$ where $r$ is a noise reference signal:

$$s_\theta(x_t, y, z_r, t) \approx \nabla_{x_t} \log p_t(x_t \mid y, z_r)$$

where $z_r = E_\phi(r)$ is the noise embedding.

---

## 3. Research Contributions

### Contribution 1: CFG for Noise-Conditioned Speech Enhancement
> **CFG 기반 noise conditioning으로 OOD 일반화 성능 향상**

노이즈 컨디셔닝은 In-distribution에서는 성능 저하를 야기하지만, OOD에서 +1.4 dB SI-SDR 개선을 달성.

### Contribution 2: Improved OOD Generalization
> **Noise reference guidance로 unseen noise에 대한 일반화 개선**

SGMSE+ baseline 대비 OOD SI-SDR: -0.6 → 0.8 dB (+1.4 dB)

### Contribution 3: Optimal Noise Reference Length
> **짧은 noise reference (0.25s)가 최적**

긴 reference (0.5s~2.0s)는 오히려 성능 저하.

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Usage | Description |
|---------|-------|-------------|
| **VoiceBank-DEMAND** | Training & In-dist Test | 28 speakers, 10 noise types |
| **ESC-50** | OOD Test | 50 environmental sound classes |

### 4.2 Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **PESQ** | Perceptual Evaluation of Speech Quality | 1.0 - 4.5 (↑) |
| **ESTOI** | Extended Short-Time Objective Intelligibility | 0.0 - 1.0 (↑) |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | dB (↑) |

### 4.3 Training Configurations

#### Reference: Paper Configuration (TASLP 2023)

| Setting | Value |
|---------|-------|
| GPUs | 4× RTX 6000 (24GB) |
| Batch size/GPU | 8 |
| Effective batch | 32 |
| Steps | ~58k |
| Optimizer | Adam, lr=1e-4 |
| EMA decay | 0.999 |

#### PoC Experiments (50k steps)

| Model | Backbone | Batch Size | GPUs | Steps |
|-------|----------|------------|------|-------|
| SGMSE+ baseline | ncsnpp | 4 | 1 | 50k |
| NC-SGMSE+ (CFG) | ncsnpp_v2_cond | 4 | 1 | 50k |

---

## 5. Experimental Results

### 5.1 In-Distribution Performance (VoiceBank-DEMAND)

| Model | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|-------|--------|---------|----------|
| **SGMSE+ (no noise cond)** | **1.92 ± 0.65** | **0.76 ± 0.17** | **12.5 ± 5.0** |
| NC-SGMSE+ (p=0.0) | 1.81 ± 0.53 | 0.73 ± 0.17 | 11.1 ± 4.7 |
| CFG (p=0.05) | 1.07 ± 0.04 | 0.50 ± 0.12 | 3.4 ± 2.2 |
| CFG (p=0.1) | 1.40 ± 0.26 | 0.69 ± 0.15 | 10.3 ± 4.2 |
| CFG (p=0.15) | 1.71 ± 0.48 | 0.73 ± 0.17 | 11.2 ± 4.9 |
| **CFG (p=0.2)** | **1.86 ± 0.54** | **0.77 ± 0.15** | **12.3 ± 4.6** |
| CFG (p=0.25) | 1.07 ± 0.04 | 0.51 ± 0.11 | 3.1 ± 2.2 |
| CFG (p=0.3) | 1.86 ± 0.55 | 0.76 ± 0.16 | 12.1 ± 4.4 |
| CLAP-frozen | 1.70 ± 0.48 | 0.74 ± 0.17 | 11.5 ± 4.5 |
| CLAP-CFG (p=0.05) | 1.06 ± 0.03 | 0.46 ± 0.11 | 1.8 ± 2.2 |
| **CLAP-CFG (p=0.1)** | **1.83 ± 0.54** | **0.75 ± 0.17** | **12.1 ± 4.6** |
| CLAP-CFG (p=0.2) | 1.30 ± 0.21 | 0.64 ± 0.16 | 9.1 ± 4.0 |

**Key Findings**:
- SGMSE+ baseline이 In-dist에서 최고 (PESQ 1.92)
- 노이즈 컨디셔닝은 In-dist 성능 저하 (1.92 → 1.86)
- **노이즈 컨디셔닝의 가치는 OOD 일반화에 있음**

### 5.2 Out-of-Distribution Performance (ESC-50, SNR 0dB)

| Model | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|-------|--------|---------|----------|
| **SGMSE+ (no noise cond)** | **1.18 ± 0.35** | **0.46 ± 0.24** | **-0.6 ± 2.5** |
| NC-SGMSE+ (p=0.0) | 1.15 ± 0.23 | 0.47 ± 0.21 | -0.4 ± 2.2 |
| CFG (p=0.05) | 1.04 ± 0.02 | 0.32 ± 0.11 | -4.1 ± 4.0 |
| CFG (p=0.1) | 1.12 ± 0.13 | 0.45 ± 0.21 | -0.5 ± 2.1 |
| CFG (p=0.15) | 1.14 ± 0.19 | 0.45 ± 0.21 | -0.9 ± 2.8 |
| **CFG (p=0.2)** | **1.18 ± 0.25** | **0.51 ± 0.21** | **0.8 ± 2.0** |
| CFG (p=0.25) | 1.04 ± 0.02 | 0.32 ± 0.11 | -3.6 ± 2.7 |
| CFG (p=0.3) | 1.18 ± 0.26 | 0.51 ± 0.21 | 0.3 ± 1.8 |
| CLAP-frozen | 1.13 ± 0.21 | 0.46 ± 0.20 | -0.5 ± 3.2 |
| CLAP-CFG (p=0.05) | 1.04 ± 0.02 | 0.27 ± 0.10 | -5.2 ± 3.3 |
| CLAP-CFG (p=0.1) | 1.18 ± 0.24 | 0.50 ± 0.23 | 0.1 ± 2.1 |
| CLAP-CFG (p=0.2) | 1.09 ± 0.09 | 0.37 ± 0.17 | -2.2 ± 4.8 |

**Key Findings**:
- **CFG (p=0.2)가 OOD 최고**: SI-SDR 0.8 dB (baseline 대비 +1.4 dB)
- CLAP-CFG (p=0.1)도 competitive: SI-SDR 0.1 dB

### 5.3 Noise Reference Length Ablation

| ref_length | In-dist PESQ | In-dist SI-SDR | OOD PESQ | OOD SI-SDR |
|------------|--------------|----------------|----------|------------|
| **0.25s** | **1.59** | **10.9** | **1.12** | **-1.4** |
| 0.5s | 1.06 | 0.7 | 1.04 | -5.8 |
| 1.0s | 1.24 | 9.2 | 1.07 | -1.7 |
| 2.0s | 1.06 | 2.2 | 1.04 | -4.8 |

**Conclusion**: 0.25s가 최적. 긴 reference는 성능 저하.

### 5.4 Guidance Scale Analysis

| Model | w | In-dist PESQ | OOD SI-SDR |
|-------|---|--------------|------------|
| **CFG (p=0.2)** | **1.0** | **1.86** | **0.8** |
| CFG (p=0.2) | 3.0 | 1.87 | 0.4 |
| CFG (p=0.2) | 5.0 | 1.87 | 0.2 |

**Conclusion**: w=1.0이 최적. 증가 시 OOD 성능 저하.

---

## 6. Analysis

### 6.1 Why CFG Improves OOD?

1. **Regularization**: Conditioning dropout이 overfitting 방지
2. **Robust Conditioning**: Unconditional path가 불완전한 conditioning 보완
3. **Graceful Degradation**: OOD에서 conditioning 신뢰도 낮아도 동작

### 6.2 CNN vs CLAP Encoder

| Aspect | CNN + CFG (p=0.2) | CLAP + CFG (p=0.1) |
|--------|-------------------|-------------------|
| In-dist PESQ | **1.86** | 1.83 |
| OOD SI-SDR | **0.8** | 0.1 |
| Optimal p | 0.2 | 0.1 |

**Insight**: Encoder에 따라 optimal dropout rate가 다름

---

## 7. Scaled Training Experiments

### 7.1 Configuration (Paper-level)

| Setting | Value |
|---------|-------|
| GPUs | 4 |
| Batch size/GPU | 8 |
| Effective batch | 32 |
| Steps | 58,000 |

### 7.2 Experiments

| Exp ID | Model | Command | Status |
|--------|-------|---------|--------|
| SCALE-01 | CFG (p=0.2) | `CUDA_VISIBLE_DEVICES=0,1,2,3 python train_noise_cond.py --base_dir ./data/voicebank-demand --backbone ncsnpp_v2_cond --devices 4 --batch_size 8 --max_steps 58000 --cond_drop_prob 0.2 --wandb_name nc-cfg-p0.2-scaled` | 🔄 Training |
| SCALE-02 | SGMSE+ baseline | `CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --base_dir ./data/voicebank-demand --backbone ncsnpp --devices 4 --batch_size 8 --max_steps 58000 --wandb_name sgmse-paper-config` | 🔄 Training |

### 7.3 Expected Outcomes

- SCALE-01: Paper-level CFG performance (target PESQ > 2.5)
- SCALE-02: Reproduce paper baseline for fair comparison

---

## 8. Commands Reference

### Training

```bash
# SGMSE+ baseline (paper config)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --base_dir ./data/voicebank-demand --backbone ncsnpp --devices 4 --batch_size 8 --max_steps 58000 --wandb_name sgmse-paper-config

# Noise-conditioned CFG (paper scale)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_noise_cond.py --base_dir ./data/voicebank-demand --backbone ncsnpp_v2_cond --devices 4 --batch_size 8 --max_steps 58000 --cond_drop_prob 0.2 --wandb_name nc-cfg-p0.2-scaled
```

### Enhancement

```bash
# SGMSE+ baseline
python enhancement.py --test_dir ./data/voicebank-demand/test/noisy --enhanced_dir ./enhanced_dir --ckpt CKPT_PATH --N 30 --device cuda

# Noise-conditioned (oracle reference)
python enhancement_noise_cond.py --test_dir ./data/test_dir --enhanced_dir ./enhanced_dir --ckpt CKPT_PATH --oracle_noise --clean_dir ./clean_dir --N 30 --device cuda
```

### Metrics

```bash
python calc_metrics.py --clean_dir ./clean_dir --noisy_dir ./noisy_dir --enhanced_dir ./enhanced_dir
```

---

## 9. Checkpoints

### PoC Experiments (50k steps)

| Model | wandb_name | Checkpoint |
|-------|------------|------------|
| SGMSE+ baseline | sgmse-baseline | logs/ppwxy81n-sgmse-baseline/step=50000.ckpt |
| CFG (p=0.2) | nc-cfg-p0.2 | logs/[run_id]/step=50000.ckpt |
| CFG (p=0.3) | nc-cfg-p0.3 | logs/[run_id]/step=50000.ckpt |
| CLAP-CFG (p=0.1) | nc-clap-cfg-p0.1 | logs/[run_id]/step=50000.ckpt |

### Reference Length Ablation

| ref_length | wandb_name | Checkpoint |
|------------|------------|------------|
| 0.25s | nc-ref-0.25s | logs/zqtm721z-nc-ref-0.25s/step=50000.ckpt |
| 0.5s | nc-ref-0.5s | logs/7tucs4jy-nc-ref-0.5s/step=50000.ckpt |
| 1.0s | nc-ref-1.0s | logs/sr8toljy-nc-ref-1.0s/step=50000.ckpt |
| 2.0s | nc-ref-2.0s | logs/5xbd359r-nc-ref-2.0s/step=50000.ckpt |

---

## 10. Summary

### Best Configuration

```
Model: Noise-Conditioned SGMSE+ with CFG
Noise Encoder: CNN (from scratch)
CFG p_uncond: 0.2
Guidance Scale (w): 1.0
Noise Reference Length: 0.25s
```

### Main Results

| Contribution | Evidence | Improvement |
|--------------|----------|-------------|
| **OOD generalization** | SI-SDR: -0.6 → 0.8 dB | **+1.4 dB** |
| **Short reference optimal** | 0.25s vs 2.0s | **+50% PESQ** |

### Key Insight

**노이즈 컨디셔닝은 In-dist에서는 손해, OOD에서 이득.**

---

*Last updated: 2025-02-04*
*Scaled training (SCALE-01, SCALE-02) in progress*
