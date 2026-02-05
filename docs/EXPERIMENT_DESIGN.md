# Experiment Design: CFG for OOD-Robust Speech Enhancement

> 이 문서는 논문 contribution을 증명하기 위한 실험 설계를 정리합니다.

---

## 1. Core Hypothesis

**"Classifier-Free Guidance enables graceful degradation for OOD noise in noise-conditioned speech enhancement."**

### Predictions to Verify

1. **In-distribution**: NC+CFG ≈ NC (CFG doesn't hurt in-dist)
2. **OOD**: NC+CFG > NC (CFG prevents misleading conditioning)
3. **Guidance scale**: Lower w → better OOD, Higher w → better In-dist
4. **Conditioning effect**: Real embedding > Zero embedding (for in-dist)

---

## 2. Experiment Overview

| Exp ID | Purpose | Models | Key Metric |
|--------|---------|--------|------------|
| **E1** | Main comparison | SGMSE+, NASE, Ours | PESQ, SI-SDR on In-dist/OOD |
| **E2** | Guidance scale ablation | Ours (w=0,0.5,1,1.5) | OOD performance vs w |
| **E3** | CFG dropout ablation | Ours (p=0,0.1,0.2,0.3) | In-dist/OOD trade-off |
| **E4** | Conditioning effect | Real vs Zero embedding | Verify conditioning works |
| **E5** | Embedding analysis | t-SNE visualization | In-dist vs OOD separation |

---

## 3. Detailed Experiment Specifications

### E1: Main Comparison

**Purpose**: Show CFG improves OOD robustness over NASE

**Models**:
| Model | Config |
|-------|--------|
| SGMSE+ | Baseline (no conditioning) |
| NASE | BEATs + NC loss, p=0 |
| Ours | BEATs + NC loss, p=0.2 |

**Datasets**:
- **In-dist**: VoiceBank-DEMAND test (824 files)
- **OOD-1**: ESC-50 noise at 0dB SNR
- **OOD-2**: UrbanSound8K noise at 0dB SNR

**Expected Results**:
| Method | In-dist PESQ | OOD PESQ |
|--------|--------------|----------|
| SGMSE+ | ~2.9 | ~1.2 |
| NASE | ~3.0 | ~1.1 (misleading conditioning) |
| Ours (w=1) | ~3.0 | ~1.3 (graceful degradation) |
| Ours (w=0) | ~2.9 | ~1.2 (unconditional fallback) |

**Training Command**:
```bash
# SGMSE+ baseline
python train.py --base_dir ./data/voicebank-demand --backbone ncsnpp_v2 --gpus 4

# NASE (no CFG)
python train_nase.py --base_dir ./data/voicebank-demand \
    --backbone ncsnpp_nase --use_nc_loss --p_uncond 0.0 --gpus 4

# Ours (CFG p=0.2)
python train_nase.py --base_dir ./data/voicebank-demand \
    --backbone ncsnpp_nase --use_nc_loss --p_uncond 0.2 --gpus 4
```

---

### E2: Guidance Scale Ablation

**Purpose**: Show that guidance scale controls OOD robustness

**Setup**: Use Ours (p=0.2) checkpoint, vary w at inference

**Evaluation**:
```bash
# For each w in [0.0, 0.5, 1.0, 1.5]:
python enhancement_nase.py --ckpt <ours_ckpt> --w <w> --test_dir ./data/test_mixtures/vb_esc50/snr_0dB
```

**Expected Pattern**:
| w | In-dist PESQ | OOD PESQ | Interpretation |
|---|--------------|----------|----------------|
| 0.0 | ~2.9 | ~1.2 | Unconditional fallback |
| 0.5 | ~2.95 | ~1.25 | Balanced |
| 1.0 | ~3.0 | ~1.3 | Trust conditioning |
| 1.5 | ~3.0 | ~1.1 | Over-reliance on conditioning |

**Key Insight**: w=0.5 might be optimal for OOD, w=1.0 for In-dist

---

### E3: CFG Dropout Rate Ablation

**Purpose**: Find optimal dropout rate p

**Training**: Train separate models for each p

```bash
for p in 0.0 0.1 0.2 0.3; do
    python train_nase.py --base_dir ./data/voicebank-demand \
        --backbone ncsnpp_nase --use_nc_loss --p_uncond $p \
        --exp_name nase_p${p} --gpus 4
done
```

**Expected Pattern**:
| p | In-dist | OOD | Notes |
|---|---------|-----|-------|
| 0.0 | Best | Worst | No CFG, misleading conditioning |
| 0.1 | Good | Medium | Low dropout |
| 0.2 | Medium | Good | Balanced (NASE-style) |
| 0.3 | Lower | Best | High dropout, more unconditional |

---

### E4: Conditioning Effect Verification

**Purpose**: Verify that noise conditioning is actually being used

**Test**: Compare real embedding vs zero embedding

```python
# Real embedding (oracle noise reference)
x_hat_real = model.enhance(y, noise_ref=n, w=1.0)

# Zero embedding (unconditional)
x_hat_zero = model.enhance(y, noise_ref=None, w=0.0)  # or w=1.0 with zero ref
```

**Expected**:
- **In-dist**: real >> zero (conditioning helps)
- **OOD**: real ≈ zero or real < zero (conditioning can hurt)

**Critical Verification**:
If real ≈ zero for in-dist → conditioning is not working!

---

### E5: Embedding Analysis

**Purpose**: Visualize what noise encoder learns

**Method**:
1. Extract embeddings for all DEMAND noise types (training distribution)
2. Extract embeddings for ESC-50 noise types (OOD)
3. t-SNE visualization

**Expected**:
- DEMAND noises cluster together (in-distribution)
- ESC-50 noises are spread out or form separate clusters (OOD)
- Clear separation between in-dist and OOD regions

**Script**:
```python
# scripts/analyze_embeddings.py
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load model and extract embeddings
embeddings = []
labels = []
for noise_file in demand_files + esc50_files:
    z = model.noise_encoder(load_audio(noise_file))
    embeddings.append(z.cpu().numpy())
    labels.append(get_noise_type(noise_file))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30)
emb_2d = tsne.fit_transform(np.stack(embeddings))

# Plot
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels)
plt.savefig('embedding_tsne.png')
```

---

## 4. Training Schedule

### Phase 1: Baseline Verification (1일)
- [ ] SGMSE+ baseline 학습 또는 pretrained 사용
- [ ] Evaluation 코드 검증 (pretrained로 PESQ ~2.9 확인)

### Phase 2: Main Experiments (3-5일)
- [ ] NASE baseline (p=0) 학습
- [ ] Ours (p=0.2) 학습
- [ ] E1 main comparison 실행

### Phase 3: Ablations (2-3일)
- [ ] E2: Guidance scale ablation (inference only)
- [ ] E3: CFG dropout ablation (추가 학습 필요)
- [ ] E4: Conditioning effect verification

### Phase 4: Analysis (1-2일)
- [ ] E5: Embedding t-SNE
- [ ] Error analysis: 어떤 noise에서 CFG가 가장 도움되는지

---

## 5. Evaluation Commands

### Quick Eval (VB-DEMAND only)
```bash
python scripts/eval_nase.py --phase all --gpus 4,5,6,7 --datasets vb
```

### Full Eval (VB-DEMAND + OOD)
```bash
python scripts/eval_nase.py --phase all --gpus 4,5,6,7 --datasets all
```

### Single Model Eval
```bash
# Enhancement
python enhancement_nase.py \
    --ckpt ./logs/<exp_id>/last.ckpt \
    --test_dir ./data/voicebank-demand/test \
    --enhanced_dir ./enhanced_<exp_name> \
    --w 1.0 --N 50

# Metrics
python calc_metrics.py \
    --clean_dir ./data/voicebank-demand/test/clean \
    --noisy_dir ./data/voicebank-demand/test/noisy \
    --enhanced_dir ./enhanced_<exp_name>
```

---

## 6. Expected Paper Tables

### Table 1: Main Results
| Method | In-dist PESQ | In-dist SDR | OOD PESQ | OOD SDR |
|--------|--------------|-------------|----------|---------|
| Noisy | 1.97 | 8.4 | 1.18 | 0.0 |
| SGMSE+ | 2.93 | 17.3 | 1.20 | -0.5 |
| NASE | **2.98** | **17.8** | 1.15 | -1.0 |
| Ours (w=1.0) | 2.95 | 17.5 | **1.30** | **0.5** |

### Table 2: Guidance Scale
| w | In-dist PESQ | OOD PESQ |
|---|--------------|----------|
| 0.0 | 2.93 | 1.20 |
| 0.5 | 2.95 | 1.25 |
| 1.0 | 2.95 | 1.30 |
| 1.5 | 2.96 | 1.10 |

---

## 7. Success Criteria

논문 accept를 위한 최소 조건:

1. **SGMSE+ 재현**: In-dist PESQ > 2.8 ✓
2. **NASE 재현**: In-dist PESQ > SGMSE+ ✓
3. **CFG 효과**: OOD에서 NASE 대비 +0.1 PESQ 이상 ✓
4. **Graceful degradation**: w=0에서 OOD 성능 ≥ SGMSE+ ✓

---

*Created: 2026-02-05*
*Status: Ready for implementation*
