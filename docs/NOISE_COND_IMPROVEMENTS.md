# Noise Conditioning Improvements: Analysis and Proposals

## 1. Problem Analysis

### 1.1 Current Limitations

#### Embedding Space Coverage Problem

| Aspect | Time Embedding (t) | Noise Embedding (z_r) |
|--------|-------------------|----------------------|
| Range | [0, 1] bounded | ℝ^512 unbounded |
| Training Coverage | Uniform sampling, full coverage | Only DEMAND noise distribution |
| Generalization | Guaranteed for all t values | Unknown for OOD noise |

**Core Issue**: The score network learns `s_θ(x_t, y, z_r, t)` where `t` is uniformly sampled from [0,1], but `z_r` is only exposed to the embedding distribution of DEMAND noise. When new noise types map to embeddings outside this distribution, the score function may not work correctly.

#### Stationary Noise Assumption

Current Noise Encoder architecture:
```
Input [B, 2, F, T]
  → Conv×4 (stride=2)
  → AdaptiveAvgPool(time axis compression)
  → FC×2
  → z_r [B, 512]
```

**Problems**:
- Temporal average pooling → loss of non-stationary information
- Examples: sudden impact sounds, changing background noise → characteristics lost through averaging
- Cannot capture time-varying noise patterns

### 1.2 Theoretical Concerns

1. **Score Function Validity**: For diffusion models, the score function must be well-defined across the conditioning space. With limited noise types during training, the model may learn a score function that's only valid in a narrow region of the embedding space.

2. **Manifold Hypothesis**: If DEMAND noises lie on a low-dimensional manifold in embedding space, the model only learns to denoise along this manifold. OOD noises may lie off-manifold, leading to poor performance.

3. **Conditioning Collapse**: Without proper regularization, the model might learn to ignore the noise conditioning if it's not strictly necessary for DEMAND noise removal.

---

## 2. Proposed Solutions

### 2.1 Solution A: Classifier-Free Guidance (CFG)

**Concept**: Train the model to work both with and without noise conditioning, then combine them at inference.

**Training**:
```python
# With probability p_uncond, drop the noise conditioning
if random() < p_uncond:
    z_r = torch.zeros_like(z_r)  # null condition

score = model(x_t, y, z_r, t)
loss = score_matching_loss(score, target)
```

**Inference**:
```python
score_cond = model(x_t, y, z_r, t)      # conditional score
score_uncond = model(x_t, y, 0, t)       # unconditional score
score = score_uncond + w * (score_cond - score_uncond)  # guided score
```

**Benefits**:
- Model learns both conditional and unconditional denoising
- Guidance scale `w` controls conditioning strength
- Graceful degradation when conditioning is weak/missing
- Minimal code changes required

**Hyperparameters**:
- `p_uncond`: 0.1 - 0.2 (dropout probability during training)
- `w`: 1.0 - 7.0 (guidance scale at inference, higher = stronger conditioning)

**References**:
- [Ho & Salimans 2022] Classifier-Free Diffusion Guidance
- [Rethinking CFG 2024] Independent Condition Guidance

---

### 2.2 Solution B: Pre-trained Audio Encoder (CLAP/PANNs)

**Concept**: Replace the from-scratch noise encoder with a pre-trained audio encoder that has seen diverse audio.

**Architecture**:
```python
class CLAPNoiseEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze_clap=True):
        super().__init__()
        self.clap = CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()

        # Optional: freeze CLAP weights
        if freeze_clap:
            for p in self.clap.parameters():
                p.requires_grad = False

        # Projection to match score network dimension
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, audio_waveform):
        # audio_waveform: [B, T] at 48kHz (CLAP default)
        with torch.no_grad() if self.freeze_clap else nullcontext():
            emb = self.clap.get_audio_embedding_from_data(audio_waveform)
        return self.proj(emb)
```

**Benefits**:
- Pre-trained on AudioSet (5000+ hours, 527 classes)
- Generalized audio representations
- Zero-shot capability built-in
- Faster convergence

**Considerations**:
- CLAP expects 48kHz audio (need resampling from 16kHz)
- CLAP embedding is 512-dim (matches our current design)
- Can freeze CLAP and only train projection layer

**References**:
- [LAION CLAP] Contrastive Language-Audio Pretraining
- [AudioLDM] Text-to-Audio with CLAP conditioning
- [CLAPSep] Sound extraction with CLAP embeddings

---

### 2.3 Solution C: Cross-Attention for Non-Stationary Noise

**Concept**: Instead of global pooling, use cross-attention to allow the score network to selectively attend to different temporal parts of the noise reference.

**Modified Noise Encoder**:
```python
class SequenceNoiseEncoder(nn.Module):
    def __init__(self, output_dim=512, num_tokens=16):
        super().__init__()
        # CNN backbone (no global pooling)
        self.conv_layers = nn.Sequential(
            ConvBlock(2, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
        )
        # Reduce to fixed number of tokens
        self.pool = nn.AdaptiveAvgPool2d((4, num_tokens))  # [B, 512, 4, num_tokens]
        self.proj = nn.Linear(512 * 4, output_dim)

    def forward(self, r):
        # r: [B, 2, F, T]
        h = self.conv_layers(r)      # [B, 512, F', T']
        h = self.pool(h)              # [B, 512, 4, num_tokens]
        h = h.permute(0, 3, 1, 2)     # [B, num_tokens, 512, 4]
        h = h.flatten(2)              # [B, num_tokens, 512*4]
        z_seq = self.proj(h)          # [B, num_tokens, output_dim]
        return z_seq
```

**Score Network with Cross-Attention**:
```python
class ResBlockWithCrossAttention(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True
        )
        self.context_proj = nn.Linear(context_dim, channels)

        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, z_seq):
        # x: [B, C, H, W], z_seq: [B, L, D]
        B, C, H, W = x.shape

        h = self.conv1(F.silu(self.norm1(x)))

        # Cross-attention
        h_flat = h.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        context = self.context_proj(z_seq)       # [B, L, C]
        h_attn, _ = self.cross_attn(h_flat, context, context)
        h = h + h_attn.permute(0, 2, 1).view(B, C, H, W)

        h = self.conv2(F.silu(self.norm2(h)))
        return x + h
```

**Benefits**:
- Captures temporal variations in noise
- Model learns to attend to relevant noise characteristics
- Better for non-stationary noise

**Considerations**:
- Increased computational cost
- More complex architecture changes
- May need more training data to learn attention patterns

---

### 2.4 Solution D: Noise Data Augmentation

**Concept**: Expand the training noise distribution by augmenting with diverse noise sources.

**Augmentation Strategies**:

1. **Additional Noise Datasets**:
   - ESC-50 (environmental sounds)
   - AudioSet (large-scale)
   - UrbanSound8K (urban noise)
   - WHAM! (ambient noise)

2. **Synthetic Noise Generation**:
   ```python
   def generate_synthetic_noise(length, sr=16000):
       noise_type = random.choice(['white', 'pink', 'brown', 'babble'])
       if noise_type == 'white':
           return np.random.randn(length)
       elif noise_type == 'pink':
           return pink_noise(length)
       elif noise_type == 'brown':
           return brown_noise(length)
       elif noise_type == 'babble':
           return generate_babble(length)
   ```

3. **Noise Mixing**:
   ```python
   def mix_noises(noise1, noise2, alpha=None):
       if alpha is None:
           alpha = np.random.uniform(0.3, 0.7)
       return alpha * noise1 + (1 - alpha) * noise2
   ```

4. **Temporal Augmentation**:
   - Time stretching
   - Random cropping from different positions
   - Concatenating noise segments

**Benefits**:
- Expands embedding space coverage
- More robust to diverse noise types
- Easy to implement

---

## 3. Task Priority and Experiment Tracking

### 3.0 Task Priority (Expected Impact Order)

| Rank | Task | Expected Impact | Difficulty | Status |
|------|------|-----------------|------------|--------|
| 1 | **CFG Implementation** | ⭐⭐⭐⭐⭐ | Low | ✅ Completed |
| 2 | **CLAP Encoder** | ⭐⭐⭐⭐⭐ | Medium | ✅ Completed |
| 3 | **Scaled-up Training** | ⭐⭐⭐⭐⭐ | Low | 🔲 Not Started |
| 4 | **Noise Augmentation** | ⭐⭐⭐ | Low | 🔲 Not Started |
| 5 | **Cross-Attention** | ⭐⭐⭐ | High | 🔲 Not Started |

**Legend**: 🔲 Not Started | 🔄 In Progress | ✅ Completed | ❌ Abandoned

---

## 4. Experimental Design

### 4.1 Phase 1: Classifier-Free Guidance (Priority: High)

**Objective**: Validate CFG improves OOD generalization with minimal changes.

**Experiments**:

| Exp ID | p_uncond | w (inference) | Dataset | Metrics |
|--------|----------|---------------|---------|---------|
| CFG-01 | 0.1 | 1.0 | VB-DEMAND | PESQ, ESTOI, SI-SDR |
| CFG-02 | 0.1 | 3.0 | VB-DEMAND | PESQ, ESTOI, SI-SDR |
| CFG-03 | 0.1 | 5.0 | VB-DEMAND | PESQ, ESTOI, SI-SDR |
| CFG-04 | 0.2 | 1.0 | VB-DEMAND | PESQ, ESTOI, SI-SDR |
| CFG-05 | 0.2 | 3.0 | VB-DEMAND | PESQ, ESTOI, SI-SDR |
| CFG-06 | 0.2 | 5.0 | VB-DEMAND | PESQ, ESTOI, SI-SDR |

**OOD Evaluation** (for best CFG config):
- ESC-50 noise at SNR 0, 5, 10 dB
- Compare with baseline (no CFG)

**Implementation Changes**:
1. Add `--cond_drop_prob` argument
2. Modify `training_step` to drop conditioning
3. Modify `enhancement_noise_cond.py` to support guidance scale

---

### 4.2 Phase 2: Pre-trained Encoder (Priority: High)

**Objective**: Test if CLAP embeddings generalize better than from-scratch encoder.

**Experiments**:

| Exp ID | Encoder | Freeze | Training | Metrics |
|--------|---------|--------|----------|---------|
| CLAP-01 | CLAP | Yes | 50k steps | PESQ, ESTOI, SI-SDR |
| CLAP-02 | CLAP | No | 50k steps | PESQ, ESTOI, SI-SDR |
| CLAP-03 | CLAP + CFG | Yes | 50k steps | PESQ, ESTOI, SI-SDR |
| PANNs-01 | PANNs | Yes | 50k steps | PESQ, ESTOI, SI-SDR |

**Evaluation**:
- In-distribution: VB-DEMAND test
- OOD: ESC-50, UrbanSound8K noise

**Implementation**:
1. Create `sgmse/clap_encoder.py`
2. Add CLAP dependencies
3. Handle sample rate conversion (16kHz → 48kHz)

---

### 4.3 Phase 3: Cross-Attention (Priority: Medium)

**Objective**: Test if cross-attention helps with non-stationary noise.

**Experiments**:

| Exp ID | Architecture | Noise Type | Metrics |
|--------|--------------|------------|---------|
| XAttn-01 | Cross-Attention | Stationary (DEMAND) | PESQ, ESTOI, SI-SDR |
| XAttn-02 | Cross-Attention | Non-stationary (mixed) | PESQ, ESTOI, SI-SDR |
| XAttn-03 | Global Pool (baseline) | Non-stationary (mixed) | PESQ, ESTOI, SI-SDR |

**Non-stationary Test Set Creation**:
- Concatenate different noise types within single file
- Time-varying SNR
- Sudden noise events

---

### 4.4 Phase 4: Combined Approach (Priority: Medium)

**Objective**: Find best combination of improvements.

**Experiments**:

| Exp ID | Encoder | CFG | Attention | Expected Benefit |
|--------|---------|-----|-----------|------------------|
| COMB-01 | CLAP | Yes | Global | Best OOD generalization |
| COMB-02 | CLAP | Yes | Cross | + Non-stationary handling |
| COMB-03 | CLAP | Yes | Cross + Aug | Full solution |

---

---

## 5. Experiment Results

### 5.1 Phase 1: CFG Results

#### Experiment Rationale

**Problem**: Noise encoder가 학습 시 본 노이즈(DEMAND)에만 의존하여, OOD 노이즈에서 conditioning이 오히려 성능을 저하시킬 수 있음.

**Solution**: Classifier-Free Guidance (CFG)로 모델이 conditioning 없이도 동작하도록 학습. Inference 시 guidance scale로 conditioning 강도 조절.

**Hypotheses**:

| 실험 | 가설 | 검증 방법 |
|------|------|----------|
| **p_uncond=0.1** | 10% dropout으로 unconditional 능력 학습, conditional 성능 유지 | w=1.0에서 baseline과 유사, w>1에서 향상 |
| **p_uncond=0.2** | 더 많은 dropout으로 더 강한 unconditional 능력 | OOD에서 더 안정적, 단 in-distribution 성능 저하 가능 |
| **w (guidance scale)** | w>1로 conditioning 강조, w<1로 약화 | In-dist: w=1~3 최적, OOD: w 조절로 graceful degradation |

**Expected Outcome**:
- In-distribution: w=1.0에서 기존과 유사, w 증가 시 약간 향상 가능
- OOD: w=1.0 (conditional only)보다 w<1.0이나 w>1.0 조절로 더 안정적인 성능

#### Training Runs

| Exp ID | p_uncond | batch_size | steps | wandb_name | Checkpoint | Status |
|--------|----------|------------|-------|------------|------------|--------|
| CFG-01 | 0.1 | 4 | 50k | nc-cfg-p0.1 | logs/e8f9ztov-None | ✅ Done |
| CFG-02 | 0.2 | 4 | 50k | nc-cfg-p0.2 | logs/kvue4el4-None | ✅ Done |

#### In-Distribution Results (VB-DEMAND Test)

| Exp ID | w | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|---|--------|---------|----------|
| Baseline (nc_ref0.25s) | - | 1.59 ± 0.42 | 0.68 ± 0.16 | 10.1 ± 4.3 |
| CFG-01 (p=0.1) | 1.0 | 1.40 ± 0.26 | 0.69 ± 0.15 | 10.3 ± 4.2 |
| CFG-01 (p=0.1) | 3.0 | 1.39 ± 0.26 | 0.68 ± 0.17 | 10.4 ± 4.7 |
| CFG-01 (p=0.1) | 5.0 | 1.37 ± 0.25 | 0.67 ± 0.19 | 10.4 ± 5.1 |
| **CFG-02 (p=0.2)** | **1.0** | **1.86 ± 0.54** | **0.77 ± 0.15** | **12.3 ± 4.6** |
| CFG-02 (p=0.2) | 3.0 | 1.87 ± 0.57 | 0.76 ± 0.17 | 12.3 ± 4.8 |
| CFG-02 (p=0.2) | 5.0 | 1.87 ± 0.58 | 0.75 ± 0.19 | 12.3 ± 4.9 |

#### OOD Results (ESC-50 Noise, SNR 0dB)

| Exp ID | w | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|---|--------|---------|----------|
| Baseline (nc_ref0.25s) | - | 1.12 ± 0.17 | 0.42 ± 0.23 | -1.4 ± 3.4 |
| CFG-01 (p=0.1) | 1.0 | 1.12 ± 0.13 | 0.45 ± 0.21 | -0.5 ± 2.1 |
| CFG-01 (p=0.1) | 3.0 | 1.12 ± 0.14 | 0.42 ± 0.24 | -0.9 ± 2.6 |
| CFG-01 (p=0.1) | 5.0 | 1.12 ± 0.17 | 0.35 ± 0.25 | -2.4 ± 4.4 |
| **CFG-02 (p=0.2)** | **1.0** | **1.18 ± 0.25** | **0.51 ± 0.21** | **0.8 ± 2.0** |
| CFG-02 (p=0.2) | 3.0 | 1.18 ± 0.29 | 0.47 ± 0.24 | 0.4 ± 2.6 |
| CFG-02 (p=0.2) | 5.0 | 1.19 ± 0.28 | 0.44 ± 0.27 | 0.2 ± 2.7 |

#### Analysis

**Key Findings:**

1. **p_uncond=0.2가 p_uncond=0.1보다 일관되게 우수**
   - In-dist: PESQ 1.86 vs 1.40 (+0.46)
   - OOD: PESQ 1.18 vs 1.12, SI-SDR 0.8 vs -0.5

2. **Guidance scale (w) 효과 미미**
   - w=1.0, 3.0, 5.0 간 성능 차이 거의 없음
   - 예상과 달리 w 증가가 성능 향상으로 이어지지 않음

3. **OOD 일반화 개선 확인**
   - CFG-02 (p=0.2, w=1.0)이 baseline 대비 OOD에서 개선
   - SI-SDR: -1.4 → 0.8 (+2.2 dB)
   - ESTOI: 0.42 → 0.51 (+0.09)

**Conclusion:** CFG with p_uncond=0.2가 가장 효과적. Guidance scale 조정보다 dropout 비율이 더 중요.

---

### 5.2 Phase 2: CLAP Encoder Results

#### Experiment Rationale

**Problem**: 현재 NoiseEncoder는 DEMAND 노이즈만 학습하여 OOD 노이즈에 일반화가 어려움.

**Solution**: Pre-trained CLAP (Contrastive Language-Audio Pretraining)은 대규모 오디오 데이터(AudioSet 등)로 학습되어 다양한 소리에 대한 일반화된 representation을 제공함.

#### Why LAION-CLAP?

**Encoder 후보군 비교:**

| Encoder | 학습 데이터 | 특징 | 선택 이유 |
|---------|------------|------|----------|
| **LAION-CLAP** ✓ | AudioSet (5M clips), LAION-Audio-630K | Contrastive audio-text learning | 범용 오디오 이해, 환경음/노이즈에 강함 |
| PANNs | AudioSet (2M clips) | Audio tagging 목적 | 대안으로 고려 가능 |
| BEATs | AudioSet | Self-supervised, SOTA audio classification | 복잡한 구조, 무거움 |
| wav2vec 2.0 | LibriSpeech | Speech-focused SSL | 음성 특화, 노이즈 부적합 |
| HuBERT | LibriSpeech | Speech-focused SSL | 음성 특화, 노이즈 부적합 |
| AudioMAE | AudioSet | Masked autoencoder | 최신이나 구현 복잡 |

**LAION-CLAP 선택 근거:**

1. **학습 데이터 다양성**: AudioSet + LAION-Audio-630K로 환경음, 음악, 음성 등 다양한 소리 포함
2. **Contrastive Learning**: Audio-text pair로 학습하여 의미론적 오디오 이해 가능
3. **Embedding 차원**: 512차원으로 현재 아키텍처와 일치
4. **오픈소스 & 접근성**: `pip install laion-clap`으로 쉽게 사용 가능
5. **검증된 성능**: AudioLDM, Make-An-Audio 등 생성 모델에서 검증됨
6. **노이즈 적합성**: wav2vec/HuBERT는 음성 특화라 환경 노이즈 인코딩에 부적합

**향후 비교 실험 (Optional)**:
- PANNs와 비교 실험 가능 (audio tagging 특화)
- 성능 차이 없으면 더 가벼운 모델 선택

**Hypotheses**:

| 실험 | 가설 | 검증 방법 |
|------|------|----------|
| **CLAP-frozen** | Pre-trained representation이 noise encoding에 충분히 유용하다 | Frozen CLAP + projection layer만으로 baseline 대비 OOD 성능 향상 |
| **CLAP-finetune** | Task-specific fine-tuning이 추가 성능 향상을 가져온다 | Fine-tuned vs Frozen 비교. 단, overfitting 위험 모니터링 필요 |
| **CLAP-CFG** | CLAP의 일반화 + CFG의 guidance가 시너지 효과를 낸다 | CLAP-frozen + CFG가 개별 적용보다 OOD에서 더 좋은 성능 |

**Expected Outcome**:
- In-distribution: Baseline과 유사하거나 약간 낮을 수 있음 (CLAP이 noise-specific하지 않으므로)
- OOD: 유의미한 성능 향상 기대 (CLAP의 일반화 능력)

#### Training Runs

| Exp ID | Encoder | Freeze | p_uncond | steps | wandb_name | Checkpoint | Status |
|--------|---------|--------|----------|-------|------------|------------|--------|
| CLAP-01 | CLAP | Yes | - | 50k | nc-clap-frozen | logs/zxihf3ec-nc-clap-frozen | ✅ Done |
| CLAP-02 | CLAP | No | - | 50k | nc-clap-finetune | - | ❌ Skipped (DDP issue) |
| CLAP-CFG-0.1 | CLAP + CFG | Yes | 0.1 | 50k | nc-clap-cfg | logs/8y7s6j16-nc-clap-cfg | ✅ Done |
| CLAP-CFG-0.2 | CLAP + CFG | Yes | 0.2 | 50k | nc-clap-cfg-p0.2 | logs/40hr9bep-nc-clap-cfg-p0.2 | ✅ Done |

**Note**: CLAP-finetune (CLAP-02) was skipped due to DDP unused parameters issue. CLAP fine-tuning requires gradient-compatible forward pass which is not currently supported.

#### In-Distribution Results (VB-DEMAND Test)

| Exp ID | p_uncond | w | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|----------|---|--------|---------|----------|
| CLAP-01 (frozen) | - | 1.0 | 1.70 ± 0.48 | 0.74 ± 0.17 | 11.5 ± 4.5 |
| **CLAP-CFG-0.1** | **0.1** | **1.0** | **1.83 ± 0.54** | **0.75 ± 0.17** | **12.1 ± 4.6** |
| CLAP-CFG-0.1 | 0.1 | 3.0 | 1.77 ± 0.54 | 0.72 ± 0.20 | 11.7 ± 5.2 |
| CLAP-CFG-0.1 | 0.1 | 5.0 | 1.71 ± 0.53 | 0.69 ± 0.21 | 11.1 ± 5.5 |
| CLAP-CFG-0.2 | 0.2 | 1.0 | 1.30 ± 0.21 | 0.64 ± 0.16 | 9.1 ± 4.0 |
| CLAP-CFG-0.2 | 0.2 | 3.0 | 1.28 ± 0.21 | 0.60 ± 0.19 | 8.5 ± 4.5 |
| CLAP-CFG-0.2 | 0.2 | 5.0 | 1.26 ± 0.22 | 0.56 ± 0.21 | 7.9 ± 5.1 |

#### OOD Results (ESC-50 Noise, SNR 0dB)

| Exp ID | p_uncond | w | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|----------|---|--------|---------|----------|
| CLAP-01 (frozen) | - | 1.0 | 1.13 ± 0.21 | 0.46 ± 0.20 | -0.5 ± 3.2 |
| **CLAP-CFG-0.1** | **0.1** | **1.0** | **1.18 ± 0.24** | **0.50 ± 0.23** | **0.1 ± 2.1** |
| CLAP-CFG-0.1 | 0.1 | 3.0 | 1.16 ± 0.22 | 0.45 ± 0.25 | -0.5 ± 1.6 |
| CLAP-CFG-0.1 | 0.1 | 5.0 | 1.15 ± 0.24 | 0.41 ± 0.26 | -0.7 ± 2.7 |
| CLAP-CFG-0.2 | 0.2 | 1.0 | 1.09 ± 0.09 | 0.37 ± 0.17 | -2.2 ± 4.8 |
| CLAP-CFG-0.2 | 0.2 | 3.0 | 1.09 ± 0.09 | 0.35 ± 0.17 | -2.5 ± 4.5 |
| CLAP-CFG-0.2 | 0.2 | 5.0 | 1.08 ± 0.07 | 0.31 ± 0.18 | -2.4 ± 3.5 |

#### Analysis

**Key Findings:**

1. **CLAP-CFG에서 p_uncond=0.1이 최적** (p=0.2는 성능 저하)
   - In-dist: p=0.1 → PESQ 1.83, p=0.2 → PESQ 1.30 (-0.53)
   - OOD: p=0.1 → SI-SDR 0.1, p=0.2 → SI-SDR -2.2 (-2.3 dB)

2. **CNN과 CLAP에서 optimal p_uncond가 다름**
   - CNN: p=0.2가 최적 (PESQ 1.86)
   - CLAP: p=0.1이 최적 (PESQ 1.83)
   - 가설: CLAP embedding은 이미 풍부한 정보를 담고 있어 높은 dropout이 유용한 정보 손실 야기

3. **Guidance scale 증가 시 일관된 성능 저하**
   - 모든 설정에서 w=1.0이 최적

4. **최종 비교: CNN-CFG (p=0.2) vs CLAP-CFG (p=0.1)**
   - In-dist: CNN 1.86 vs CLAP 1.83 (CNN 우위)
   - OOD SI-SDR: CNN 0.8 vs CLAP 0.1 (CNN 우위)

**Conclusion:**
- **CNN + CFG (p=0.2)가 최종 best configuration**
- CLAP의 pre-trained representation이 항상 우수하지는 않음
- Encoder 특성에 따라 최적 CFG dropout rate가 다름

---

### 5.3 Phase 3: Cross-Attention Results

#### Training Runs

| Exp ID | Architecture | steps | wandb_name | Checkpoint | Status |
|--------|--------------|-------|------------|------------|--------|
| XAttn-01 | Cross-Attn | 50k | nc-xattn | TBD | 🔲 |
| XAttn-CFG | Cross-Attn + CFG | 50k | nc-xattn-cfg | TBD | 🔲 |

#### Non-Stationary Noise Results

| Exp ID | Noise Type | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|------------|--------|---------|----------|
| Baseline (Global Pool) | Non-stationary | TBD | TBD | TBD |
| XAttn-01 | Non-stationary | TBD | TBD | TBD |
| XAttn-CFG | Non-stationary | TBD | TBD | TBD |

---

### 5.4 Comparison Summary

#### PoC Results (50k steps, batch=4, 1 GPU)

| Method | VB-DEMAND PESQ | OOD PESQ | OOD SI-SDR | Notes |
|--------|----------------|----------|------------|-------|
| Baseline (nc_ref0.25s) | 1.59 ± 0.42 | 1.12 ± 0.17 | -1.4 ± 3.4 | From-scratch encoder |
| **CFG (p=0.2, w=1.0)** | **1.86 ± 0.54** | **1.18 ± 0.25** | **0.8 ± 2.0** | Best overall (PoC) |
| CLAP-frozen | 1.70 ± 0.48 | 1.13 ± 0.21 | -0.5 ± 3.2 | Pre-trained encoder |
| CLAP-CFG (w=1.0) | 1.83 ± 0.54 | 1.18 ± 0.24 | 0.1 ± 2.1 | CLAP + CFG (p=0.1) |

#### Scaled Results (200k steps, batch=8×4, 4 GPUs) - Pending

| Method | VB-DEMAND PESQ | OOD PESQ | OOD SI-SDR | Notes |
|--------|----------------|----------|------------|-------|
| CFG scaled (p=0.2) | TBD | TBD | TBD | Phase 5 |
| CLAP-CFG scaled | TBD | TBD | TBD | Phase 5 |
| SGMSE+ Baseline | TBD | TBD | TBD | No conditioning |
| SGMSE+ Paper [2] | 2.93 | - | 17.3 | Reference |

**Key Insights:**
- CFG (p=0.2)가 현재까지 최고 성능
- CLAP-CFG는 p_uncond=0.1로 학습됨 → p_uncond=0.2로 재학습 시 추가 향상 기대
- OOD SI-SDR에서 CFG > CLAP-CFG > CLAP-frozen > Baseline 순

---

### 5.5 Phase 5: Scaled-up Training (Paper-level Comparison)

#### Experiment Rationale

**Problem**: 현재까지의 실험들은 50k steps, batch_size=4 (single GPU)로 진행됨. 논문에서 보고된 SGMSE+ 성능과 직접 비교하려면 동일한 학습 조건이 필요함.

**Solution**: 4 GPU를 활용하여 논문 수준의 학습 진행:
- Effective batch size 증가 (4 → 32)
- 학습 steps 증가 (50k → 200k)
- 원본 SGMSE+ 체크포인트와 동일 조건 비교

**Hypotheses**:

| 실험 | 가설 | 검증 방법 |
|------|------|----------|
| **Steps 증가 (50k→200k)** | 더 긴 학습이 수렴 및 성능 향상 | 50k vs 200k 동일 조건 비교 |
| **Batch size 증가** | Larger batch로 안정적인 gradient estimation | Single GPU vs 4 GPU DDP 비교 |
| **CFG + Scale-up** | CFG의 이점이 scale-up에서도 유지됨 | Scaled CFG vs Scaled baseline |

**Expected Outcome**:
- 50k → 200k: PESQ +0.1~0.2 향상 예상
- Batch size 증가: 학습 안정성 향상, 성능 유사하거나 소폭 향상
- 원본 SGMSE+ (no conditioning) 대비 noise-cond + CFG 우위 확인

#### Training Configuration Comparison

| Setting | PoC (현재) | Paper-level | 비고 |
|---------|----------|-------------|------|
| GPUs | 1 | 4 | DDP strategy |
| Batch size/GPU | 4 | 8 | 메모리 허용 시 |
| Effective batch | 4 | 32 | 8×4 GPUs |
| Steps | 50k | 200k | 4배 증가 |
| Learning rate | 1e-4 | 1e-4 | 동일 유지 |
| EMA decay | 0.999 | 0.999 | 동일 유지 |

#### Training Runs

| Exp ID | Method | GPUs | batch/GPU | Steps | wandb_name | Status |
|--------|--------|------|-----------|-------|------------|--------|
| SCALE-01 | CFG (p=0.2) | 4 | 8 | 200k | nc-cfg-scaled | 🔲 |
| SCALE-02 | CLAP-CFG (p=0.2) | 4 | 8 | 200k | nc-clap-cfg-scaled | 🔲 |
| SCALE-03 | Baseline (no cond)* | 4 | 8 | 200k | sgmse-baseline | 🔲 |

*SCALE-03: 원본 SGMSE+ 학습 또는 pretrained checkpoint 사용

#### Training Commands

**SCALE-01: CFG (p=0.2) Scaled**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_noise_cond.py \
    --base_dir ./data/voicebank-demand \
    --backbone ncsnpp_v2_cond \
    --batch_size 8 \
    --devices 4 \
    --max_steps 200000 \
    --cond_drop_prob 0.2 \
    --wandb_name nc-cfg-scaled \
    --save_ckpt_interval 50000
```

**SCALE-02: CLAP-CFG (p=0.2) Scaled**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_noise_cond.py \
    --base_dir ./data/voicebank-demand \
    --backbone ncsnpp_v2_cond \
    --batch_size 8 \
    --devices 4 \
    --max_steps 200000 \
    --noise_encoder_type clap \
    --freeze_clap \
    --return_ref_waveform \
    --cond_drop_prob 0.2 \
    --wandb_name nc-clap-cfg-scaled \
    --save_ckpt_interval 50000
```

#### Evaluation Plan

**In-Distribution (VB-DEMAND Test)**:
| Exp ID | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|--------|---------|----------|
| SCALE-01 (CFG scaled) | TBD | TBD | TBD |
| SCALE-02 (CLAP-CFG scaled) | TBD | TBD | TBD |
| SCALE-03 (Baseline) | TBD | TBD | TBD |

**OOD (ESC-50 Noise, SNR 0dB)**:
| Exp ID | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ |
|--------|--------|---------|----------|
| SCALE-01 (CFG scaled) | TBD | TBD | TBD |
| SCALE-02 (CLAP-CFG scaled) | TBD | TBD | TBD |
| SCALE-03 (Baseline) | TBD | TBD | TBD |

**Reference (SGMSE+ Paper [2] VB-DEMAND)**:
| Method | PESQ | ESTOI | SI-SDR |
|--------|------|-------|--------|
| SGMSE+ (reported) | 2.93 | 0.87 | 17.3 |

**Success Criteria**:
1. SCALE-01/02가 50k 실험 대비 성능 향상
2. Noise-conditioned model이 baseline (SCALE-03) 대비 OOD에서 우수
3. In-distribution에서 SGMSE+ paper 성능에 근접 (PESQ > 2.5 목표)

---

## 6. Implementation Roadmap

### 6.1 Immediate (This Week)

1. **CFG Implementation**
   - [x] Add `cond_drop_prob` to `NoiseCondScoreModel` (already in model_cond.py)
   - [x] Modify training loop for conditional dropout (already implemented)
   - [x] Add guidance scale to enhancement script (--cfg_scale in enhancement_noise_cond.py)
   - [ ] Run CFG experiments (🔄 Training in progress: p_uncond=0.1, 0.2)

### 6.2 Short-term (Next 2 Weeks)

2. **CLAP Integration**
   - [ ] Create `CLAPNoiseEncoder` class
   - [ ] Handle audio preprocessing (resample, normalize)
   - [ ] Test frozen vs fine-tuned CLAP
   - [ ] Run CLAP experiments

### 6.3 Medium-term (Month)

3. **Cross-Attention Architecture**
   - [ ] Implement `SequenceNoiseEncoder`
   - [ ] Modify backbone for cross-attention
   - [ ] Create non-stationary test set
   - [ ] Run cross-attention experiments

4. **Combined System**
   - [ ] Integrate best components
   - [ ] Full evaluation on all test sets
   - [ ] Ablation studies

---

## 7. Success Metrics

### 7.1 Primary Metrics

| Scenario | Target Improvement |
|----------|-------------------|
| In-distribution (VB-DEMAND) | Match or exceed baseline |
| OOD (ESC-50 noise) | >0.3 PESQ improvement over no-conditioning |
| Non-stationary noise | >0.2 PESQ improvement over global pooling |

### 7.2 Secondary Metrics

- Training convergence speed
- Inference latency
- Model size increase

---

## 8. References

1. [Ho & Salimans 2022] Classifier-Free Diffusion Guidance
2. [AudioLDM 2023] Text-to-Audio Generation with Latent Diffusion Models
3. [CLAP 2023] Contrastive Language-Audio Pretraining
4. [CLAPSep 2024] Multi-Modal Query-Conditioned Target Sound Extraction
5. [GDiffuSE 2025] Guided Diffusion for Speech Enhancement
6. [URGENT Challenge 2025] Universal Speech Enhancement
7. [Rethinking CFG 2024] No Training, No Problem

---

## Appendix: Code Snippets

### A.1 CFG Training Modification

```python
# In NoiseCondScoreModel.training_step()

def training_step(self, batch, batch_idx):
    x, y, r = batch

    # Encode noise reference
    z_r = self.noise_encoder(r)

    # Classifier-free guidance: randomly drop conditioning
    if self.training and self.cond_drop_prob > 0:
        mask = torch.rand(z_r.shape[0], device=z_r.device) < self.cond_drop_prob
        z_r = torch.where(mask.unsqueeze(-1), torch.zeros_like(z_r), z_r)

    # Rest of training step...
```

### A.2 CFG Inference

```python
# In enhancement_noise_cond.py

def enhance_with_cfg(model, y, z_r, guidance_scale=3.0):
    # Conditional score
    score_cond = model.score(x_t, y, z_r, t)

    # Unconditional score
    z_null = torch.zeros_like(z_r)
    score_uncond = model.score(x_t, y, z_null, t)

    # Guided score
    score = score_uncond + guidance_scale * (score_cond - score_uncond)

    return score
```

### A.3 CLAP Encoder

```python
# sgmse/clap_encoder.py

import torch
import torch.nn as nn
from laion_clap import CLAP_Module
import torchaudio.transforms as T

class CLAPNoiseEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze=True):
        super().__init__()

        # Load pre-trained CLAP
        self.clap = CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.clap.load_ckpt()

        if freeze:
            for p in self.clap.parameters():
                p.requires_grad = False

        # Resample 16kHz -> 48kHz for CLAP
        self.resample = T.Resample(16000, 48000)

        # Projection layer
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, audio):
        # audio: [B, T] at 16kHz
        audio_48k = self.resample(audio)

        with torch.no_grad():
            emb = self.clap.get_audio_embedding_from_data(
                audio_48k.cpu().numpy(),
                use_tensor=False
            )
            emb = torch.from_numpy(emb).to(audio.device)

        return self.proj(emb)
```
