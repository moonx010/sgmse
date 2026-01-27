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

## 3. Experimental Design

### 3.1 Phase 1: Classifier-Free Guidance (Priority: High)

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

### 3.2 Phase 2: Pre-trained Encoder (Priority: High)

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

### 3.3 Phase 3: Cross-Attention (Priority: Medium)

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

### 3.4 Phase 4: Combined Approach (Priority: Medium)

**Objective**: Find best combination of improvements.

**Experiments**:

| Exp ID | Encoder | CFG | Attention | Expected Benefit |
|--------|---------|-----|-----------|------------------|
| COMB-01 | CLAP | Yes | Global | Best OOD generalization |
| COMB-02 | CLAP | Yes | Cross | + Non-stationary handling |
| COMB-03 | CLAP | Yes | Cross + Aug | Full solution |

---

## 4. Implementation Roadmap

### 4.1 Immediate (This Week)

1. **CFG Implementation**
   - [ ] Add `cond_drop_prob` to `NoiseCondScoreModel`
   - [ ] Modify training loop for conditional dropout
   - [ ] Add guidance scale to enhancement script
   - [ ] Run CFG experiments

### 4.2 Short-term (Next 2 Weeks)

2. **CLAP Integration**
   - [ ] Create `CLAPNoiseEncoder` class
   - [ ] Handle audio preprocessing (resample, normalize)
   - [ ] Test frozen vs fine-tuned CLAP
   - [ ] Run CLAP experiments

### 4.3 Medium-term (Month)

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

## 5. Success Metrics

### 5.1 Primary Metrics

| Scenario | Target Improvement |
|----------|-------------------|
| In-distribution (VB-DEMAND) | Match or exceed baseline |
| OOD (ESC-50 noise) | >0.3 PESQ improvement over no-conditioning |
| Non-stationary noise | >0.2 PESQ improvement over global pooling |

### 5.2 Secondary Metrics

- Training convergence speed
- Inference latency
- Model size increase

---

## 6. References

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
