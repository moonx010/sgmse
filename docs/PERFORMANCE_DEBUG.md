# Paper Improvement & Debugging Tracker

> 이 문서는 Interspeech 2026 논문 제출을 위해 해결해야 할 기술적 문제와 논문 품질 개선 사항을 추적합니다.

---

## 0. Reviewer Perspective Analysis (Interspeech 기준)

### 0.1 예상 Rejection 사유

#### ❌ Critical Issues (Accept 불가)

| Issue | Severity | 현재 상태 | 해결 방안 |
|-------|----------|----------|----------|
| **실험 결과가 주장을 뒷받침하지 않음** | Critical | CFG가 OOD에서 오히려 나쁨 | 원인 파악 및 재실험 필수 |
| **Baseline 재현 실패** | Critical | PESQ 2.9 vs 1.88 | Eval 설정 검증 필요 |
| **Novelty 부족** | Major | CFG를 SE에 적용만 함 | 추가 contribution 필요 |

#### ⚠️ Major Issues (Major Revision)

| Issue | Description | 해결 방안 |
|-------|-------------|----------|
| **비교 실험 부족** | 다른 noise-aware SE 방법과 비교 없음 | MetricGAN+, DEMUCS 등과 비교 |
| **OOD 데이터셋 단일** | ESC-50만 사용 | UrbanSound8K, AudioSet 추가 |
| **Analysis 부족** | Noise encoder가 뭘 학습하는지 분석 없음 | t-SNE, attention map 시각화 |
| **Ablation 불충분** | Reference length, guidance scale 등 | 체계적 ablation 추가 |

#### 📝 Minor Issues (Minor Revision)

| Issue | Description |
|-------|-------------|
| Abstract/Conclusion 미작성 | 실험 완료 후 작성 필요 |
| Related Work 섹션 없음 | 필요시 추가 |
| Figure 부재 | Architecture diagram 필요 |

---

### 0.2 논문 강화를 위한 필수 실험

#### A. Baseline 검증 (최우선)
```
목표: 우리 evaluation이 정확한지 확인
방법: 논문 pretrained checkpoint로 동일 결과 재현
기대: PESQ ~2.9, SI-SDR ~17
```

#### B. Conditioning 효과 증명
```
목표: Noise conditioning이 실제로 작동하는지 증명
실험:
1. Zero embedding vs Real embedding 비교
2. Random noise reference vs Oracle reference 비교
3. Mismatched noise reference 테스트 (다른 noise type으로 conditioning)
기대: Real > Zero, Oracle > Random, Matched > Mismatched
```

#### C. OOD 일반화 증명
```
목표: 다양한 OOD 환경에서 개선 확인
실험:
1. ESC-50 (현재)
2. UrbanSound8K
3. AudioSet subset
4. Real-world recordings
기대: 모든 OOD에서 baseline 대비 개선
```

#### D. 비교 실험
```
목표: 다른 방법들과 공정한 비교
비교 대상:
1. SGMSE+ (baseline)
2. MetricGAN+ (discriminative)
3. DEMUCS (end-to-end)
4. CDiffuSE (다른 diffusion SE)
```

#### E. Analysis & Visualization
```
목표: 논문의 설득력 강화
실험:
1. Noise embedding t-SNE (noise type별 clustering)
2. Conditional vs Unconditional score difference map
3. Enhancement 과정 시각화 (spectrogram)
4. Failure case 분석
```

---

### 0.3 Related Work 정리 (차별화 포인트)

#### 직접 경쟁 논문들

| Paper | Venue | Method | 한계점 | 우리와의 차이 |
|-------|-------|--------|--------|--------------|
| **NASE** | Interspeech 2023 | Noise classification → embedding | 잘못된 분류 시 성능 저하 | CFG로 unconditional fallback |
| **NADiffuSE** | ASRU 2023 | Direct noise encoding | OOD 명시적 처리 없음 | CFG dropout으로 OOD robustness |
| **N-HANS** | 2021 | Auxiliary sub-networks | Task-specific 학습 필요 | End-to-end joint training |

#### 우리의 차별점 (Novelty)
1. **CFG 기반 OOD Robustness**: 기존 방법들은 conditioning이 정확하다고 가정. 우리는 CFG로 unreliable conditioning에 대한 graceful degradation 제공
2. **Classification 불필요**: NASE와 달리 discrete noise class 없이 continuous embedding 직접 사용
3. **Inference-time 유연성**: Guidance scale w로 conditioning 강도 조절 가능

#### References
- NASE: https://arxiv.org/abs/2307.08029
- NADiffuSE: https://arxiv.org/abs/2309.01212

---

### 0.4 논문 Contribution 강화 방안

현재 contribution이 약함. 다음 중 1-2개 추가 필요:

| 추가 Contribution | 난이도 | Impact | 설명 |
|------------------|--------|--------|------|
| **Noise-type adaptive guidance** | Medium | High | Noise type에 따라 guidance scale 자동 조절 |
| **Self-supervised noise encoder** | High | High | Contrastive learning으로 noise encoder 사전학습 |
| **Lightweight noise encoder** | Low | Medium | 효율적인 encoder로 실시간 처리 가능 |
| **Multi-condition fusion** | Medium | Medium | SNR + noise type 동시 conditioning |
| **Theoretical analysis** | High | High | CFG가 왜 OOD에 도움되는지 이론적 분석 |

---

## 1. Problem Statement

### 1.1 현재 상황
Noise-conditioned SGMSE+ with CFG가 기대한 성능 개선을 보이지 않음.

### 1.2 기대 vs 실제 결과

| Model | Dataset | Expected PESQ | Actual PESQ | Expected SI-SDR | Actual SI-SDR |
|-------|---------|---------------|-------------|-----------------|---------------|
| SGMSE+ baseline | VB-DEMAND | ~2.9 (논문) | 1.88 | ~17 (논문) | 13.1 |
| SGMSE+ baseline | OOD (ESC-50) | - | 1.17 | - | -0.2 |
| CFG p=0.2 | VB-DEMAND | ≥ baseline | 1.75 | ≥ baseline | 11.8 |
| CFG p=0.2 | OOD (ESC-50) | > baseline | 1.17 | > baseline | -0.6 |

### 1.3 핵심 문제
1. **Baseline 성능 gap**: 논문 PESQ 2.9 vs 우리 1.88 (1.0 차이)
2. **CFG가 오히려 성능 저하**: In-dist에서 baseline보다 나쁨 (1.88 → 1.75)
3. **OOD 개선 없음**: CFG가 OOD에서도 baseline보다 나쁨 (-0.2 → -0.6)
4. **PoC와 Scaled 결과 불일치**: PoC에서는 CFG가 OOD +1.4dB 개선이었음

---

## 2. 가설 및 분석

### 2.1 Hypothesis A: Evaluation 설정 문제

**증상**: Baseline 성능이 논문 대비 낮음

**가능한 원인**:
- [ ] N (sampling steps) 부족: 30 vs 논문 50?
- [ ] Corrector 설정 차이
- [ ] EMA weights 미적용
- [ ] 다른 evaluation 프로토콜

**검증 방법**:
```bash
# 1. N=50으로 테스트
python scripts/eval_batch.py --phase all --gpus 4,5,6,7  # N_STEPS=50

# 2. 논문 pretrained checkpoint로 우리 eval 코드 검증
# 논문 체크포인트 다운로드 후 동일 eval 실행

# 3. EMA 로딩 확인
python -c "
import torch
ckpt = torch.load('./logs/55jxu1gw/last.ckpt', map_location='cpu')
print('EMA in checkpoint:', 'ema' in ckpt)
if 'ema' in ckpt:
    print('EMA keys:', ckpt['ema'].keys())
"
```

---

### 2.2 Hypothesis B: Noise Encoder 문제

**증상**: CFG 모델이 baseline보다 성능 저하

**가능한 원인**:
- [ ] Noise embedding이 유용한 정보를 담지 못함
- [ ] Noise encoder가 underfitting
- [ ] Conditioning injection 방식 문제 (FiLM vs cross-attention)
- [ ] Noise reference 길이/품질 문제

**검증 방법**:
```python
# 1. Noise embedding 분석
# - 같은 noise type의 embedding이 cluster를 형성하는지?
# - t-SNE/UMAP으로 시각화

# 2. Noise encoder output 확인
# - Embedding이 collapse 되지 않았는지 (모두 비슷한 값?)
# - Embedding의 variance 확인

# 3. Ablation: Noise embedding을 zero로 고정하고 테스트
# - 성능 차이 없으면 conditioning이 무시되고 있는 것
```

**코드**:
```python
# embedding_analysis.py
import torch
from sgmse.model_cond import NoiseCondScoreModel

model = NoiseCondScoreModel.load_from_checkpoint(ckpt_path)
model.eval()

# 여러 noise sample의 embedding 추출
embeddings = []
for noise_sample in noise_samples:
    z_r = model.noise_encoder(noise_sample)
    embeddings.append(z_r.detach().cpu())

# 분석
embeddings = torch.stack(embeddings)
print(f"Embedding mean: {embeddings.mean():.4f}")
print(f"Embedding std: {embeddings.std():.4f}")
print(f"Embedding norm range: [{embeddings.norm(dim=-1).min():.4f}, {embeddings.norm(dim=-1).max():.4f}]")
```

---

### 2.3 Hypothesis C: CFG Training 문제

**증상**: CFG p=0.2가 PoC에서는 효과 있었으나 scaled에서는 없음

**가능한 원인**:
- [ ] Scaled training에서 CFG dropout이 다르게 동작
- [ ] Multi-GPU DDP에서 dropout 동기화 문제
- [ ] Batch size 증가로 CFG 효과 감소
- [ ] Learning rate scheduling 차이

**검증 방법**:
```bash
# 1. PoC 설정으로 scaled 모델 재학습
# - 1 GPU, batch=4, 50k steps로 scaled checkpoint에서 fine-tune

# 2. CFG dropout 실제 동작 확인
# - Training 중 실제로 conditioning이 drop 되는지 로깅

# 3. Unconditional score와 conditional score 비교
# - 차이가 없으면 conditioning이 학습 안 된 것
```

---

### 2.4 Hypothesis D: Architecture 문제

**증상**: Noise conditioning 자체가 효과 없음

**가능한 원인**:
- [ ] t_emb + noise_emb 단순 덧셈이 비효과적
- [ ] Noise embedding dimension (512) 부적절
- [ ] Score network가 noise embedding을 무시하도록 학습됨

**검증 방법**:
```python
# 1. Gradient flow 확인
# - Noise encoder로 gradient가 흐르는지 확인

# 2. Feature importance 분석
# - Noise embedding을 perturbation했을 때 output 변화 측정

# 3. 다른 conditioning 방식 테스트
# - Cross-attention
# - Adaptive normalization (AdaIN, AdaGN)
```

---

### 2.5 Hypothesis E: Data/Task 문제

**증상**: Noise conditioning의 근본적 효용 의문

**가능한 원인**:
- [ ] VB-DEMAND noise가 이미 충분히 다양해서 conditioning 불필요
- [ ] Noise reference에서 추출 가능한 정보가 noisy input에 이미 있음
- [ ] Oracle noise reference 설정이 현실적이지 않음

**검증 방법**:
```bash
# 1. Random noise reference로 테스트
# - 성능 차이 없으면 conditioning이 실제로 활용되지 않는 것

# 2. 완전히 다른 noise reference로 테스트
# - 성능 하락 없으면 conditioning 무시되는 것

# 3. Noise type별 성능 분석
# - 특정 noise type에서만 효과 있는지 확인
```

---

## 3. Debugging Priority

### Phase 1: Evaluation 검증 (최우선)
1. [TODO] N=50으로 재평가
2. [TODO] 논문 pretrained checkpoint로 eval 코드 검증
3. [TODO] EMA 로딩 상태 확인

### Phase 2: Conditioning 효과 검증
4. [TODO] Zero embedding vs real embedding 비교
5. [TODO] Random noise reference 테스트
6. [TODO] Noise embedding 시각화 (t-SNE)

### Phase 3: Architecture 분석
7. [TODO] Gradient flow 확인
8. [TODO] Conditional vs unconditional score 차이 분석

### Phase 4: 대안 탐색
9. [TODO] Cross-attention conditioning
10. [TODO] Stronger noise encoder (larger, pretrained)

---

## 4. Action Items

### 즉시 실행
```bash
# 1. N=50 평가 (진행 중)
python scripts/eval_batch.py --phase all --gpus 4,5,6,7

# 2. 논문 pretrained checkpoint 테스트 (진행 중)
# Enhancement (진행 중):
python enhancement.py --test_dir ./data/voicebank-demand/test/noisy --enhanced_dir ./enhanced_pretrained --ckpt pretrained_vbdmd.ckpt --N 50

# Metrics (enhance 완료 후 실행):
python calc_metrics.py --clean_dir ./data/voicebank-demand/test/clean --noisy_dir ./data/voicebank-demand/test/noisy --enhanced_dir ./enhanced_pretrained
```

### 분석 스크립트 작성 필요
- [ ] `scripts/analyze_embeddings.py`: Noise embedding 분석
- [ ] `scripts/test_conditioning.py`: Conditioning 효과 테스트
- [ ] `scripts/compare_scores.py`: Conditional vs unconditional score 비교

---

## 5. 실험 로그

### 2026-02-05: Initial Debug

**N=30 결과**:
| Model | In-dist PESQ | In-dist SI-SDR | OOD PESQ | OOD SI-SDR |
|-------|--------------|----------------|----------|------------|
| sgmse_scaled | 1.88 | 13.1 | 1.17 | -0.2 |
| cfg_p0.2_scaled | 1.75 | 11.8 | 1.17 | -0.6 |

**N=50 결과**: (대기 중)
| Model | In-dist PESQ | In-dist SI-SDR | OOD PESQ | OOD SI-SDR |
|-------|--------------|----------------|----------|------------|
| sgmse_scaled | TBD | TBD | TBD | TBD |
| cfg_p0.2_scaled | TBD | TBD | TBD | TBD |

---

## 6. 참고 자료

### 관련 문서
- `docs/EXPERIMENT_REPORT.md`: 전체 실험 결과
- `docs/NOISE_COND_IMPROVEMENTS.md`: 방법론 상세

### 관련 코드
- `sgmse/model_cond.py`: Noise-conditioned model
- `sgmse/backbones/ncsnpp_v2_cond.py`: Conditioned backbone
- `enhancement_noise_cond.py`: Noise-conditioned enhancement

---

## 7. Paper Improvement Roadmap

### Phase 1: 기술적 문제 해결 (현재)
- [ ] Baseline 성능 gap 원인 파악
- [ ] CFG 모델 성능 저하 원인 파악
- [ ] Evaluation 코드 검증

### Phase 2: 핵심 실험 보완
- [ ] Conditioning 효과 증명 (zero vs real embedding)
- [ ] 추가 OOD 데이터셋 (UrbanSound8K)
- [ ] 비교 실험 (MetricGAN+ 등)

### Phase 3: Analysis 강화
- [ ] Noise embedding 시각화 (t-SNE)
- [ ] Score difference 분석
- [ ] Failure case 분석

### Phase 4: 논문 작성 완료
- [ ] Abstract 작성
- [ ] Results 테이블 업데이트
- [ ] Conclusion 작성
- [ ] Architecture Figure 추가

---

## 8. Quick Reference: 핵심 질문들

논문 accept를 위해 답해야 할 질문들:

1. **Why noise conditioning?**
   - Noisy input에 이미 noise 정보가 있는데 왜 별도 reference가 필요한가?
   - → 답: Explicit conditioning으로 더 정확한 noise characterization 가능

2. **Why CFG?**
   - 단순 conditioning 대신 CFG를 쓰는 이유는?
   - → 답: OOD noise에서 graceful degradation, unconditional fallback

3. **What does noise encoder learn?**
   - Encoder가 의미있는 noise representation을 학습하는가?
   - → 답: t-SNE로 noise type clustering 시각화 필요

4. **When does it fail?**
   - 어떤 상황에서 baseline보다 나빠지는가?
   - → 답: Failure case 분석 필요

5. **Is it practical?**
   - 실제 환경에서 noise reference를 어떻게 얻는가?
   - → 답: Voice activity detection으로 noise-only 구간 추출

---

## 9. Experiment Checklist for Submission

### Must Have (Accept 필수조건)
- [ ] Baseline 성능 재현 (PESQ > 2.5)
- [ ] CFG가 OOD에서 baseline 대비 개선
- [ ] Conditioning 효과 증명 실험
- [ ] 최소 2개 OOD 데이터셋

### Should Have (경쟁력 확보)
- [ ] 1개 이상 비교 방법
- [ ] Noise embedding 시각화
- [ ] Ablation study (p, w, ref_length)

### Nice to Have (강력한 논문)
- [ ] 3개 이상 비교 방법
- [ ] Real-world evaluation
- [ ] 추가 contribution (adaptive guidance 등)
- [ ] Theoretical analysis

---

*Created: 2026-02-05*
*Last Updated: 2026-02-05*
*Status: Active Investigation - Phase 1*
