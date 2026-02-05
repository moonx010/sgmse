# Performance Debugging: Noise-Conditioned SGMSE+

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

*Created: 2026-02-05*
*Status: Active Investigation*
