# 멘토님 미팅 사전 공유자료 (2026-02-06)

## 1. 현재 상황 요약

### 1.1 문제점
기존 접근법 (CNN encoder + CFG)으로 학습한 결과, **기대한 성능 개선이 나타나지 않았습니다.**

| Model | In-dist PESQ | OOD SI-SDR | 기대 결과 |
|-------|--------------|------------|----------|
| SGMSE+ baseline | 1.88 | -0.2 | ~2.9 (논문) |
| NC+CFG (p=0.2) | 1.75 | -0.6 | > baseline |

**핵심 문제:**
1. Baseline 재현 실패 (PESQ 2.9 → 1.88)
2. CFG가 오히려 성능 저하 (OOD에서도 baseline보다 나쁨)
3. PoC 결과와 scaled training 결과 불일치

### 1.2 원인 분석

관련 연구 (NASE, NADiffuSE, DiTSE)를 조사한 결과, **우리 구현의 근본적 문제점**을 파악했습니다:

| 요소 | NASE (잘 동작) | 우리 (문제 있음) |
|------|----------------|-----------------|
| **Noise Encoder** | Pre-trained BEATs (768-dim) | 4-layer CNN from scratch (512-dim) |
| **Encoder Supervision** | NC loss (classification) | 없음 |
| **Injection 방식** | Input addition | FiLM via time embedding |
| **CFG** | 없음 | p=0.2 dropout |

**핵심 원인:**
1. **Noise encoder가 너무 약함**: Scratch CNN은 의미있는 noise representation을 학습하기 어려움
2. **Encoder supervision 부재**: NC loss 없이는 encoder가 noise type을 구분하도록 강제되지 않음
3. **CFG dropout이 학습을 방해**: 이미 약한 encoder에서 20%를 drop하면 학습 신호가 더 약해짐

---

## 2. 연구 Pivot 방향

### 2.1 새로운 접근법: NASE + CFG

**핵심 아이디어:**
> NASE의 강점(BEATs encoder + NC loss)을 가져오되, CFG로 **OOD graceful degradation**을 추가

**차별화 포인트:**

| NASE 한계 | 우리의 해결책 |
|-----------|--------------|
| OOD noise에서 misleading conditioning | CFG로 unconditional fallback |
| Guidance strength 고정 | Inference-time guidance scale 조절 |
| OOD 명시적 처리 없음 | w=0으로 OOD에서 안전하게 fallback |

### 2.2 관련 연구와의 차별화

| Paper | Venue | 접근법 | 우리와의 차이 |
|-------|-------|--------|--------------|
| **NASE** | Interspeech 2023 | BEATs + NC loss | 우리: +CFG for OOD |
| **NADiffuSE** | ASRU 2023 | Noise category conditioning | 우리: Continuous embedding + CFG |
| **DiTSE** | arXiv 2025 | CFG for all conditioning | 우리: **CFG for noise only, OOD focus** |

**DiTSE와의 핵심 차이:**
- DiTSE: CFG로 "conditioning을 더 잘 사용하게"
- **우리: CFG로 "unreliable conditioning에서 graceful degradation"**
- DiTSE는 OOD를 언급하지 않음 → 우리의 novelty

---

## 3. 수정된 Contribution

### Paper Title
> **Robust Speech Enhancement via Classifier-Free Guidance: Graceful Degradation for Out-of-Distribution Noise**

### Contributions

1. **Graceful Degradation via CFG**
   - CFG enables noise-conditioned models to fall back to unconditional mode for OOD noise
   - Prevents misleading conditioning from hurting performance

2. **Adaptive Guidance Scale**
   - Inference-time w tuning: w=1 for in-dist, w=0 for OOD
   - Practitioners can control based on expected noise reliability

3. **Empirical Analysis**
   - When does noise conditioning help vs. hurt?
   - How CFG mitigates failure cases

---

## 4. 실험 계획

### 4.1 필수 실험

| 실험 | 목적 | 예상 소요 |
|------|------|----------|
| **E1: Main comparison** | SGMSE+ vs NASE vs Ours | 학습 3-5일 |
| **E2: Guidance scale ablation** | w=0,0.5,1,1.5 비교 | Inference only |
| **E3: CFG dropout ablation** | p=0,0.1,0.2,0.3 비교 | 추가 학습 필요 |
| **E4: Conditioning verification** | Real vs Zero embedding | Inference only |

### 4.2 예상 결과 (가설)

| Method | In-dist PESQ | OOD PESQ |
|--------|--------------|----------|
| SGMSE+ | ~2.9 | ~1.2 |
| NASE (no CFG) | ~3.0 | **~1.1** (misleading) |
| Ours (CFG, w=1.0) | ~3.0 | ~1.3 |
| Ours (CFG, w=0) | ~2.9 | ~1.2 (safe fallback) |

**핵심 가설:**
> CFG를 사용하면 OOD에서 NASE보다 좋거나, 최소한 SGMSE+ baseline만큼은 나와야 함

---

## 5. 일정 (D-18, 2/24 제출)

### Week 1 (2/6-2/12)
- [ ] 2/6: 멘토님 미팅, 방향 확정
- [ ] 2/7-2/9: NASE baseline 학습 (p=0)
- [ ] 2/10-2/12: Ours 학습 (p=0.2)

### Week 2 (2/13-2/19)
- [ ] 2/13-2/15: Main comparison 실험 (E1)
- [ ] 2/16-2/17: Ablation 실험 (E2, E3)
- [ ] 2/18-2/19: Analysis (E4, embedding visualization)

### Week 3 (2/20-2/24)
- [ ] 2/20-2/22: 논문 작성 (Results, Discussion)
- [ ] 2/23: Final review
- [ ] 2/24: **제출**

---

## 6. 코드 준비 상태

### 완료
- [x] `sgmse/model_nase.py`: NASE-style model with CFG
- [x] `sgmse/beats_encoder.py`: BEATs encoder (+ fallback CNN)
- [x] `sgmse/backbones/ncsnpp_nase.py`: Input addition backbone
- [x] `train_nase.py`: Training script
- [x] `docs/paper.tex`: Paper draft with new contributions

### 작업 필요
- [ ] `enhancement_nase.py`: Enhancement script
- [ ] `scripts/eval_nase.py`: Batch evaluation
- [ ] BEATs checkpoint download 및 테스트

---

## 7. 질문 및 논의 사항

### 멘토님께 여쭤볼 점

1. **Pivot 방향 타당성**
   - NASE + CFG 조합이 충분한 novelty인가?
   - DiTSE와의 차별화 (OOD focus)가 설득력 있는가?

2. **일정 현실성**
   - 18일 안에 실험 + 논문 작성 가능한가?
   - 필수 실험 우선순위 조정 필요한가?

3. **만약 가설이 틀리면?**
   - CFG가 OOD에서도 도움 안 되면 어떻게 할지?
   - Plan B: Negative result로 작성? 또는 다른 방향?

---

## 8. 첨부 자료

- `docs/paper.tex`: 논문 초안 (새 contribution 반영)
- `docs/EXPERIMENT_DESIGN.md`: 상세 실험 계획
- `docs/PERFORMANCE_DEBUG.md`: 기존 문제 분석 기록

---

*작성: 2026-02-05*
*다음 미팅: 2026-02-06*
