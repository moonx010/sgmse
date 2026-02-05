# Paper Writing Context

## 1. Paper Information

- **Target Venue**: Interspeech 2026
- **Page Limit**: 4 pages (+ references)
- **Submission Deadline**: TBD
- **Paper File**: `docs/paper.tex`
- **Template**: Interspeech 2026 LaTeX template (double-blind)

---

## 2. Research Overview

### 2.1 Title
**Noise-Conditioned Diffusion for Robust Speech Enhancement with Classifier-Free Guidance**

### 2.2 Core Idea
SGMSE+ 기반 speech enhancement 모델에 noise conditioning을 추가하여 OOD (Out-of-Distribution) noise에 대한 일반화 성능을 향상시킴.

### 2.3 Key Insight
- **Problem**: 기존 speech enhancement 모델은 학습 시 본 noise에서는 잘 동작하지만, 새로운 noise 환경에서 성능이 저하됨
- **Solution**: Classifier-Free Guidance (CFG)를 활용하여 noise conditioning을 학습하면서 동시에 unconditional path도 학습
- **Result**: OOD noise에서 SI-SDR +1.4 dB 개선

---

## 3. Contributions (Paper에 기재된 내용)

1. **Noise-Conditional Diffusion Framework**
   - CFG를 speech enhancement에 적용
   - Noise reference embedding으로 reverse diffusion process를 conditioning

2. **Noise Reference Guidance (NRG)**
   - Inference time에 noise reference를 활용하는 전략
   - Unseen noise에 대한 robustness 향상

3. **Comprehensive Encoder Analysis**
   - CNN vs CLAP encoder 비교
   - Optimal CFG dropout rate는 encoder에 따라 다름 (CNN: p=0.2, CLAP: p=0.1)

---

## 4. Experimental Results Summary

### 4.1 Main Results (PoC: 50k steps, batch=4)

| Method | In-dist PESQ | In-dist SI-SDR | OOD PESQ | OOD SI-SDR |
|--------|--------------|----------------|----------|------------|
| SGMSE+ (baseline) | **1.92** | **12.5** | 1.18 | -0.6 |
| NC + CFG (p=0.2) | 1.86 | 12.3 | **1.18** | **0.8** |

### 4.2 Key Findings

1. **Noise conditioning은 In-dist에서 약간 손해, OOD에서 이득**
   - In-dist: 1.92 → 1.86 (PESQ -0.06)
   - OOD: -0.6 → 0.8 (SI-SDR +1.4 dB)

2. **Optimal p_uncond는 encoder 특성에 따라 다름**
   - CNN encoder: p=0.2 최적
   - CLAP encoder: p=0.1 최적

3. **Guidance scale w는 1.0이 최적**
   - w 증가 시 오히려 OOD 성능 저하

### 4.3 Pending Experiments (Scaled Training)

| Exp ID | Model | Status | Expected Results |
|--------|-------|--------|------------------|
| SCALE-01 | CFG (p=0.2), 4 GPU, 58k steps | Training | PESQ > 2.5 목표 |
| SCALE-02 | SGMSE+ baseline, 4 GPU, 58k steps | Training | Fair comparison baseline |

---

## 5. Paper Structure

### Current Status
- [x] Title, Authors, Keywords
- [x] Abstract (draft - 숫자 업데이트 필요)
- [x] Introduction (구조 완성, 내용 보완 필요)
- [x] Methodology (수식 포함, 보완 필요)
- [x] Experiments (Table 포함, 최종 결과 업데이트 필요)
- [x] Conclusion (draft)
- [ ] Figure 1: Overall architecture diagram
- [ ] References (mybib.bib 작성 필요)

### TODO List

#### High Priority (실험 완료 후)
1. **Table 1 숫자 업데이트**: Scaled training 결과로 교체
2. **Abstract 숫자 업데이트**: 최종 SI-SDR 개선량
3. **Figure 1 작성**: Architecture diagram (TikZ or external image)

#### Medium Priority
4. **Introduction 보완**: Related work 언급 추가
5. **Methodology 보완**: Noise encoder architecture 상세 설명
6. **Related Work section**: 필요시 추가

#### Low Priority
7. **Ablation 추가**: Reference length, non-stationary noise 등
8. **References 추가**: mybib.bib 파일 작성

---

## 6. Key References to Cite

```bibtex
@article{fan2024stable,
  title={Stable speech enhancement with score-based generative models},
  author={Fan, Xiugen and ...},
  journal={IEEE/ACM TASLP},
  year={2024}
}

@inproceedings{ho2022classifier,
  title={Classifier-free diffusion guidance},
  author={Ho, Jonathan and Salimans, Tim},
  booktitle={NeurIPS Workshop},
  year={2022}
}

@inproceedings{richter2023sgmse,
  title={Speech Enhancement and Dereverberation with Diffusion-Based Generative Models},
  author={Richter, Julius and ...},
  booktitle={IEEE/ACM TASLP},
  year={2023}
}
```

---

## 7. Writing Guidelines

### Style (Interspeech)
- 간결하고 명확한 문장
- Passive voice 허용 but active 권장
- 숫자는 소수점 2자리까지 (PESQ), 1자리 (SI-SDR)
- Table caption은 위에, Figure caption은 아래

### 한국어 메모 규칙
- `% TODO:` 로 시작하는 주석은 작성 필요한 부분
- 본문에는 영어만 사용

---

## 8. Related Documents

| Document | Purpose |
|----------|---------|
| `docs/paper.tex` | Main paper LaTeX file |
| `docs/EXPERIMENT_REPORT.md` | Detailed experimental results and analysis |
| `docs/NOISE_COND_IMPROVEMENTS.md` | Technical proposals and implementation details |
| `Interspeech2026-Paper-Kit/` | Template files and examples |
| `mybib.bib` | BibTeX references (to be created) |

---

## 9. Mentor Meeting Notes

### 2025-02-04 Meeting Summary

**피드백 요약:**
- CFG 기반 noise conditioning이 OOD 일반화에 효과적임을 잘 보여줌
- Noise reference guidance 개념을 contribution으로 명확히 제시
- Encoder 설계 분석이 실용적 가이드 제공

**논문 방향:**
- OOD generalization을 main story로
- In-dist 성능 저하는 trade-off로 설명
- Scaled training 결과로 최종 논문 완성

---

*Last updated: 2025-02-04*
