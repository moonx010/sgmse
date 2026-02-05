# 멘토님께 드리는 현황 공유

지금 연구하는거 결과가 잘 안나와서, 잠시 실험을 멈추고 관련 연구들 및 방법론들, 그리고 논문에서 주장할 수 있는 유효한 contribution에 대해서 고민을 해보았는데요.

---

## 현재 뭐가 문제였는지

기존에 CNN encoder + CFG로 학습했던 결과가 예상과 많이 달랐습니다.

| Model | In-dist PESQ | OOD SI-SDR |
|-------|--------------|------------|
| SGMSE+ baseline | 1.88 | -0.2 |
| NC+CFG (p=0.2) | 1.75 | -0.6 |

baseline도 논문(PESQ 2.9)보다 훨씬 낮게 나왔고, CFG를 적용한 모델은 오히려 baseline보다 더 안 좋게 나왔습니다. OOD에서 개선되어야 하는데 오히려 악화됐어요.

관련 연구들(NASE, NADiffuSE, DiTSE)을 찾아보니까 원인이 보이더라구요:

1. **Noise encoder가 너무 약함**: NASE는 pretrained BEATs(768-dim)를 쓰는데, 저희는 scratch CNN(512-dim)을 썼음
2. **Encoder supervision이 없음**: NASE는 NC loss로 encoder가 noise type을 구분하도록 강제하는데, 저희는 그게 없었음
3. **CFG dropout이 오히려 방해**: 이미 약한 encoder에서 20%를 drop하니까 학습 신호가 더 약해진 것 같음

---

## 연구 방향 전환

그래서 NASE의 강점(BEATs encoder + NC loss)을 가져오되, CFG로 **OOD graceful degradation**을 추가하는 방향으로 pivot하려고 합니다.

**NASE와의 차이점:**
- NASE는 OOD noise에서 misleading conditioning 문제가 있음 (잘못된 embedding이 들어가면 오히려 성능 하락)
- 저희는 CFG를 통해 unreliable한 conditioning일 때 unconditional로 fallback할 수 있음

**DiTSE(2025)와의 차이점:**
- DiTSE도 CFG를 SE에 적용하긴 했는데, "conditioning을 더 잘 사용하게" 하는 목적이고 OOD는 언급이 없음
- 저희는 **OOD에서 graceful degradation**이 핵심 contribution

---

## 수정된 Contribution

1. **Graceful Degradation via CFG**: OOD noise에서 misleading conditioning을 방지
2. **Adaptive Guidance Scale**: inference time에 w를 조절해서 conditioning 강도 제어 (w=0이면 완전 unconditional)
3. **Empirical Analysis**: 언제 conditioning이 도움이 되고, 언제 해가 되는지 분석

---

## 앞으로의 계획 (D-18)

| 기간 | 할 일 |
|------|-------|
| 2/6-2/12 | NASE baseline(p=0) + Ours(p=0.2) 학습 |
| 2/13-2/19 | Main comparison + Ablation 실험 |
| 2/20-2/24 | 논문 작성 및 제출 |

코드는 이미 준비해놨습니다:
- `sgmse/model_nase.py`: NASE-style model + CFG
- `sgmse/beats_encoder.py`: BEATs encoder + NC head
- `train_nase.py`, `enhancement_nase.py`

---

## 논의하고 싶은 점

1. NASE + CFG 조합이 충분한 novelty가 될 수 있을지?
2. 18일 안에 실험 + 논문 작성이 현실적인지?
3. 만약 CFG가 OOD에서도 효과가 없으면 Plan B를 어떻게 가져가야 할지?

상세 내용은 `docs/MENTOR_MEETING_NOTES_0206.md`에 정리해놨습니다.

내일 미팅에서 같이 얘기해보면 좋을 것 같습니다!
