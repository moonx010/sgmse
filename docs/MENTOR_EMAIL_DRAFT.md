# 멘토님께 공유드리는 연구 현황 및 방향 전환

안녕하세요 멘토님,

내일 미팅 전에 현재 연구 상황과 방향 전환에 대해 미리 공유드리고자 합니다.

---

## 현재 상황

기존 접근법(CNN encoder + CFG)으로 학습한 결과가 기대와 다르게 나왔습니다:
- Baseline 재현 실패: PESQ 2.9(논문) → 1.88(우리)
- CFG 모델이 오히려 baseline보다 나쁨

원인을 분석해본 결과, 관련 연구(NASE, DiTSE 등)와 비교했을 때 **우리 구현의 근본적 문제**를 파악했습니다:
1. Noise encoder가 너무 약함 (scratch CNN vs pretrained BEATs)
2. Encoder supervision 부재 (NC loss 없음)
3. CFG dropout이 이미 약한 encoder의 학습을 더 방해

---

## 연구 방향 전환

**새로운 접근: NASE + CFG**

NASE의 강점(BEATs encoder + NC loss)을 가져오되, CFG로 **OOD graceful degradation**을 추가하는 방향으로 pivot하려 합니다.

**차별화 포인트:**
- NASE: OOD에서 misleading conditioning 문제 있음
- DiTSE: CFG를 사용하지만 OOD 고려 없음
- **우리: CFG로 OOD에서 unconditional fallback** → graceful degradation

수정된 contribution:
1. CFG enables graceful degradation for OOD noise
2. Adaptive guidance scale at inference time
3. Empirical analysis of when conditioning helps vs. hurts

---

## 일정 (D-18)

- 2/6-2/12: NASE baseline + Ours 학습
- 2/13-2/19: 실험 및 ablation
- 2/20-2/24: 논문 작성 및 제출

---

## 논의하고 싶은 점

1. NASE + CFG 조합이 충분한 novelty인지?
2. 18일 안에 실험 + 논문 작성 현실성?
3. 만약 가설이 틀리면(CFG가 OOD에서도 도움 안 되면) Plan B?

상세 내용은 첨부한 문서(`MENTOR_MEETING_NOTES_0206.md`)에 정리해두었습니다.

감사합니다.

---

**첨부:**
- `docs/MENTOR_MEETING_NOTES_0206.md`: 상세 현황 및 계획
- `docs/paper.tex`: 논문 초안
- `docs/EXPERIMENT_DESIGN.md`: 실험 설계
