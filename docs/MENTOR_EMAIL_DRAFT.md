멘토님, 지금 연구하는거 결과가 잘 안나와서 잠시 실험을 멈추고 관련 연구들 및 방법론들, 그리고 논문에서 주장할 수 있는 유효한 contribution에 대해서 고민을 해보았는데요.

일단 현재 문제는, 기존에 CNN encoder + CFG로 학습했던 결과가 예상과 많이 달랐습니다. CFG를 적용한 모델은 오히려 baseline보다 더 안 좋게 나왔습니다. OOD에서도 개선되어야 하는데 오히려 악화됐습니다.

그래서 일단 실험을 잠시 중단하고 오늘 관련 noise conditioning을 적용한 기존 연구들을 찾아보았습니다. 그중 NASE가 SGMSE+에 noise conditioning을 적용한 연구라 해당 방법이 어떻게 설계를 했는지 비교해보았는데,

1) Noise encoder가 너무 약함 - NASE는 pretrained BEATs(768-dim)를 쓰는데, 저희는 scratch CNN(512-dim)을 썼음
2) Encoder supervision이 없음 - NASE는 NC loss로 encoder가 noise type을 구분하도록 강제하는데, 저희는 그게 없었음
3) CFG dropout이 오히려 방해 - 이미 약한 encoder에서 20%를 drop하니까 학습 신호가 더 약해진 것 같음

그래서 NASE의 강점(BEATs encoder + NC loss)을 가져오되, CFG로 OOD graceful degradation을 추가하는 방향으로 pivot하려고 합니다.

NASE와의 차이점은, NASE는 OOD noise에서 misleading conditioning 문제가 있거든요. 잘못된 embedding이 들어가면 오히려 성능이 하락하는데, 저희는 CFG를 통해 unreliable한 conditioning일 때 unconditional로 fallback할 수 있습니다.

그리고 DiTSE(2025)라는 논문도 찾아봤는데, 얘네도 CFG를 SE에 적용했습니다. 근데 얘네는 WavLM feature랑 conditioner network 출력 등 모든 conditioning을 한꺼번에 10% 확률로 drop하는 방식이고, 목적이 "모델이 conditioning feature를 더 잘 활용하게" 하는거예요. 그래서 inference할 때 guidance scale도 따로 조절 안하고, OOD 상황에 대한 고려가 없습니다.

저희는 noise embedding만 선택적으로 drop하고, inference 때 guidance scale w를 조절해서 noise가 신뢰할 수 없을 때(OOD) w를 낮춰서 unconditional 쪽으로 fallback하는게 핵심이에요. 즉 DiTSE는 "conditioning을 더 잘 쓰자"가 목적이고, 저희는 "conditioning이 unreliable할 때 안전하게 fallback하자"가 목적이라 방향이 다릅니다.

그래서 contribution을 다시 정리해보면:

1) OOD Graceful Degradation: noise conditioning이 OOD일 때 misleading되는 문제를 CFG로 해결. guidance scale w=0으로 하면 완전히 unconditional로 fallback해서 최소한 baseline 성능은 보장됨

2) Adaptive Inference: 같은 모델로 in-distribution에서는 w=1로 conditioning 활용하고, OOD에서는 w를 낮춰서 안전하게 동작. 상황에 따라 유연하게 대응 가능

3) Empirical Analysis: noise conditioning이 언제 도움이 되고 언제 해가 되는지, guidance scale에 따라 성능이 어떻게 변하는지 체계적으로 분석

지금은 NASE 스타일 아키텍처로 코드를 다시 짜고 있고, 논문 초안도 새 contribution에 맞게 수정해놨습니다. 실험 설계도 contribution을 증명할 수 있는 방향으로 다시 정리했습니다.

근데 멘토님이랑 논의하고 싶은게, NASE + CFG 조합이 충분한 novelty가 될 수 있을지, 그리고 만약 CFG가 OOD에서도 효과가 없으면 어떻게 해야할지가 좀 고민이에요.

내일 미팅에서 같이 얘기해보면 좋을 것 같습니다!

---
참고한 논문들:
- NASE (Interspeech 2023): https://arxiv.org/abs/2307.08029
- NADiffuSE (ASRU 2023): https://arxiv.org/abs/2309.01212
- DiTSE (2025): https://arxiv.org/abs/2504.09381
