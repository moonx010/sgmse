멘토님, 지금 연구하는거 결과가 잘 안나와서 잠시 실험을 멈추고 관련 연구들 및 방법론들, 그리고 논문에서 주장할 수 있는 유효한 contribution에 대해서 고민을 해보았는데요.

일단 현재 문제가 뭐였냐면, 기존에 CNN encoder + CFG로 학습했던 결과가 예상과 많이 달랐습니다. baseline도 논문(PESQ 2.9)보다 훨씬 낮게(1.88) 나왔고, CFG를 적용한 모델은 오히려 baseline보다 더 안 좋게 나왔어요. OOD에서 개선되어야 하는데 오히려 악화됐습니다.

관련 연구들(NASE, NADiffuSE, DiTSE)을 찾아보니까 원인이 보이더라구요.

1) Noise encoder가 너무 약함 - NASE는 pretrained BEATs(768-dim)를 쓰는데, 저희는 scratch CNN(512-dim)을 썼음
2) Encoder supervision이 없음 - NASE는 NC loss로 encoder가 noise type을 구분하도록 강제하는데, 저희는 그게 없었음
3) CFG dropout이 오히려 방해 - 이미 약한 encoder에서 20%를 drop하니까 학습 신호가 더 약해진 것 같음

그래서 NASE의 강점(BEATs encoder + NC loss)을 가져오되, CFG로 OOD graceful degradation을 추가하는 방향으로 pivot하려고 합니다.

NASE와의 차이점은, NASE는 OOD noise에서 misleading conditioning 문제가 있거든요. 잘못된 embedding이 들어가면 오히려 성능이 하락하는데, 저희는 CFG를 통해 unreliable한 conditioning일 때 unconditional로 fallback할 수 있습니다.

DiTSE(2025)도 찾아봤는데, 얘네도 CFG를 SE에 적용하긴 했어요. 근데 "conditioning을 더 잘 사용하게" 하는 목적이고 OOD는 언급이 없어서, 저희는 OOD에서 graceful degradation을 핵심 contribution으로 잡으면 차별화가 될 것 같습니다.

수정된 contribution은:
1) Graceful Degradation via CFG - OOD noise에서 misleading conditioning 방지
2) Adaptive Guidance Scale - inference time에 w 조절로 conditioning 강도 제어
3) Empirical Analysis - 언제 conditioning이 도움되고 언제 해가 되는지 분석

일정은 D-18인데, 2/6-12에 NASE baseline + Ours 학습하고, 2/13-19에 실험, 2/20-24에 논문 작성 및 제출하려고 합니다. 코드는 이미 준비해놨어요.

논의하고 싶은 점은:
1) NASE + CFG 조합이 충분한 novelty가 될 수 있을지?
2) 18일 안에 실험 + 논문 작성이 현실적인지?
3) 만약 CFG가 OOD에서도 효과가 없으면 Plan B를 어떻게 가져가야 할지?

내일 미팅에서 같이 얘기해보면 좋을 것 같습니다!
