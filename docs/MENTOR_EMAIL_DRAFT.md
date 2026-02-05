멘토님, 지금 연구하는거 결과가 잘 안나와서 잠시 실험을 멈추고 관련 연구들 및 방법론들, 그리고 논문에서 주장할 수 있는 유효한 contribution에 대해서 고민을 해보았는데요.

일단 현재 문제는, 기존에 CNN encoder + CFG로 학습했던 결과가 예상과 많이 달랐습니다. CFG를 적용한 모델은 오히려 baseline보다 더 안 좋게 나왔어요. OOD에서 개선되어야 하는데 오히려 악화됐습니다.

그래서 일단 실험을 잠시 중단하고 오늘 관련 noise conditioning을 적용한 기존 연구들을 찾아보았습니다. 그중 NASE가 SGMSE+에 noise conditioning을 적용한 연구라 해당 방법이 어떻게 설계를 했는지 비교해보았는데,

1) Noise encoder가 너무 약함 - NASE는 pretrained BEATs(768-dim)를 쓰는데, 저희는 scratch CNN(512-dim)을 썼음
2) Encoder supervision이 없음 - NASE는 NC loss로 encoder가 noise type을 구분하도록 강제하는데, 저희는 그게 없었음
3) CFG dropout이 오히려 방해 - 이미 약한 encoder에서 20%를 drop하니까 학습 신호가 더 약해진 것 같음

그래서 NASE의 강점(BEATs encoder + NC loss)을 가져오되, CFG로 OOD graceful degradation을 추가하는 방향으로 pivot하려고 합니다.

NASE와의 차이점은, NASE는 OOD noise에서 misleading conditioning 문제가 있거든요. 잘못된 embedding이 들어가면 오히려 성능이 하락하는데, 저희는 CFG를 통해 unreliable한 conditioning일 때 unconditional로 fallback할 수 있습니다.

그리고 DiTSE(2025)라는 논문도 찾아봤는데, 얘네도 CFG를 SE에 적용하긴 했어요. 근데 목적이 "conditioning을 더 잘 활용하게" 하는거고 OOD 상황은 고려를 안해서, 저희가 OOD graceful degradation을 contribution으로 잡으면 차별화가 될 것 같습니다.

그래서 contribution을 다시 정리해보면, 1) CFG를 통한 OOD graceful degradation, 2) inference time에 guidance scale 조절로 conditioning 강도 제어, 3) conditioning이 언제 도움되고 언제 해가 되는지 분석 정도로 잡으면 될 것 같아요.

일정은 제출까지 D-18인데, 이번주에 NASE baseline이랑 저희 모델 학습하고, 다음주에 실험 돌리고, 마지막 주에 논문 작성하면 될 것 같습니다. 코드는 오늘 미리 준비해놨어요.

근데 멘토님이랑 논의하고 싶은게, NASE + CFG 조합이 충분한 novelty가 될 수 있을지, 그리고 만약 CFG가 OOD에서도 효과가 없으면 어떻게 해야할지가 좀 고민이에요.

내일 미팅에서 같이 얘기해보면 좋을 것 같습니다!
