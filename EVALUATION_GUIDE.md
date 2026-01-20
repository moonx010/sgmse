# Model Evaluation Guide

이 가이드는 SGMSE 모델을 다양한 테스트 데이터셋에서 평가하는 방법을 설명합니다.

## 기본 평가 방법 (단일 데이터셋)

### 1단계: Enhancement (음성 개선)

```bash
python enhancement.py \
    --test_dir /path/to/noisy/audio \
    --enhanced_dir ./enhanced_output \
    --ckpt checkpoints/model.ckpt \
    --N 30 \
    --corrector ald \
    --corrector_steps 1 \
    --snr 0.5
```

**주요 옵션:**
- `--test_dir`: Noisy 오디오 파일들이 있는 디렉토리 (`.wav` 또는 `.flac`)
- `--enhanced_dir`: 개선된 오디오를 저장할 출력 디렉토리
- `--ckpt`: 학습된 모델 체크포인트 경로
- `--N`: Reverse diffusion 스텝 수 (기본값: 30, 더 높으면 품질 향상하지만 느림)
- `--corrector`: Corrector 타입 (`ald`, `langevin`, `none`)
- `--corrector_steps`: Corrector 스텝 수 (기본값: 1)
- `--snr`: Langevin dynamics SNR 값 (기본값: 0.5)
- `--sampler_type`: Sampler 타입 (`pc` 또는 `ode`)

**특수 설정:**
- WSJ0-REVERB 모델의 경우: `--N 50 --snr 0.33` 사용 권장

### 2단계: Metrics 계산

```bash
python calc_metrics.py \
    --clean_dir /path/to/clean/audio \
    --noisy_dir /path/to/noisy/audio \
    --enhanced_dir ./enhanced_output
```

**계산되는 메트릭:**
- **PESQ** (1-4.5): 음성 품질 지각 평가
- **ESTOI** (0-1): 음성 명료도 측정
- **SI-SDR** (dB): 신호 대 왜곡 비율
- **SI-SIR** (dB): 신호 대 간섭 비율
- **SI-SAR** (dB): 신호 대 아티팩트 비율

**출력 파일:**
- `_avg_results.txt`: 평균 및 표준편차
- `_results.csv`: 파일별 상세 결과

## 다중 데이터셋 자동 평가

여러 데이터셋에 대해 자동으로 평가를 수행하는 스크립트를 제공합니다.

### 1. 설정 파일 준비

`datasets_config_example.yaml`을 복사하여 수정:

```bash
cp datasets_config_example.yaml my_datasets.yaml
```

설정 파일 예시:
```yaml
sampler:
  N: 30
  corrector: ald
  corrector_steps: 1
  snr: 0.5
  sampler_type: pc

datasets:
  - name: VoiceBank-DEMAND
    clean_dir: /data/voicebank/clean
    noisy_dir: /data/voicebank/noisy
    enhanced_dir: ./results/voicebank

  - name: WSJ0-CHiME3
    clean_dir: /data/wsj0_chime3/clean
    noisy_dir: /data/wsj0_chime3/noisy
    enhanced_dir: ./results/wsj0_chime3
```

### 2. 평가 실행

```bash
python evaluate_multiple_datasets.py \
    --config my_datasets.yaml \
    --ckpt checkpoints/model.ckpt \
    --results_dir ./evaluation_results
```

**옵션:**
- `--config`: 데이터셋 설정 파일 경로
- `--ckpt`: 모델 체크포인트 경로
- `--results_dir`: 요약 결과를 저장할 디렉토리
- `--skip_enhancement`: Enhancement 단계 건너뛰기 (이미 생성된 파일 사용)
- `--skip_metrics`: Metrics 계산 건너뛰기

### 3. 결과 확인

스크립트는 다음을 생성합니다:

1. **각 데이터셋별 결과:**
   - `enhanced_dir/_avg_results.txt`: 평균 결과
   - `enhanced_dir/_results.csv`: 상세 결과

2. **통합 요약:**
   - `results_dir/summary_YYYYMMDD_HHMMSS.csv`: 모든 데이터셋 비교 표

예시 요약:
```
Dataset            PESQ_mean  PESQ_std  ESTOI_mean  ESTOI_std  SI-SDR_mean  ...
VoiceBank-DEMAND      3.15      0.42       0.92       0.05        15.2
WSJ0-CHiME3           2.87      0.38       0.88       0.06        13.8
```

## 유용한 팁

### 1. GPU 메모리 절약
긴 오디오 파일이나 메모리가 부족한 경우:
```bash
# 더 적은 스텝 사용
python enhancement.py --N 20 ...

# 또는 ODE sampler 사용 (더 빠르지만 품질 약간 낮음)
python enhancement.py --sampler_type ode ...
```

### 2. 배치 처리
여러 체크포인트를 평가하는 경우:
```bash
#!/bin/bash
CHECKPOINTS=(
    "checkpoints/epoch_100.ckpt"
    "checkpoints/epoch_200.ckpt"
    "checkpoints/epoch_300.ckpt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "Evaluating $ckpt"
    python evaluate_multiple_datasets.py \
        --config my_datasets.yaml \
        --ckpt "$ckpt" \
        --results_dir "results/$(basename $ckpt .ckpt)"
done
```

### 3. 이미 생성된 Enhanced 파일 재사용
Enhancement는 시간이 오래 걸리므로, 이미 생성된 파일로 metrics만 재계산:
```bash
python evaluate_multiple_datasets.py \
    --config my_datasets.yaml \
    --ckpt dummy.ckpt \
    --skip_enhancement
```

### 4. 데이터셋 디렉토리 구조
모델이 기대하는 디렉토리 구조:

```
dataset/
├── clean/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── noisy/
    ├── file1.wav
    ├── file2.wav
    └── ...
```

또는 서브디렉토리 포함:
```
dataset/
├── clean/
│   ├── subset1/
│   │   ├── file1.wav
│   │   └── file2.wav
│   └── subset2/
│       └── file3.wav
└── noisy/
    └── (같은 구조)
```

**주의사항:**
- Clean과 noisy 디렉토리에 동일한 파일명이 있어야 함
- `.wav` 또는 `.flac` 포맷 지원
- Sampling rate가 다르면 자동으로 리샘플링됨 (16kHz 또는 48kHz)

## 문제 해결

### PESQ 계산 오류
PESQ는 16kHz에서만 작동하므로, 코드가 자동으로 리샘플링합니다.

### 파일명 불일치
Noisy 파일에 SNR 정보가 포함된 경우 (예: `p232_001_-5dB.wav`),
`calc_metrics.py`는 자동으로 clean 파일 (`p232_001.wav`)을 찾습니다.

### Out of Memory
- `--N` 값을 줄이기 (30 → 20)
- `--sampler_type ode` 사용
- 더 작은 배치로 파일 나누기

### 느린 처리 속도
- GPU 사용 확인: `--device cuda`
- 더 적은 스텝 사용: `--N 20`
- ODE sampler 사용: `--sampler_type ode`
- Corrector 비활성화: `--corrector none`

## 모델별 권장 설정

### SGMSE+ (VoiceBank-DEMAND / WSJ0-CHiME3)
```bash
--N 30 --corrector ald --corrector_steps 1 --snr 0.5
```

### WSJ0-REVERB Dereverberation
```bash
--N 50 --corrector ald --corrector_steps 1 --snr 0.33
```

### 48kHz Models (EARS dataset)
```bash
--N 30 --corrector ald --corrector_steps 1 --snr 0.5
# 모델이 자동으로 48kHz 처리
```

### Schrödinger Bridge Models
```bash
--sampler_type ode --N 30
# SB 모델은 자동으로 감지되어 적절한 sampler 사용
```

## 참고 자료

- 메인 README: `README.md`
- Claude 참고 문서: `claude.md`
- 예시 설정 파일: `datasets_config_example.yaml`
