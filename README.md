# Denoising Versus Classification

소아 흉부 X-ray에 합성 노이즈를 넣은 뒤, 노이즈를 줄여도 같은 분류기와 같은 문턱값의 정상, 폐렴 판정이 남는지를 보는 실험이다. 화질이 원본과 가까워졌는지는 PSNR, SSIM, 경계 상관으로 따로 적는다. 픽셀 점수가 더 높다는 이유로 방법을 고르지 않는다.

공개 데이터에 노이즈를 합성한 개인 학습 프로젝트이며, 특정 장비, 치과 X-ray, CT, 실제 저선량 촬영과 무관하다. 의사 판독은 하지 않았다.

## 질문

분류기는 깨끗한 학습 영상만 본다. 노이즈 영상이나 복원 영상으로 다시 학습하지 않는다. 공식 테스트에는 네 조건을 같은 가중치, 같은 문턱값으로 한 번씩 넣는다.

1. 깨끗함
2. 합성 노이즈. 포아송 기댓값 50 다음에 가우시안 표준편차 0.02
3. 그 노이즈에 가우시안 필터. 표준편차 1.0, 학습 없음
4. 그 노이즈에 잔차 합성곱망. 채널 32, 5층, 입력과 잔차의 합

문턱값은 깨끗한 검증에서 macro-F1이 가장 높은 값으로 고정한다. 이번 실행에서는 0.7이었다. 대표 점수는 macro-F1이다. 폐렴이 더 많아서 정확도만으로 좋아 보일 수 있으므로, 폐렴 미탐과 정상 오탐을 같이 본다.

## 데이터

[Kermany et al., Cell 2018](https://doi.org/10.1016/j.cell.2018.02.010)의 소아 흉부 X-ray이다. 1세에서 5세, 전후 방향, 정상과 폐렴이다. 이 저장소는 Mendeley 버전 2와 같은 [Kaggle 사본](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)을 쓴다. 라이선스는 CC BY 4.0이다. 버전 3과 OCT 폴더는 쓰지 않는다. 원본 영상은 포함하지 않는다.

받은 폴더는 학습 5,216장, 공식 테스트 624장, 공식 검증 16장이다. 공식 검증 16장은 개수 확인만 하고, 학습 폴더의 환자를 시드 42로 20퍼센트 나누어 검증에 쓴다. 공식 테스트는 문턱값을 저장한 뒤에 한 번만 연다. 공식 학습 폴더와 테스트의 환자 키는 170명 겹치며, 테스트 파일은 빼지 않고 그 수만 `outputs/counts.json`에 남긴다. 따라서 공식 테스트는 처음 보는 환자만의 평가가 아니다.

## 이번 실행의 결과

숫자는 `outputs/summary.md`에 스크립트가 쓴 값이다. 장치는 CPU이다.

| 조건 | 폐렴 미탐 | 정상 오탐 | macro-F1 | PSNR |
| --- | --- | --- | --- | --- |
| 깨끗함 | 4 | 107 | 0.7851 | 기준 |
| 노이즈 | 0 | 234 | 0.3846 | 19.4716 |
| 가우시안 | 0 | 234 | 0.3846 | 29.6843 |
| 잔차망 | 0 | 188 | 0.5672 | 33.0798 |

합성 노이즈는 폐렴을 숨기지 않고, 정상 234장을 모두 폐렴으로 밀었다. 가우시안 필터는 PSNR을 올렸으나 그 판정은 바꾸지 않았다. 잔차망은 픽셀 지표가 가장 높았고, 정상 오탐을 234장에서 188장으로 줄였다. macro-F1이 가장 높은 조건은 깨끗한 영상이고, PSNR이 가장 높은 조건은 잔차망이다.

## 실행

```bash
pip install -r requirements.txt
python data.py --search-root /path/to/chest_xray_parent --output-dir outputs
python train_classifier.py --output-dir outputs
python train_denoiser.py --output-dir outputs
python evaluate.py --output-dir outputs
```

Kaggle에서는 데이터셋 `paultimothymooney/chest-xray-pneumonia`를 붙이고 `--search-root /kaggle/input`으로 둔다. 노트북 `denoising-versus-classification.ipynb`는 같은 순서이다. 환자 키가 `outputs/filename_samples.txt`와 맞는지 확인하기 전에는 학습 셀로 넘어가지 않는다.

CUDA가 있으면 `cuda:0` 한 장을 쓰고, 없으면 CPU로 학습한다. 실험 상수는 `configs.py`에만 있다. `outputs/classifier.pt`와 `outputs/denoiser.pt`는 용량 때문에 올리지 않는다. 이미 있으면 학습을 건너뛰므로, 같은 시드로 처음부터 다시 돌릴 때만 그 파일을 지운다.
