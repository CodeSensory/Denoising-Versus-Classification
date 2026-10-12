# 노이즈를 줄여도 흉부 X-ray 분류가 남는가

소아 흉부 X-ray에 합성 노이즈를 넣고, 가우시안 필터와 잔차망이 같은 문턱값의 정상, 폐렴 판정을 얼마나 되돌리는지 보는 코드다. 실험 상수는 `configs.py`에만 둔다. 테스트 점수를 보고 노이즈 세기, 학습률, 문턱값을 바꾸지 않는다. 분류기는 깨끗한 학습 영상만 보고, 네 조건은 같은 가중치와 같은 문턱값을 쓴다.

데이터는 포함하지 않는다. Kaggle 데이터셋 `paultimothymooney/chest-xray-pneumonia`를 붙이거나, 로컬에서는 그 버전 2의 흉부 X-ray 폴더 경로를 넘긴다. OCT는 읽지 않는다. 원본 영상과 `outputs/*.pt`는 Git에 올리지 않는다.

CUDA가 있으면 `cuda:0` 한 장을 쓰고, 없으면 중앙처리 장치로 학습한다. 두 장을 데이터 병렬로 묶지 않는다.

## 실행

하루 경계는 노트북 `denoising-versus-classification.ipynb`와 같다. 1일 차 확인 셀 뒤의 오류는 `split.json`의 환자 규칙 문장을 읽은 뒤에만 지운다. 전부 실행은 쓰지 않는다.

```bash
python data.py --search-root /kaggle/input --output-dir /kaggle/working/outputs
python train_classifier.py --output-dir /kaggle/working/outputs
python train_denoiser.py --output-dir /kaggle/working/outputs
python evaluate.py --output-dir /kaggle/working/outputs
```

로컬이면 `--output-dir outputs`로 바꾼다. `classifier.pt`가 이미 있으면 분류기 학습을 건너뛰고 문턱값으로 간다. `denoiser.pt`가 있으면 잔차망 학습을 건너뛴다. 같은 코드와 같은 시드로 처음부터 다시 돌릴 때만 그 파일을 지운다.

Windows에서는 데이터 로더 프로세스 수를 0으로 둔다. Kaggle에서는 2다.

슬라이드 숫자는 `outputs/summary.md`에 스크립트가 쓴 값만 옮긴다.
