# 평가 요약

숫자는 이 스크립트가 계산했다.

## clean
혼동행렬 [정상, 폐렴] 순: [[127, 107], [4, 386]]
폐렴 미탐 4장, 정상 오탐 107장
macro-F1 0.7851

## noise
혼동행렬 [정상, 폐렴] 순: [[0, 234], [0, 390]]
폐렴 미탐 0장, 정상 오탐 234장
macro-F1 0.3846

## gaussian
혼동행렬 [정상, 폐렴] 순: [[0, 234], [0, 390]]
폐렴 미탐 0장, 정상 오탐 234장
macro-F1 0.3846

## residual
혼동행렬 [정상, 폐렴] 순: [[46, 188], [0, 390]]
폐렴 미탐 0장, 정상 오탐 188장
macro-F1 0.5672

## 정확도 각주
clean 정확도 0.8221. 대표 점수는 macro-F1이다.
noise 정확도 0.6250. 대표 점수는 macro-F1이다.
gaussian 정확도 0.6250. 대표 점수는 macro-F1이다.
residual 정확도 0.6987. 대표 점수는 macro-F1이다.

## 픽셀
noise PSNR 19.4716 완전 일치 0장 SSIM 0.2330 경계 상관 0.1757 경계를 뺀 장수 0
gaussian PSNR 29.6843 완전 일치 0장 SSIM 0.7048 경계 상관 0.5045 경계를 뺀 장수 0
residual PSNR 33.0798 완전 일치 0장 SSIM 0.8679 경계 상관 0.7557 경계를 뺀 장수 0

macro-F1이 가장 높은 조건은 clean이다. PSNR이 가장 높은 조건은 residual이다. 둘은 다르다.

## 추론 시간
classifier 이미지당 44.035 ms, 기기 cpu
gaussian 이미지당 0.607 ms, 기기 cpu
residual 이미지당 28.009 ms, 기기 cpu

## 예시
test_correct.png: test/PNEUMONIA/person100_bacteria_475.jpeg
test_recovered.png: 해당 유형 없음
test_still_missed.png: 해당 유형 없음
test_false_positive.png: test/NORMAL/IM-0001-0001.jpeg

검증에서 잔차망 출력과 노이즈 입력의 평균 절대 오차 0.082295
