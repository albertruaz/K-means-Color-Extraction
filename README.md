# Color Extractor (색상 추출기)

이미지에서 주요 색상을 추출하는 파이썬 프로그램입니다.

## 환경 설정

### Conda 환경 생성 및 활성화

```bash
# conda 환경 생성
conda create -n color-extraction python=3.8
conda activate color-extraction
```

### 필요한 패키지 설치

```bash
pip install opencv-contrib-python
pip install scikit-learn
pip install numpy
pip install matplotlib
```

## 사용 방법

### 1. 기존 색상 추출기 (dominant_color_extractor.py)

```bash
python dominant_color_extractor.py
```

### 2. K-means 기반 색상 추출기 (color_extractor_kmeans.py)

```bash
python color_extractor_kmeans.py
```

## 프로그램 설명

### dominant_color_extractor.py

- ICaS 변환과 saliency detection을 사용한 고급 색상 추출
- 이미지의 주요 영역에서 색상을 추출

### color_extractor_kmeans.py

- K-means 클러스터링을 사용한 색상 추출
- 최대 3개의 우세한 색상 추출
- 80% 커버리지 기준으로 색상 선택
- 시각화 기능 포함

## 출력 결과

프로그램은 이미지에서 추출한 주요 색상들을 RGB 형식으로 출력합니다.
출력 예시:

```
추출된 주요 색상 (RGB): [(255, 0, 0), (0, 255, 0), ...]
클러스터별 퍼센트:
클러스터 0: 45.2%
클러스터 1: 32.1%
클러스터 2: 22.7%
총 커버리지: 100.0%
```

## 주의사항

- 입력 이미지는 `image1.png`라는 이름으로 저장되어 있어야 합니다.
- 지원하는 이미지 형식: PNG, JPG, JPEG
- Python 3.8 이상 버전을 권장합니다.
# K-means-Color-Extraction
