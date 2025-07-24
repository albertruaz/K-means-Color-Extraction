# Color Extractor

이미지에서 주요 색상을 추출하는 Python 애플리케이션입니다.

## 주요 기능

### K-means 기반 색상 추출 (color_extractor_kmeans.py)

- **LAB 색 공간**을 사용하여 인간이 인식하는 색상 차이를 정확하게 반영
- **최적의 클러스터 수 자동 결정** (Silhouette Score + Elbow Method)
- **상위 3개 색상** 자동 선택
- **이미지 리사이징** (최대 300px)으로 처리 속도 향상
- **시각화 결과**를 `results/` 폴더에 저장

### 색상 분류 기반 추출 (color_classification.py)

- 사전 정의된 색상 팔레트와 비교하여 가장 유사한 색상 찾기
- XKCD 색상 (949개), CSS4 색상 (148개) 지원
- 상위 3개 색상 선택

## 설치 방법

### 1. Conda 환경 생성

```bash
conda create -n color-extraction python=3.8
conda activate color-extraction
```

### 2. 필요한 패키지 설치

```bash
pip install opencv-contrib-python scikit-learn numpy matplotlib
```

## 사용 방법

### 기본 실행

```bash
python main.py
```

### 개별 파일 실행

```bash
# K-means 기반 색상 추출
python color_extractor_kmeans.py

# 색상 분류 기반 추출
python color_classification.py
```

## 파일 구조

```
color-extractor/
├── main.py                    # 메인 실행 파일
├── color_extractor_kmeans.py  # K-means 기반 색상 추출
├── color_classification.py    # 색상 분류 기반 추출
├── color_list.json           # 색상 정의 파일
├── images/                   # 입력 이미지 폴더
├── results/                  # 결과 저장 폴더
└── README.md
```

## 입력 이미지 형식

지원하는 이미지 형식:

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`

## 출력 결과

### K-means 방법

- **최적 클러스터 수**: 자동으로 결정된 k 값
- **주요 색상**: 상위 3개 색상 (RGB 값)
- **색상 비율**: 각 색상이 차지하는 픽셀 비율
- **시각화**: 원본 이미지와 추출된 색상 팔레트

### 색상 분류 방법

- **색상 이름**: 가장 유사한 사전 정의 색상명
- **RGB 값**: 해당 색상의 RGB 값
- **색상 비율**: 각 색상이 차지하는 픽셀 비율

## 성능 최적화

- **이미지 리사이징**: 300px 이하로 자동 조정
- **배치 처리**: 대용량 이미지 처리 시 메모리 효율성
- **벡터화 연산**: NumPy를 활용한 고속 색상 분류

## 그 외

- 필요시 배경제거 모델 도입 제안
- 최대 3개의 색상 추출
