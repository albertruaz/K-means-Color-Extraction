import cv2
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

def find_optimal_k(pixels_lab, max_k=5):
    """
    Find optimal k using silhouette score and elbow method.
    
    Args:
        pixels_lab (np.array): Lab color space pixels
        max_k (int): Maximum k to test
    
    Returns:
        int: Optimal k value
    """
    start_time = time.time()
    print("최적 k 탐색 중...")
    
    # Sample data if too large to prevent memory issues
    if len(pixels_lab) > 10000:
        print(f"데이터가 너무 큽니다 ({len(pixels_lab)} 픽셀). 샘플링 중...")
        indices = np.random.choice(len(pixels_lab), 10000, replace=False)
        pixels_sample = pixels_lab[indices]
    else:
        pixels_sample = pixels_lab
    
    print(f"사용할 픽셀 수: {len(pixels_sample)}")
    
    # Silhouette score method
    silhouette_scores = {}
    inertias = {}
    
    for k in range(2, max_k + 1):
        k_start_time = time.time()
        print(f"k={k} 테스트 중...")
        try:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=5)
            labels = kmeans.fit_predict(pixels_sample)
            
            # Calculate silhouette score with error handling
            try:
                sil_score = silhouette_score(pixels_sample, labels)
                silhouette_scores[k] = sil_score
                print(f"  실루엣 점수: {sil_score:.3f}")
            except Exception as sil_error:
                print(f"  실루엣 점수 계산 실패: {sil_error}")
                # Use inertia as fallback
                silhouette_scores[k] = -kmeans.inertia_  # Negative for minimization
            
            inertias[k] = kmeans.inertia_
            print(f"  관성: {kmeans.inertia_:.0f}")
            
        except Exception as e:
            print(f"k={k}에서 오류 발생: {e}")
            continue
        
        k_time = time.time() - k_start_time
        print(f"  k={k} 처리 시간: {k_time:.2f}초")
    
    if not silhouette_scores:
        print("실루엣 점수 계산 실패, 기본값 k=3 사용")
        return 3
    
    # Find best k by silhouette score
    best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
    
    # Elbow method (find the elbow point)
    k_values = list(inertias.keys())
    inertia_values = list(inertias.values())
    
    if len(inertia_values) > 1:
        # Calculate the rate of change
        changes = []
        for i in range(1, len(inertia_values)):
            change = (inertia_values[i-1] - inertia_values[i]) / inertia_values[i-1]
            changes.append(change)
        
        # Find the elbow point (where the rate of change drops significantly)
        elbow_k = k_values[changes.index(max(changes)) + 1]
    else:
        elbow_k = best_k_silhouette
    
    print(f"Silhouette 최적 k: {best_k_silhouette} (점수: {silhouette_scores[best_k_silhouette]:.3f})")
    print(f"Elbow 최적 k: {elbow_k}")
    
    # Use silhouette score as primary method, elbow as fallback
    optimal_k = best_k_silhouette
    
    total_time = time.time() - start_time
    print(f"최적 k 탐색 완료: {total_time:.2f}초")
    
    return optimal_k

def extract_dominant_colors_kmeans(image_path, max_colors=4, coverage_threshold=0.8):
    """
    Extract dominant colors from an image using K-means clustering in CIELAB space.
    
    Args:
        image_path (str): Path to the input image file.
        max_colors (int): Maximum number of colors to extract (default: 4).
        coverage_threshold (float): Threshold for cumulative coverage (default: 0.8).
    
    Returns:
        List of RGB tuples representing dominant colors.
    """
    total_start_time = time.time()
    print(f"이미지 로딩 중: {image_path}")
    
    # Load image and convert to LAB color space
    load_start_time = time.time()
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    print(f"이미지 크기: {bgr.shape}")
    
    # Convert to LAB color space for better clustering
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    # Reshape image to 2D array of pixels
    h, w, _ = lab.shape
    pixels_lab = lab.reshape(-1, 3)
    
    print(f"픽셀 데이터 크기: {pixels_lab.shape}")
    load_time = time.time() - load_start_time
    print(f"이미지 로딩 및 변환 시간: {load_time:.2f}초")
    
    # Find optimal k using silhouette score and elbow method
    k_search_start_time = time.time()
    optimal_k = find_optimal_k(pixels_lab, max_colors)
    k_search_time = time.time() - k_search_start_time
    print(f"최적 k 탐색 시간: {k_search_time:.2f}초")
    
    print(f"K-means 클러스터링 시작 (k={optimal_k})")
    clustering_start_time = time.time()
    
    # Apply K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels_lab)
    
    # Get cluster centers in LAB space
    cluster_centers_lab = kmeans.cluster_centers_.astype(int)
    
    clustering_time = time.time() - clustering_start_time
    print(f"K-means 클러스터링 시간: {clustering_time:.2f}초")
    
    print("LAB 색 공간에서 RGB로 변환 중...")
    conversion_start_time = time.time()
    
    # Convert LAB centers back to RGB
    cluster_centers_rgb = []
    for i, center in enumerate(cluster_centers_lab):
        try:
            # Reshape for cv2 conversion
            center_lab = center.reshape(1, 1, 3).astype(np.uint8)
            center_bgr = cv2.cvtColor(center_lab, cv2.COLOR_LAB2BGR)
            center_rgb = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2RGB)
            cluster_centers_rgb.append(tuple(center_rgb[0, 0]))
            print(f"클러스터 {i} 색상: {tuple(center_rgb[0, 0])}")
        except Exception as e:
            print(f"클러스터 {i} 색상 변환 오류: {e}")
            # Fallback: use original LAB values as RGB
            cluster_centers_rgb.append(tuple(center))
    
    conversion_time = time.time() - conversion_start_time
    print(f"색 공간 변환 시간: {conversion_time:.2f}초")
    
    # Calculate the percentage of pixels in each cluster
    label_counts = Counter(labels)
    total_pixels = len(labels)
    cluster_percentages = {label: count/total_pixels for label, count in label_counts.items()}
    
    # Sort clusters by percentage (most dominant first)
    sorted_clusters = sorted(cluster_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Select colors until we reach the coverage threshold
    selected_colors = []
    cumulative_coverage = 0.0
    
    # 모든 클러스터의 색상을 선택 (80% 기준 비활성화)
    # 5% 이상인 색상만 포함
    for cluster_label, percentage in sorted_clusters:
        if percentage >= 0.03:  # 5% 이상인 경우만 포함
            selected_colors.append(cluster_centers_rgb[cluster_label])
            cumulative_coverage += percentage
    
    print(f"선택된 색상 수: {len(selected_colors)}")
    print(f"총 커버리지: {cumulative_coverage*100:.1f}%")
    
    total_time = time.time() - total_start_time
    print(f"전체 색상 추출 시간: {total_time:.2f}초")
    
    return selected_colors, cluster_percentages, cumulative_coverage, optimal_k 