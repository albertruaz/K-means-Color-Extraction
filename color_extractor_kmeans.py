import cv2
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from collections import Counter

def find_optimal_k(pixels_lab, max_k):
    sample_size = min(5000, len(pixels_lab))
    if len(pixels_lab) > sample_size:
        indices = np.random.choice(len(pixels_lab), sample_size, replace=False)
        pixels_sample = pixels_lab[indices]
    else:
        pixels_sample = pixels_lab
    silhouette_scores = {}
    for k in range(2, max_k + 1):
        try:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=3)
            labels = kmeans.fit_predict(pixels_sample)
            score = silhouette_score(pixels_sample, labels)
            silhouette_scores[k] = score
        except Exception as e:
            continue
    if not silhouette_scores:
        return 3
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    return best_k

def extract_dominant_colors_kmeans(image_path, max_colors=5):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image at {image_path}")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, w, _ = lab.shape
    pixels_lab = lab.reshape(-1, 3)
    optimal_k = find_optimal_k(pixels_lab, max_colors)
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=1024, n_init=3)
    labels = kmeans.fit_predict(pixels_lab)
    cluster_centers_lab = kmeans.cluster_centers_.astype(np.uint8)
    lab_centers = cluster_centers_lab.reshape(-1, 1, 3)
    bgr_centers = cv2.cvtColor(lab_centers, cv2.COLOR_LAB2BGR)
    rgb_centers = cv2.cvtColor(bgr_centers, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    cluster_centers_rgb = [tuple(c) for c in rgb_centers]
    label_counts = Counter(labels)
    total_pixels = len(labels)
    cluster_percentages = {label: count/total_pixels for label, count in label_counts.items()}
    sorted_clusters = sorted(cluster_percentages.items(), key=lambda x: x[1], reverse=True)
    selected_colors = []
    cumulative_coverage = 0.0
    for i, (cluster_label, percentage) in enumerate(sorted_clusters):
        if i < 4:
            selected_colors.append(cluster_centers_rgb[cluster_label])
            cumulative_coverage += percentage
    return selected_colors, cluster_percentages, cumulative_coverage, optimal_k 