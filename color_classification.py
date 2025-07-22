import cv2
import numpy as np
import json
import time
from collections import Counter
from sklearn.metrics import silhouette_score

def load_color_dictionary(json_path='color_list.json'):
    """
    Load color dictionary from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing color definitions
    
    Returns:
        dict: Dictionary mapping color names to RGB values
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            color_dict = json.load(f)
        
        # Convert hex colors to RGB tuples
        rgb_dict = {}
        for name, hex_color in color_dict.items():
            if hex_color.startswith('#'):
                # Convert hex to RGB
                hex_color = hex_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgb_dict[name] = rgb
        
        print(f"로드된 색상 수: {len(rgb_dict)}개")
        return rgb_dict
    except Exception as e:
        print(f"색상 딕셔너리 로드 실패: {e}")
        return {}

def find_closest_color(pixel_rgb, color_dict):
    """
    Find the closest color from the dictionary for a given pixel.
    
    Args:
        pixel_rgb (tuple): RGB values of the pixel
        color_dict (dict): Dictionary of color names to RGB values
    
    Returns:
        str: Name of the closest color
    """
    min_distance = float('inf')
    closest_color = None
    
    for color_name, color_rgb in color_dict.items():
        # Calculate Euclidean distance in RGB space
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pixel_rgb, color_rgb)))
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

def extract_dominant_colors_classification(image_path, max_colors=4, coverage_threshold=0.8):
    """
    Extract dominant colors from an image using color classification method.
    
    Args:
        image_path (str): Path to the input image file.
        max_colors (int): Maximum number of colors to extract (default: 4).
        coverage_threshold (float): Threshold for cumulative coverage (default: 0.8).
    
    Returns:
        List of RGB tuples representing dominant colors.
    """
    total_start_time = time.time()
    print(f"이미지 로딩 중: {image_path}")
    
    # Load color dictionary
    load_start_time = time.time()
    color_dict = load_color_dictionary()
    if not color_dict:
        raise ValueError("색상 딕셔너리를 로드할 수 없습니다.")
    
    load_time = time.time() - load_start_time
    print(f"색상 딕셔너리 로드 시간: {load_time:.2f}초")
    
    # Load and resize image
    img_start_time = time.time()
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    original_size = bgr.shape
    print(f"원본 이미지 크기: {original_size}")
    
    # Resize image if it's too large to reduce processing time
    h, w = bgr.shape[:2]
    max_size = 300
    if max(h, w) > max_size:
        # Calculate new dimensions maintaining aspect ratio
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"이미지 리사이즈: {original_size} → {bgr.shape}")
    else:
        print(f"이미지 크기가 적절함 (최대 {max_size}px): {bgr.shape}")
    
    print(f"처리할 이미지 크기: {bgr.shape}")
    
    # Convert to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Reshape image to 2D array of pixels
    h, w, _ = rgb.shape
    pixels_rgb = rgb.reshape(-1, 3)
    
    print(f"픽셀 데이터 크기: {pixels_rgb.shape}")
    img_load_time = time.time() - img_start_time
    print(f"이미지 로딩 및 변환 시간: {img_load_time:.2f}초")
    
    # Classify each pixel to the closest color
    classification_start_time = time.time()
    print("픽셀 색상 분류 중...")
    
    color_counts = Counter()
    total_pixels = len(pixels_rgb)
    
    # Process pixels in batches to avoid memory issues
    batch_size = 10000
    for i in range(0, total_pixels, batch_size):
        batch = pixels_rgb[i:i+batch_size]
        for pixel in batch:
            closest_color = find_closest_color(tuple(pixel), color_dict)
            color_counts[closest_color] += 1
    
    classification_time = time.time() - classification_start_time
    print(f"색상 분류 시간: {classification_time:.2f}초")
    
    # Get the most frequent colors
    most_common_colors = color_counts.most_common()
    
    # Calculate percentages and select colors
    selected_colors = []
    cumulative_coverage = 0.0
    
    for color_name, count in most_common_colors:
        percentage = count / total_pixels
        if percentage >= 0.03:  # 3% 이상인 경우만 포함
            selected_colors.append(color_dict[color_name])
            cumulative_coverage += percentage
            print(f"선택된 색상: {color_name} - RGB{color_dict[color_name]} ({percentage*100:.1f}%)")
    
    print(f"선택된 색상 수: {len(selected_colors)}")
    print(f"총 커버리지: {cumulative_coverage*100:.1f}%")
    
    # Create a dummy percentages dict for compatibility
    percentages = {i: count/total_pixels for i, (color_name, count) in enumerate(most_common_colors)}
    
    total_time = time.time() - total_start_time
    print(f"전체 색상 분류 시간: {total_time:.2f}초")
    
    # Return dummy optimal_k for compatibility
    optimal_k = len(selected_colors) if selected_colors else 3
    
    return selected_colors, percentages, cumulative_coverage, optimal_k

def find_optimal_k(pixels_lab, max_k=5):
    """
    Dummy function for compatibility with the existing interface.
    This method doesn't use k-means clustering.
    
    Args:
        pixels_lab (np.array): Lab color space pixels (not used)
        max_k (int): Maximum k to test (not used)
    
    Returns:
        int: Default k value
    """
    return 3 