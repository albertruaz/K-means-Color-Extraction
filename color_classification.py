import cv2
import numpy as np
import json
import time
from collections import Counter
from sklearn.metrics import silhouette_score
from matplotlib.colors import XKCD_COLORS

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

def load_xkcd_color_dictionary():
    """
    Load XKCD color dictionary from matplotlib.
    
    Returns:
        dict: Dictionary mapping color names to RGB values
    """
    try:
        # Convert XKCD colors to RGB format
        xkcd_dict = {}
        for name, hex_color in XKCD_COLORS.items():
            # Remove 'xkcd:' prefix and convert hex to RGB
            clean_name = name.replace('xkcd:', '')
            # Convert hex to RGB (hex_color format: '#RRGGBB')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
            xkcd_dict[clean_name] = rgb
        
        print(f"XKCD 색상 로드됨: {len(xkcd_dict)}개")
        return xkcd_dict
    except Exception as e:
        print(f"XKCD 색상 딕셔너리 로드 실패: {e}")
        return {}

def find_closest_color_batch(pixels_rgb, color_dict):
    """
    Find the closest colors for a batch of pixels using vectorized operations.
    
    Args:
        pixels_rgb (np.array): Array of RGB pixel values (N, 3)
        color_dict (dict): Dictionary of color names to RGB values
    
    Returns:
        list: List of closest color names for each pixel
    """
    color_names = list(color_dict.keys())
    color_values = np.array(list(color_dict.values()))  # shape (C, 3)
    
    # Vectorized distance calculation using broadcasting
    # pixels_rgb (N,3), color_values (C,3) → (N,C,3)
    diff = pixels_rgb[:, None, :] - color_values[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # shape (N, C)
    closest_indices = np.argmin(dists, axis=1)
    
    return [color_names[idx] for idx in closest_indices]

def extract_dominant_colors_classification_XKCD(image_path, max_colors=4, coverage_threshold=0.8):
    """
    Extract dominant colors from an image using XKCD color classification method.
    
    Args:
        image_path (str): Path to the input image file.
        max_colors (int): Maximum number of colors to extract (default: 4).
        coverage_threshold (float): Threshold for cumulative coverage (default: 0.8).
    
    Returns:
        List of RGB tuples representing dominant colors.
    """
    total_start_time = time.time()
    print(f"이미지 로딩 중: {image_path}")
    
    # Load XKCD color dictionary
    load_start_time = time.time()
    color_dict = load_xkcd_color_dictionary()
    if not color_dict:
        raise ValueError("XKCD 색상 딕셔너리를 로드할 수 없습니다.")
    
    load_time = time.time() - load_start_time
    print(f"XKCD 색상 딕셔너리 로드 시간: {load_time:.2f}초")
    
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
    print("픽셀 색상 분류 중 (XKCD 색상 기준)...")
    
    color_counts = Counter()
    total_pixels = len(pixels_rgb)
    
    # Process pixels in batches to avoid memory issues
    batch_size = 10000
    for i in range(0, total_pixels, batch_size):
        batch = pixels_rgb[i:i+batch_size]
        closest_colors_batch = find_closest_color_batch(batch, color_dict)
        for color_name in closest_colors_batch:
            color_counts[color_name] += 1
    
    classification_time = time.time() - classification_start_time
    print(f"색상 분류 시간: {classification_time:.2f}초")
    
    # Get the most frequent colors
    most_common_colors = color_counts.most_common()
    
    selected_colors = []
    cumulative_coverage = 0.0
    
    for i, (color_name, count) in enumerate(most_common_colors):
        percentage = count / total_pixels
        if i < 3:  # 상위 3개만 선택
            selected_colors.append(color_dict[color_name])
            cumulative_coverage += percentage
            print(f"선택된 색상: {color_name} - RGB{color_dict[color_name]} ({percentage*100:.1f}%)")
    
    print(f"선택된 색상 수: {len(selected_colors)}")
    print(f"총 커버리지: {cumulative_coverage*100:.1f}%")
    
    percentages = {i: count/total_pixels for i, (color_name, count) in enumerate(most_common_colors)}
    
    total_time = time.time() - total_start_time
    print(f"전체 XKCD 색상 분류 시간: {total_time:.2f}초")
    
    optimal_k = len(selected_colors) if selected_colors else 3
    
    return selected_colors, percentages, cumulative_coverage, optimal_k

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
        closest_colors_batch = find_closest_color_batch(batch, color_dict)
        for color_name in closest_colors_batch:
            color_counts[color_name] += 1
    
    classification_time = time.time() - classification_start_time
    print(f"색상 분류 시간: {classification_time:.2f}초")
    
    # Get the most frequent colors
    most_common_colors = color_counts.most_common()
    
    # Calculate percentages and select colors
    selected_colors = []
    cumulative_coverage = 0.0
    
    for i, (color_name, count) in enumerate(most_common_colors):
        percentage = count / total_pixels
        if i < 3:  # 상위 3개만 선택
            selected_colors.append(color_dict[color_name])
            cumulative_coverage += percentage
            print(f"선택된 색상: {color_name} - RGB{color_dict[color_name]} ({percentage*100:.1f}%)")
    
    print(f"선택된 색상 수: {len(selected_colors)}")
    print(f"총 커버리지: {cumulative_coverage*100:.1f}%")
    
    percentages = {i: count/total_pixels for i, (color_name, count) in enumerate(most_common_colors)}
    
    total_time = time.time() - total_start_time
    print(f"전체 색상 분류 시간: {total_time:.2f}초")
    
    optimal_k = len(selected_colors) if selected_colors else 3
    
    return selected_colors, percentages, cumulative_coverage, optimal_k

def extract_dominant_colors_classification_CSS4(image_path, max_colors=4, coverage_threshold=0.99):
    """
    Extract dominant colors from an image using CSS4 color classification method.
    
    Args:
        image_path (str): Path to the input image file.
        max_colors (int): Maximum number of colors to extract (default: 4).
        coverage_threshold (float): Threshold for cumulative coverage (default: 0.8).
    
    Returns:
        List of RGB tuples representing dominant colors.
    """
    total_start_time = time.time()
    print(f"이미지 로딩 중: {image_path}")
    
    # Load CSS4 color dictionary
    load_start_time = time.time()
    color_dict = load_css4_color_dictionary()
    if not color_dict:
        raise ValueError("CSS4 색상 딕셔너리를 로드할 수 없습니다.")
    
    load_time = time.time() - load_start_time
    print(f"CSS4 색상 딕셔너리 로드 시간: {load_time:.2f}초")
    
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
    print("픽셀 색상 분류 중 (CSS4 색상 기준)...")
    
    color_counts = Counter()
    total_pixels = len(pixels_rgb)
    
    # Process pixels in batches to avoid memory issues
    batch_size = 10000
    for i in range(0, total_pixels, batch_size):
        batch = pixels_rgb[i:i+batch_size]
        closest_colors_batch = find_closest_color_batch(batch, color_dict)
        for color_name in closest_colors_batch:
            color_counts[color_name] += 1
    
    classification_time = time.time() - classification_start_time
    print(f"색상 분류 시간: {classification_time:.2f}초")
    
    # Get the most frequent colors
    most_common_colors = color_counts.most_common()
    
    # Calculate percentages and select colors
    selected_colors = []
    cumulative_coverage = 0.0
    
    for color_name, count in most_common_colors:
        percentage = count / total_pixels
        if i < 3:  # 상위 3개만 선택
            selected_colors.append(color_dict[color_name])
            cumulative_coverage += percentage
            print(f"선택된 색상: {color_name} - RGB{color_dict[color_name]} ({percentage*100:.1f}%)")
    
    print(f"선택된 색상 수: {len(selected_colors)}")
    print(f"총 커버리지: {cumulative_coverage*100:.1f}%")
    
    percentages = {i: count/total_pixels for i, (color_name, count) in enumerate(most_common_colors)}
    
    total_time = time.time() - total_start_time
    print(f"전체 CSS4 색상 분류 시간: {total_time:.2f}초")
    
    optimal_k = len(selected_colors) if selected_colors else 3
    
    return selected_colors, percentages, cumulative_coverage, optimal_k

def load_css4_color_dictionary():
    """
    Load CSS4 color dictionary.
    
    Returns:
        dict: Dictionary mapping color names to RGB values
    """
    try:
        # CSS4 color definitions
        css4_colors = {
            'aliceblue': (240, 248, 255),
            'antiquewhite': (250, 235, 215),
            'aqua': (0, 255, 255),
            'aquamarine': (127, 255, 212),
            'azure': (240, 255, 255),
            'beige': (245, 245, 220),
            'bisque': (255, 228, 196),
            'black': (0, 0, 0),
            'blanchedalmond': (255, 235, 205),
            'blue': (0, 0, 255),
            'blueviolet': (138, 43, 226),
            'brown': (165, 42, 42),
            'burlywood': (222, 184, 135),
            'cadetblue': (95, 158, 160),
            'chartreuse': (127, 255, 0),
            'chocolate': (210, 105, 30),
            'coral': (255, 127, 80),
            'cornflowerblue': (100, 149, 237),
            'cornsilk': (255, 248, 220),
            'crimson': (220, 20, 60),
            'cyan': (0, 255, 255),
            'darkblue': (0, 0, 139),
            'darkcyan': (0, 139, 139),
            'darkgoldenrod': (184, 134, 11),
            'darkgray': (169, 169, 169),
            'darkgreen': (0, 100, 0),
            'darkgrey': (169, 169, 169),
            'darkkhaki': (189, 183, 107),
            'darkmagenta': (139, 0, 139),
            'darkolivegreen': (85, 107, 47),
            'darkorange': (255, 140, 0),
            'darkorchid': (153, 50, 204),
            'darkred': (139, 0, 0),
            'darksalmon': (233, 150, 122),
            'darkseagreen': (143, 188, 143),
            'darkslateblue': (72, 61, 139),
            'darkslategray': (47, 79, 79),
            'darkslategrey': (47, 79, 79),
            'darkturquoise': (0, 206, 209),
            'darkviolet': (148, 0, 211),
            'deeppink': (255, 20, 147),
            'deepskyblue': (0, 191, 255),
            'dimgray': (105, 105, 105),
            'dimgrey': (105, 105, 105),
            'dodgerblue': (30, 144, 255),
            'firebrick': (178, 34, 34),
            'floralwhite': (255, 250, 240),
            'forestgreen': (34, 139, 34),
            'fuchsia': (255, 0, 255),
            'gainsboro': (220, 220, 220),
            'ghostwhite': (248, 248, 255),
            'gold': (255, 215, 0),
            'goldenrod': (218, 165, 32),
            'gray': (128, 128, 128),
            'green': (0, 128, 0),
            'greenyellow': (173, 255, 47),
            'grey': (128, 128, 128),
            'honeydew': (240, 255, 240),
            'hotpink': (255, 105, 180),
            'indianred': (205, 92, 92),
            'indigo': (75, 0, 130),
            'ivory': (255, 255, 240),
            'khaki': (240, 230, 140),
            'lavender': (230, 230, 250),
            'lavenderblush': (255, 240, 245),
            'lawngreen': (124, 252, 0),
            'lemonchiffon': (255, 250, 205),
            'lightblue': (173, 216, 230),
            'lightcoral': (240, 128, 128),
            'lightcyan': (224, 255, 255),
            'lightgoldenrodyellow': (250, 250, 210),
            'lightgray': (211, 211, 211),
            'lightgreen': (144, 238, 144),
            'lightgrey': (211, 211, 211),
            'lightpink': (255, 182, 193),
            'lightsalmon': (255, 160, 122),
            'lightseagreen': (32, 178, 170),
            'lightskyblue': (135, 206, 250),
            'lightslategray': (119, 136, 153),
            'lightslategrey': (119, 136, 153),
            'lightsteelblue': (176, 196, 222),
            'lightyellow': (255, 255, 224),
            'lime': (0, 255, 0),
            'limegreen': (50, 205, 50),
            'linen': (250, 240, 230),
            'magenta': (255, 0, 255),
            'maroon': (128, 0, 0),
            'mediumaquamarine': (102, 205, 170),
            'mediumblue': (0, 0, 205),
            'mediumorchid': (186, 85, 211),
            'mediumpurple': (147, 112, 219),
            'mediumseagreen': (60, 179, 113),
            'mediumslateblue': (123, 104, 238),
            'mediumspringgreen': (0, 250, 154),
            'mediumturquoise': (72, 209, 204),
            'mediumvioletred': (199, 21, 133),
            'midnightblue': (25, 25, 112),
            'mintcream': (245, 255, 250),
            'mistyrose': (255, 228, 225),
            'moccasin': (255, 228, 181),
            'navajowhite': (255, 222, 173),
            'navy': (0, 0, 128),
            'oldlace': (253, 245, 230),
            'olive': (128, 128, 0),
            'olivedrab': (107, 142, 35),
            'orange': (255, 165, 0),
            'orangered': (255, 69, 0),
            'orchid': (218, 112, 214),
            'palegoldenrod': (238, 232, 170),
            'palegreen': (152, 251, 152),
            'paleturquoise': (175, 238, 238),
            'palevioletred': (219, 112, 147),
            'papayawhip': (255, 239, 213),
            'peachpuff': (255, 218, 185),
            'peru': (205, 133, 63),
            'pink': (255, 192, 203),
            'plum': (221, 160, 221),
            'powderblue': (176, 224, 230),
            'purple': (128, 0, 128),
            'red': (255, 0, 0),
            'rosybrown': (188, 143, 143),
            'royalblue': (65, 105, 225),
            'saddlebrown': (139, 69, 19),
            'salmon': (250, 128, 114),
            'sandybrown': (244, 164, 96),
            'seagreen': (46, 139, 87),
            'seashell': (255, 245, 238),
            'sienna': (160, 82, 45),
            'silver': (192, 192, 192),
            'skyblue': (135, 206, 235),
            'slateblue': (106, 90, 205),
            'slategray': (112, 128, 144),
            'slategrey': (112, 128, 144),
            'snow': (255, 250, 250),
            'springgreen': (0, 255, 127),
            'steelblue': (70, 130, 180),
            'tan': (210, 180, 140),
            'teal': (0, 128, 128),
            'thistle': (216, 191, 216),
            'tomato': (255, 99, 71),
            'turquoise': (64, 224, 208),
            'violet': (238, 130, 238),
            'wheat': (245, 222, 179),
            'white': (255, 255, 255),
            'whitesmoke': (245, 245, 245),
            'yellow': (255, 255, 0),
            'yellowgreen': (154, 205, 50)
        }
        
        print(f"CSS4 색상 로드됨: {len(css4_colors)}개")
        return css4_colors
    except Exception as e:
        print(f"CSS4 색상 딕셔너리 로드 실패: {e}")
        return {}

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