import os
import glob
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from color_extractor_kmeans import extract_dominant_colors_kmeans
from color_classification import extract_dominant_colors_classification, extract_dominant_colors_classification_XKCD, extract_dominant_colors_classification_CSS4

def process_all_images(folder_path='images'):
    """
    Process all images in the specified folder and return results.
    
    Args:
        folder_path (str): Path to the folder containing images
    
    Returns:
        list: List of tuples containing (image_path, colors, percentages, coverage, optimal_k)
    """
    start_time = time.time()
    print(f"=== 이미지 처리 시작: {time.strftime('%H:%M:%S')} ===")
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Get all image files in the folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"'{folder_path}' 폴더에서 이미지를 찾을 수 없습니다.")
        return []
    
    print(f"발견된 이미지 파일: {len(image_files)}개")
    for img_file in image_files:
        print(f"  - {img_file}")
    
    results = []
    total_processing_time = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"\n=== {os.path.basename(img_path)} 처리 중 ({i}/{len(image_files)}) ===")
            img_start_time = time.time()
            
            colors, percentages, coverage, optimal_k = extract_dominant_colors_kmeans(img_path)
            results.append((img_path, colors, percentages, coverage, optimal_k))
            
            img_processing_time = time.time() - img_start_time
            total_processing_time += img_processing_time
            
            print(f"추출된 주요 색상 (RGB): {colors}")
            print(f"사용된 최적 k: {optimal_k}")
            print(f"총 커버리지: {coverage*100:.1f}%")
            print(f"처리 시간: {img_processing_time:.2f}초")
            
        except Exception as e:
            print(f"오류 발생 ({img_path}): {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n=== 이미지 처리 완료 ===")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"평균 처리 시간: {total_processing_time/len(image_files):.2f}초/이미지")
    
    return results

def save_visualization_results(results, output_folder='results'):
    """
    Save visualization results as image files in the results folder.
    
    Args:
        results (list): List of tuples from process_all_images
        output_folder (str): Folder to save results
    """
    start_time = time.time()
    print(f"\n=== 시각화 결과 저장 시작: {time.strftime('%H:%M:%S')} ===")
    
    # Create results folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"결과를 '{output_folder}' 폴더에 저장 중...")
    
    for i, (img_path, colors, percentages, coverage, optimal_k) in enumerate(results, 1):
        img_start_time = time.time()
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        ax1.set_title(f'{os.path.basename(img_path)} (k={optimal_k})')
        ax1.axis('off')
        
        # Color palette
        if colors:
            color_bars = []
            labels = []
            
            # Sort percentages and get only the ones that correspond to selected colors
            sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
            # filtered_percentages = [(k, v) for k, v in sorted_percentages if v >= 0.05]
            filtered_percentages = sorted_percentages
            
            for j, color in enumerate(colors):
                color_bars.append([color])
                # Use the filtered percentages
                if j < len(filtered_percentages):
                    percentage = filtered_percentages[j][1] * 100
                    labels.append(f'Color {j+1}: {percentage:.1f}%')
                else:
                    labels.append(f'Color {j+1}: N/A')
            
            # Create color palette
            palette = np.array(color_bars)
            ax2.imshow(palette)
            ax2.set_title(f'Extracted Colors (Coverage: {coverage*100:.1f}%)')
            ax2.set_xticks([])
            ax2.set_yticks(range(len(colors)))
            ax2.set_yticklabels(labels)
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_folder, f"{base_name}_result.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        img_save_time = time.time() - img_start_time
        print(f"저장됨 ({i}/{len(results)}): {output_path} ({img_save_time:.2f}초)")
    
    total_save_time = time.time() - start_time
    print(f"\n=== 시각화 저장 완료 ===")
    print(f"총 저장 시간: {total_save_time:.2f}초")
    print(f"평균 저장 시간: {total_save_time/len(results):.2f}초/이미지")

if __name__ == '__main__':
    total_start_time = time.time()
    print("=== 색상 추출기 시작 (시각화 결과 저장 모드) ===")
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process all images in the images folder
    results = process_all_images('images')
    
    if results:
        print(f"\n=== 총 {len(results)}개 이미지 처리 완료 ===")
        
        # Save visualization results to results folder
        save_visualization_results(results, 'results')
        
        total_time = time.time() - total_start_time
        print(f"\n=== 전체 작업 완료 ===")
        print(f"총 소요 시간: {total_time:.2f}초")
        print(f"완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("처리할 이미지가 없습니다.") 