"""
픽셀 단위 이미지 처리 모듈 (Pixel-level Image Processing Module)

이 모듈은 이미지의 픽셀 단위 처리를 위한 기본 함수들을 제공합니다.
외부 라이브러리(OpenCV, PIL 등) 없이 순수 Python과 NumPy만을 사용합니다.
"""

import numpy as np
from typing import Tuple, Union


def load_image(filepath: str) -> np.ndarray:
    """
    이미지 파일을 로드합니다.
    
    Args:
        filepath: 이미지 파일 경로
        
    Returns:
        numpy array 형태의 이미지 (Height x Width x Channels)
    """
    try:
        from PIL import Image
        img = Image.open(filepath)
        return np.array(img)
    except ImportError:
        raise ImportError("PIL/Pillow 라이브러리가 필요합니다. 'pip install Pillow'로 설치하세요.")


def save_image(image: np.ndarray, filepath: str) -> None:
    """
    이미지를 파일로 저장합니다.
    
    Args:
        image: numpy array 형태의 이미지
        filepath: 저장할 파일 경로
    """
    try:
        from PIL import Image
        # numpy array를 uint8로 변환 (0-255 범위)
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(filepath)
    except ImportError:
        raise ImportError("PIL/Pillow 라이브러리가 필요합니다. 'pip install Pillow'로 설치하세요.")


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 그레이스케일로 변환합니다.
    픽셀 단위로 직접 계산: Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: RGB 이미지 (Height x Width x 3)
        
    Returns:
        그레이스케일 이미지 (Height x Width)
    """
    if len(image.shape) == 2:
        return image  # 이미 그레이스케일
    
    # 가중치를 사용한 그레이스케일 변환
    grayscale = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
            grayscale[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    
    return grayscale.astype(np.uint8)


def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
    """
    이미지의 밝기를 조정합니다.
    각 픽셀에 value를 더합니다.
    
    Args:
        image: 입력 이미지
        value: 밝기 조정 값 (-255 ~ 255)
        
    Returns:
        밝기가 조정된 이미지
    """
    result = image.astype(np.int16)
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if len(result.shape) == 3:
                for k in range(result.shape[2]):
                    result[i, j, k] = result[i, j, k] + value
            else:
                result[i, j] = result[i, j] + value
    
    # 0-255 범위로 클리핑
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    이미지의 대비를 조정합니다.
    각 픽셀을 factor만큼 스케일링합니다.
    
    Args:
        image: 입력 이미지
        factor: 대비 조정 값 (0.0 ~ 3.0, 1.0이 원본)
        
    Returns:
        대비가 조정된 이미지
    """
    result = image.astype(np.float32)
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if len(result.shape) == 3:
                for k in range(result.shape[2]):
                    result[i, j, k] = (result[i, j, k] - 128) * factor + 128
            else:
                result[i, j] = (result[i, j] - 128) * factor + 128
    
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def invert_colors(image: np.ndarray) -> np.ndarray:
    """
    이미지의 색상을 반전시킵니다.
    각 픽셀 값을 255에서 뺍니다.
    
    Args:
        image: 입력 이미지
        
    Returns:
        색상이 반전된 이미지
    """
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if len(image.shape) == 3:
                for k in range(image.shape[2]):
                    result[i, j, k] = 255 - image[i, j, k]
            else:
                result[i, j] = 255 - image[i, j]
    
    return result


def apply_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    이미지에 이진화(thresholding)를 적용합니다.
    threshold보다 큰 픽셀은 255, 작은 픽셀은 0으로 설정합니다.
    
    Args:
        image: 입력 이미지 (그레이스케일 권장)
        threshold: 임계값 (0-255)
        
    Returns:
        이진화된 이미지
    """
    if len(image.shape) == 3:
        image = rgb_to_grayscale(image)
    
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > threshold:
                result[i, j] = 255
            else:
                result[i, j] = 0
    
    return result


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    이미지를 자릅니다.
    
    Args:
        image: 입력 이미지
        x: 시작 x 좌표
        y: 시작 y 좌표
        width: 자를 너비
        height: 자를 높이
        
    Returns:
        잘린 이미지
    """
    return image[y:y+height, x:x+width].copy()


def get_pixel_value(image: np.ndarray, x: int, y: int) -> Union[int, Tuple[int, int, int]]:
    """
    특정 위치의 픽셀 값을 가져옵니다.
    
    Args:
        image: 입력 이미지
        x: x 좌표
        y: y 좌표
        
    Returns:
        픽셀 값 (그레이스케일의 경우 int, RGB의 경우 tuple)
    """
    if len(image.shape) == 3:
        return tuple(image[y, x])
    else:
        return int(image[y, x])


def set_pixel_value(image: np.ndarray, x: int, y: int, value: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    """
    특정 위치의 픽셀 값을 설정합니다.
    
    Args:
        image: 입력 이미지
        x: x 좌표
        y: y 좌표
        value: 설정할 픽셀 값
        
    Returns:
        수정된 이미지
    """
    result = image.copy()
    if len(image.shape) == 3:
        result[y, x] = value
    else:
        result[y, x] = value
    return result


def create_gradient_image(width: int, height: int, horizontal: bool = True) -> np.ndarray:
    """
    그라디언트 이미지를 생성합니다.
    
    Args:
        width: 이미지 너비
        height: 이미지 높이
        horizontal: True면 수평 그라디언트, False면 수직 그라디언트
        
    Returns:
        그라디언트 이미지
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    if horizontal:
        for i in range(height):
            for j in range(width):
                image[i, j] = int(255 * j / (width - 1))
    else:
        for i in range(height):
            for j in range(width):
                image[i, j] = int(255 * i / (height - 1))
    
    return image


def get_image_stats(image: np.ndarray) -> dict:
    """
    이미지의 통계 정보를 계산합니다.
    
    Args:
        image: 입력 이미지
        
    Returns:
        통계 정보 딕셔너리 (min, max, mean, std)
    """
    return {
        'min': int(np.min(image)),
        'max': int(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'shape': image.shape
    }
