# 2D_3D_processing.py - Unit Test
from processing_2D_3D import generate_depth_map, convert_2d_to_3d, visualize_3d_points

import pytest
import cv2
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch # 실제 창을 띄우지 않고, 시각화 함수가 호출되었는지 여부를 확인
from mpl_toolkits.mplot3d import Axes3D

# 1. depth map 생성 함수 테스트
def test_generate_deptgh_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    gray_image, depth_map = generate_depth_map(image)

    # 검증 시작
    # 출력 shape 검증
    assert gray_image.shape == (100, 100), "gray_image의 크기가 올바르지 않습니다."
    assert depth_map.shape == (100, 100, 3), "depth_map의 크기가 입력과 다릅니다."
    # 데이터 타입 검증
    assert gray_image.dtype == np.uint8, "gray_image의 데이터 타입이 올바르지 않습니다."
    assert isinstance(depth_map, np.ndarray), "depth_map이 numpy 배열이 아닙니다."

# 2. generate_depth_map 함수가 이미지를 None으로 받을 때 예외 처리 테스트
def test_generate_depth_map_none():
    with pytest.raises(ValueError) as excinfo:
        generate_depth_map(None)
    assert "이미지를 로드할 수 없습니다." in str(excinfo.value)

# 3. 3D 포인트 클라우드 생성 함수 테스트
def test_convert_2d_to_3d():
    gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    depth_map = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    points_3d = convert_2d_to_3d(gray_image, depth_map)

    # 출력 shape 검증
    assert points_3d.shape == (50, 50, 3), "points_3d의 크기가 올바르지 않습니다."
    # 데이터 타입 검증
    assert points_3d.dtype == np.float32, "points_3d의 데이터 타입이 올바르지 않습니다."

# 4. 3D 포인트 클라우드 시각화 함수 테스트
def test_visualize_3d_points():
    # 임의의 3D 포인트 클라우드 데이터 생성
    h, w = 50, 50
    dummy_points = np.zeros((h, w, 3), dtype=np.float32)

    with patch('matplotlib.pyplot.show'):
        try:
            visualize_3d_points(dummy_points)
        except Exception as e:
            pytest.fail(f"visualize_3d_points 함수 실행 중 예외 발생: {e}")

# 5. 잘못된 차원의 데이터가 입력될 때 예외 처리 테스트
def test_convert_2d_to_3d_invalid_input():
    # 채널수가 잘못된 이미지 생성
    invalid_points = np.zeros((10, 10, 2))

    # 3개의 채널이 아닌 경우 예외 발생 확인
    with pytest.raises(Exception):
        convert_2d_to_3d(invalid_points, invalid_points)


if __name__ == "__main__":
    pytest.main()