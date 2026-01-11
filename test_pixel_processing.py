"""
픽셀 단위 이미지 처리 모듈 테스트

이 파일은 pixel_processing.py의 기능을 검증하는 단위 테스트입니다.
"""

import numpy as np
import pixel_processing as pp


def test_rgb_to_grayscale():
    """RGB to grayscale 변환 테스트"""
    # 빨간색 이미지 생성
    red_image = np.zeros((10, 10, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # Red channel
    
    gray = pp.rgb_to_grayscale(red_image)
    
    # 그레이스케일 결과 확인 (0.299 * 255)
    expected = int(0.299 * 255)
    assert gray.shape == (10, 10), "그레이스케일 이미지 형태가 잘못되었습니다"
    assert np.allclose(gray, expected, atol=1), "그레이스케일 변환 값이 잘못되었습니다"
    print("✓ RGB to Grayscale 테스트 통과")


def test_adjust_brightness():
    """밝기 조정 테스트"""
    image = np.ones((10, 10), dtype=np.uint8) * 100
    
    # 밝기 증가
    bright = pp.adjust_brightness(image, 50)
    assert np.all(bright == 150), "밝기 증가가 제대로 작동하지 않습니다"
    
    # 밝기 감소
    dark = pp.adjust_brightness(image, -50)
    assert np.all(dark == 50), "밝기 감소가 제대로 작동하지 않습니다"
    
    # 범위 초과 테스트 (클리핑)
    over_bright = pp.adjust_brightness(image, 200)
    assert np.all(over_bright == 255), "밝기 오버플로우 클리핑이 작동하지 않습니다"
    
    print("✓ 밝기 조정 테스트 통과")


def test_adjust_contrast():
    """대비 조정 테스트"""
    image = np.ones((10, 10), dtype=np.uint8) * 128
    
    # 대비 증가 (128에서는 변화 없어야 함)
    contrast = pp.adjust_contrast(image, 2.0)
    assert np.allclose(contrast, 128, atol=1), "대비 조정이 제대로 작동하지 않습니다"
    
    print("✓ 대비 조정 테스트 통과")


def test_invert_colors():
    """색상 반전 테스트"""
    image = np.zeros((10, 10), dtype=np.uint8)
    
    inverted = pp.invert_colors(image)
    assert np.all(inverted == 255), "색상 반전이 제대로 작동하지 않습니다"
    
    # 다시 반전하면 원본과 같아야 함
    double_inverted = pp.invert_colors(inverted)
    assert np.all(double_inverted == image), "이중 반전이 원본을 복원하지 못했습니다"
    
    print("✓ 색상 반전 테스트 통과")


def test_apply_threshold():
    """이진화 테스트"""
    image = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    
    binary = pp.apply_threshold(image, threshold=128)
    expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    
    assert np.all(binary == expected), "이진화가 제대로 작동하지 않습니다"
    print("✓ 이진화 테스트 통과")


def test_crop_image():
    """이미지 자르기 테스트"""
    image = np.arange(100).reshape(10, 10).astype(np.uint8)
    
    cropped = pp.crop_image(image, 2, 2, 5, 5)
    
    assert cropped.shape == (5, 5), "자른 이미지의 크기가 잘못되었습니다"
    assert cropped[0, 0] == image[2, 2], "자르기가 제대로 작동하지 않습니다"
    
    print("✓ 이미지 자르기 테스트 통과")


def test_pixel_operations():
    """픽셀 읽기/쓰기 테스트"""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # 픽셀 쓰기
    new_image = pp.set_pixel_value(image, 5, 5, (255, 128, 64))
    
    # 픽셀 읽기
    value = pp.get_pixel_value(new_image, 5, 5)
    
    assert value == (255, 128, 64), "픽셀 읽기/쓰기가 제대로 작동하지 않습니다"
    print("✓ 픽셀 읽기/쓰기 테스트 통과")


def test_create_gradient():
    """그라디언트 이미지 생성 테스트"""
    gradient = pp.create_gradient_image(256, 100, horizontal=True)
    
    assert gradient.shape == (100, 256), "그라디언트 이미지 크기가 잘못되었습니다"
    assert gradient[0, 0] == 0, "그라디언트 시작값이 잘못되었습니다"
    assert gradient[0, -1] == 255, "그라디언트 끝값이 잘못되었습니다"
    
    print("✓ 그라디언트 생성 테스트 통과")


def test_image_stats():
    """이미지 통계 테스트"""
    image = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    
    stats = pp.get_image_stats(image)
    
    assert stats['min'] == 0, "최소값이 잘못되었습니다"
    assert stats['max'] == 255, "최대값이 잘못되었습니다"
    assert abs(stats['mean'] - 138.75) < 0.01, "평균값이 잘못되었습니다"
    assert stats['shape'] == (2, 2), "형태가 잘못되었습니다"
    
    print("✓ 이미지 통계 테스트 통과")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("픽셀 단위 이미지 처리 모듈 테스트 시작")
    print("=" * 60)
    print()
    
    tests = [
        test_rgb_to_grayscale,
        test_adjust_brightness,
        test_adjust_contrast,
        test_invert_colors,
        test_apply_threshold,
        test_crop_image,
        test_pixel_operations,
        test_create_gradient,
        test_image_stats
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} 실패: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} 에러: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
