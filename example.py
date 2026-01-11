"""
픽셀 단위 이미지 처리 예제 스크립트

이 스크립트는 pixel_processing.py 모듈의 다양한 기능을 시연합니다.
"""

import numpy as np
import pixel_processing as pp


def main():
    print("=" * 60)
    print("픽셀 단위 이미지 처리 예제")
    print("=" * 60)
    
    # 1. 테스트용 그라디언트 이미지 생성
    print("\n1. 그라디언트 이미지 생성 중...")
    gradient_h = pp.create_gradient_image(256, 256, horizontal=True)
    gradient_v = pp.create_gradient_image(256, 256, horizontal=False)
    
    # 이미지 저장
    try:
        pp.save_image(gradient_h, 'output_gradient_horizontal.png')
        pp.save_image(gradient_v, 'output_gradient_vertical.png')
        print("   ✓ 그라디언트 이미지 저장 완료")
        print("     - output_gradient_horizontal.png")
        print("     - output_gradient_vertical.png")
    except ImportError:
        print("   ⚠ Pillow 라이브러리가 없어 이미지 저장을 건너뜁니다.")
        print("     'pip install Pillow'로 설치하세요.")
        return
    
    # 2. 테스트용 컬러 이미지 생성
    print("\n2. 테스트 이미지 생성 중...")
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    # 빨간색 영역
    test_image[0:100, 0:100] = [255, 0, 0]
    # 초록색 영역
    test_image[0:100, 100:200] = [0, 255, 0]
    # 파란색 영역
    test_image[100:200, 0:100] = [0, 0, 255]
    # 노란색 영역
    test_image[100:200, 100:200] = [255, 255, 0]
    
    pp.save_image(test_image, 'output_test_color.png')
    print("   ✓ 테스트 컬러 이미지 저장: output_test_color.png")
    
    # 3. 이미지 통계 정보
    print("\n3. 이미지 통계 정보:")
    stats = pp.get_image_stats(test_image)
    print(f"   - 이미지 크기: {stats['shape']}")
    print(f"   - 최소값: {stats['min']}")
    print(f"   - 최대값: {stats['max']}")
    print(f"   - 평균값: {stats['mean']:.2f}")
    print(f"   - 표준편차: {stats['std']:.2f}")
    
    # 4. 그레이스케일 변환
    print("\n4. 그레이스케일 변환 중...")
    gray_image = pp.rgb_to_grayscale(test_image)
    pp.save_image(gray_image, 'output_grayscale.png')
    print("   ✓ 그레이스케일 이미지 저장: output_grayscale.png")
    
    # 5. 밝기 조정
    print("\n5. 밝기 조정 중...")
    bright_image = pp.adjust_brightness(test_image, 50)
    dark_image = pp.adjust_brightness(test_image, -50)
    pp.save_image(bright_image, 'output_bright.png')
    pp.save_image(dark_image, 'output_dark.png')
    print("   ✓ 밝게: output_bright.png")
    print("   ✓ 어둡게: output_dark.png")
    
    # 6. 대비 조정
    print("\n6. 대비 조정 중...")
    high_contrast = pp.adjust_contrast(test_image, 1.5)
    low_contrast = pp.adjust_contrast(test_image, 0.5)
    pp.save_image(high_contrast, 'output_high_contrast.png')
    pp.save_image(low_contrast, 'output_low_contrast.png')
    print("   ✓ 높은 대비: output_high_contrast.png")
    print("   ✓ 낮은 대비: output_low_contrast.png")
    
    # 7. 색상 반전
    print("\n7. 색상 반전 중...")
    inverted_image = pp.invert_colors(test_image)
    pp.save_image(inverted_image, 'output_inverted.png')
    print("   ✓ 반전된 이미지: output_inverted.png")
    
    # 8. 이진화 (Thresholding)
    print("\n8. 이진화 적용 중...")
    binary_image = pp.apply_threshold(gray_image, threshold=128)
    pp.save_image(binary_image, 'output_binary.png')
    print("   ✓ 이진화 이미지: output_binary.png")
    
    # 9. 이미지 자르기
    print("\n9. 이미지 자르기...")
    cropped = pp.crop_image(test_image, 50, 50, 100, 100)
    pp.save_image(cropped, 'output_cropped.png')
    print("   ✓ 잘린 이미지: output_cropped.png")
    
    # 10. 픽셀 값 읽기/쓰기
    print("\n10. 픽셀 값 읽기/쓰기 테스트...")
    pixel_value = pp.get_pixel_value(test_image, 50, 50)
    print(f"   - (50, 50) 위치의 픽셀 값: {pixel_value}")
    
    modified_image = test_image.copy()
    # 대각선에 흰색 선 그리기
    for i in range(min(modified_image.shape[0], modified_image.shape[1])):
        modified_image = pp.set_pixel_value(modified_image, i, i, (255, 255, 255))
    pp.save_image(modified_image, 'output_with_line.png')
    print("   ✓ 대각선이 추가된 이미지: output_with_line.png")
    
    print("\n" + "=" * 60)
    print("모든 이미지 처리 작업이 완료되었습니다!")
    print("생성된 파일들:")
    print("  - output_gradient_horizontal.png")
    print("  - output_gradient_vertical.png")
    print("  - output_test_color.png")
    print("  - output_grayscale.png")
    print("  - output_bright.png")
    print("  - output_dark.png")
    print("  - output_high_contrast.png")
    print("  - output_low_contrast.png")
    print("  - output_inverted.png")
    print("  - output_binary.png")
    print("  - output_cropped.png")
    print("  - output_with_line.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
