import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('./sample_image.jpg')

# BGR 이미지를 HSV로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색을 필터링
# 빨간색 풍선 색상을 고려하여 lower_red 범위를 유연하게 조정.
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 마스크 생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# 원본 이미지에서 빨간색 영역 마스킹 결과 생성
result = cv2.bitwise_and(image, image, mask=mask)

# 원본과 마스킹 이미지 출력
cv2.imshow('Original Image', image)
cv2.imshow('Red Filtered Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()