from datasets import load_dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# 이미지 수집 함수
def load_and_save_images(n=20, save_path='images'):
    dataset = load_dataset("ethz/food101", split=f"train[:{n}]")
    images = [data['image'] for data in dataset]

    os.makedirs(save_path, exist_ok=True)

    bgr_images = []

    for i, image in enumerate(images):
        # PIL Image를 numpy array로 변환 후 BGR로 변환
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 원본 크기 그대로 저장
        cv2.imwrite(f"{save_path}/image_{i:03d}.jpg", img_bgr)

        bgr_images.append(img_bgr)
    
    return bgr_images

# 밝기가 임계값 이하인 경우 필터링하는 함수
def filter_dark_images(images, threshold=0.3):
    filtered_images = []
    
    for i in range(len(images)):
        image = images[i]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_image) / 255.0 # 0~1 사이 값으로 정규화
        
        if brightness >= threshold:
            filtered_images.append(image)

    return filtered_images

# 필터링 거친 후 이미지 전처리
def preprocess_images(images):
    preprocessed_images = []

    for image in images:
        # 224 * 224 크기로 리사이즈
        image = cv2.resize(image, (224, 224))

        # 색상 변환 (GrayScale, Normalize 적용하기)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        normalized_image = image / 255.0

        # 노이즈 제거 - 가우시안 Blur 필터 적용
        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

        preprocessed_images.append(blurred_image)

    return preprocessed_images

# 전처리 후의 이미지 증강 및 저장
def augment_images(images, save_path='./augmented_images'):
    # 폴더가 없으면 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 저장 방식은 원본 이미지 번호 + 증강 방법
    for i in range(len(images)):
        image = images[i]
        image = (image * 255).astype(np.uint8)
        
        # 1. 좌우 반전
        flipped = cv2.flip(image, 1)
        cv2.imwrite(f"{save_path}/flipped_{i}.jpg", flipped)
        
        # 2. 회전
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(f"{save_path}/rotated_{i}.jpg", rotated)
        
        # 3. 색상 변화
        color_changed = cv2.convertScaleAbs(image, alpha=1.25, beta=30)
        cv2.imwrite(f"{save_path}/color_changed_{i}.jpg", color_changed)
        
        # 4. 채도 변화
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], 50)
        saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"{save_path}/saturated_{i}.jpg", saturated_image)

# ------------------------------------------------------------
# 코드 진행하기
# 기본 이미지 수집 및 폴더 생성하여 저장
images = load_and_save_images(n=20, save_path='images')
print(f"수집된 이미지 개수: {len(images)}")

# 밝기가 임계값 이하인 이미지 필터링
filtered_images = filter_dark_images(images)
print(f"필터링 후 이미지 개수: {len(filtered_images)}")

# 필터링 거친 후의 이미지 전처리
preprocessed_images = preprocess_images(filtered_images)
print(f"전처리 후 이미지 개수: {len(preprocessed_images)}")

# 전처리 후의 이미지 증강 및 저장
augmented_images = augment_images(preprocessed_images, save_path='augmented_images')
print('이미지 증강 및 저장 완료.')