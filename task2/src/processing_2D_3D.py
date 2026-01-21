import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 이미지 로드
image = cv2.imread('./data/image_000.jpg')

# Depth map 생성 함수
def generate_depth_map(image):
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    
    # 그레이 스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 깊이 맵 생성
    depth_map = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    return gray_image, depth_map

# 3D 포인트 클라우드 생성 함수
def convert_2d_to_3d(gray_image, depth_map):
    h, w = gray_image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray_image.astype(np.float32)

    points_3d = np.dstack((X, Y, Z)).astype(np.float32)

    return points_3d

# 3D 포인트 클라우드 시각화 함수
def visualize_3d_points(points_3d):
    flat_points = points_3d.reshape(-1, 3)

    sample_rate = 10
    sampled_points = flat_points[::sample_rate]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # X, Y, Z 좌표 추출
    xs = sampled_points[:, 0]
    ys = sampled_points[:, 1]
    zs = sampled_points[:, 2]

    # 시각화
    sc = ax.scatter(xs, ys, zs, c=zs, cmap='jet', marker='o', s=1)
    plt.colorbar(sc, label='Depth (Z value)')
    ax.set_title('3D Point Cloud Visualization')
    plt.show()

'''
---------------------------------------------------------------------------------
'''

# 그레이 스케일 변환 및 깊이 맵 생성
gray_image, depth_map = generate_depth_map(image)

# 2D 이미지를 3D 포인트 클라우드로 변환
points_3d = convert_2d_to_3d(gray_image, depth_map)

cv2.imshow("Original Image", image) # 원본 이미지 표시
cv2.imshow("Depth Map", depth_map) # 깊이 맵 표시
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3D 포인트 클라우드 시각화
visualize_3d_points(points_3d)