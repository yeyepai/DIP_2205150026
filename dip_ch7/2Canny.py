"""
对灰度图像进行Canny算子边缘检测，并和Prewitt算子检测结果进行比较。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 定义Prewitt算子
prewitt_x = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])

prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

# 使用卷积运算进行Prewitt边缘检测
edge_x = cv2.filter2D(image, -1, prewitt_x)
edge_y = cv2.filter2D(image, -1, prewitt_y)

# 计算Prewitt边缘强度
edges_prewitt = np.sqrt(np.square(edge_x) + np.square(edge_y))

# Canny边缘检测
edges_canny = cv2.Canny(image, 100, 200)  # 100, 200 为低阈值和高阈值

# 显示图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

# 原图
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# Prewitt X 方向边缘
plt.subplot(2, 3, 2)
plt.imshow(edge_x, cmap='gray')
plt.title('Prewitt X方向边缘')
plt.axis('off')

# Prewitt Y 方向边缘
plt.subplot(2, 3, 3)
plt.imshow(edge_y, cmap='gray')
plt.title('Prewitt Y方向边缘')
plt.axis('off')

# Prewitt 组合边缘
plt.subplot(2, 3, 4)
plt.imshow(edges_prewitt, cmap='gray')
plt.title('Prewitt 边缘检测结果')
plt.axis('off')

# Canny 边缘检测
plt.subplot(2, 3, 5)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny 边缘检测结果')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果图像

# 保存Prewitt X方向边缘
cv2.imwrite('2Prewitt_X方向边缘.png', edge_x)

# 保存Prewitt Y方向边缘
cv2.imwrite('2Prewitt_Y方向边缘.png', edge_y)

# 保存Prewitt组合边缘
cv2.imwrite('2Prewitt_组合边缘.png', np.uint8(edges_prewitt))

# 保存Canny边缘检测结果
cv2.imwrite('2Canny.png', edges_canny)
