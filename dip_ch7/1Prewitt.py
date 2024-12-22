"""
对灰度图像进行Prewitt梯度算子边缘检测，分析一下图片效果，考虑是否需要平滑后再次检测，
或者采用对角线的Prewitt梯度算子进行处理，并给出原因？最后进行阈值化使边缘结果更加清晰。
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

# 定义对角线Prewitt算子
prewitt_diag1 = np.array([[1, 1, 0],
                           [1, 0, -1],
                           [0, -1, -1]])

prewitt_diag2 = np.array([[0, 1, 1],
                           [-1, 0, 1],
                           [-1, -1, 0]])

# 使用卷积运算进行边缘检测
edge_x = cv2.filter2D(image, -1, prewitt_x)
edge_y = cv2.filter2D(image, -1, prewitt_y)

# 对角线方向的边缘
edge_diag1 = cv2.filter2D(image, -1, prewitt_diag1)
edge_diag2 = cv2.filter2D(image, -1, prewitt_diag2)

# 计算总边缘强度
edges = np.sqrt(np.square(edge_x) + np.square(edge_y) + np.square(edge_diag1) + np.square(edge_diag2))

# 打印边缘图像的最小值和最大值，检查其范围
print(f"Min value: {np.min(edges)}, Max value: {np.max(edges)}")

# 将浮动数据类型转换为uint8，并确保在0-255范围内
edges_normalized = np.uint8(np.clip(edges, 0, 255))

# 使用固定阈值进行二值化
threshold_value = 13  # 调低阈值
_, edges_binary = cv2.threshold(edges_normalized, threshold_value, 255, cv2.THRESH_BINARY)

# 显示结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

# 原图
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# x方向边缘
plt.subplot(2, 3, 2)
plt.imshow(edge_x, cmap='gray')
plt.title('Prewitt X方向边缘')
plt.axis('off')

# y方向边缘
plt.subplot(2, 3, 3)
plt.imshow(edge_y, cmap='gray')
plt.title('Prewitt Y方向边缘')
plt.axis('off')

# 对角线1方向边缘
plt.subplot(2, 3, 4)
plt.imshow(edge_diag1, cmap='gray')
plt.title('Prewitt 对角线1方向边缘')
plt.axis('off')

# 对角线2方向边缘
plt.subplot(2, 3, 5)
plt.imshow(edge_diag2, cmap='gray')
plt.title('Prewitt 对角线2方向边缘')
plt.axis('off')

# 显示阈值化前的边缘图像
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.imshow(edges_normalized, cmap='gray')
plt.title('阈值化前的边缘图像')
plt.axis('off')

# 显示阈值化后的边缘
plt.subplot(1, 2, 2)
plt.imshow(edges_binary, cmap='gray')
plt.title('阈值化后的边缘图像')
plt.axis('off')

plt.show()

# 保存所有图像到本地

# 保存x方向边缘
cv2.imwrite('1Prewitt_X方向边缘.png', edge_x)

# 保存y方向边缘
cv2.imwrite('1Prewitt_Y方向边缘.png', edge_y)

# 保存对角线1方向边缘
cv2.imwrite('1Prewitt_对角线1方向边缘.png', edge_diag1)

# 保存对角线2方向边缘
cv2.imwrite('1Prewitt_对角线2方向边缘.png', edge_diag2)

# 保存归一化后的边缘图像（阈值化前）
cv2.imwrite('1Prewitt_阈值化前.png', edges_normalized)

# 保存阈值化后的边缘图像
cv2.imwrite('1Prewitt_阈值化后.png', edges_binary)
