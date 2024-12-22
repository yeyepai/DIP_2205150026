"""
直方图统计与均衡化
"""

import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 计算原始灰度图像的直方图
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

# 对图像进行直方图均衡化
image_eq = cv2.equalizeHist(image)

# 计算均衡化后图像的直方图
hist_equalized = cv2.calcHist([image_eq], [0], None, [256], [0, 256])

# 显示原始图像和均衡化图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

# 原始灰度图像
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# 均衡化后的灰度图像
plt.subplot(2, 2, 2)
plt.imshow(image_eq, cmap='gray')
plt.title('直方图均衡化后')
plt.axis('off')

# 原始图像直方图
plt.subplot(2, 2, 3)
plt.plot(hist_original, color='black')
plt.title('原图的直方图')
plt.xlim([0, 256])
plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)  # 隐藏刻度标签
plt.grid(False)

# 均衡化后图像直方图
plt.subplot(2, 2, 4)
plt.plot(hist_equalized, color='black')
plt.title('直方图均衡化后的直方图')
plt.xlim([0, 256])
plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)  # 隐藏刻度标签
plt.grid(False)

plt.tight_layout()
plt.show()

# 保存均衡化后的图像
cv2.imwrite('3直方图均衡化.png', image_eq)
