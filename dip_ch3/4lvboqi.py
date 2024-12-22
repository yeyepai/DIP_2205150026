"""
对原图进行三种不同的平滑处理，选择合适的均值滤波器、方框滤波器以及高斯滤波器，需要有前后处理的图片对比，以及说明哪种滤波器最好。
"""

import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 应用均值滤波器
mean_filtered = cv2.blur(image, (5, 5))

# 应用方框滤波器
box_filtered = cv2.boxFilter(image, -1, (5, 5), normalize=True)

# 应用高斯滤波器
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

# 显示原始图像和处理后的图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# 均值滤波处理后的图像
plt.subplot(2, 2, 2)
plt.imshow(mean_filtered, cmap='gray')
plt.title('均值滤波后')
plt.axis('off')

# 方框滤波处理后的图像
plt.subplot(2, 2, 3)
plt.imshow(box_filtered, cmap='gray')
plt.title('方框滤波后')
plt.axis('off')

# 高斯滤波处理后的图像
plt.subplot(2, 2, 4)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('高斯滤波后')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存处理后的图像
cv2.imwrite('4均值滤波.png', mean_filtered)
cv2.imwrite('4方框滤波.png', box_filtered)
cv2.imwrite('4高斯滤波.png', gaussian_filtered)
