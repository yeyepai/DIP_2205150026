"""
使用Otsu方法直接进行图像分割，再按照指定的方法进行图像分割，将两者进行比较分析。
（指定方法）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 1. 使用拉普拉斯算子计算边缘图像（绝对值）
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian_abs = np.abs(laplacian)

# 2. 计算阈值T，例如10%的最大值
T = 0.1 * np.max(laplacian_abs)

# 3. 根据阈值T进行二值化，得到边缘图像g1
_, g1 = cv2.threshold(laplacian_abs, T, 255, cv2.THRESH_BINARY)

# 4. 从原图f中选取对应于g1像素值为1的位置的像素，形成模板图像
mask = (g1 == 255)  # 找出g1中值为255的位置
template_image = np.zeros_like(image)
template_image[mask] = image[mask]

# 5. 计算直方图，仅对g1为255的位置进行计算
masked_pixels = image[mask]
histogram = cv2.calcHist([masked_pixels], [0], None, [256], [0, 256])

# 6. 使用直方图来选择一个阈值进行全局分割
# 这里我们可以选择直方图的峰值（最大频率的位置）作为阈值
hist_max_idx = np.argmax(histogram)  # 找到直方图中的最大值的位置
hist_threshold = hist_max_idx  # 以最大值的灰度级作为阈值

# 使用计算得到的阈值对原图像进行全局分割
_, manual_threshold_result = cv2.threshold(image, hist_threshold, 255, cv2.THRESH_BINARY)

# 7. 使用Otsu方法进行全局分割（作为对比）
_, otsu_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示图像和结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(14, 10))

# 原图
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# 拉普拉斯算子绝对值结果
plt.subplot(2, 3, 2)
plt.imshow(laplacian_abs, cmap='gray')
plt.title('拉普拉斯算子绝对值结果')
plt.axis('off')

# 阈值化后的g1
plt.subplot(2, 3, 3)
plt.imshow(g1, cmap='gray')
plt.title('阈值化后的g1图')
plt.axis('off')

# 模板图像
plt.subplot(2, 3, 4)
plt.imshow(template_image, cmap='gray')
plt.title('强边缘模板图')
plt.axis('off')

# 手动分割结果
plt.subplot(2, 3, 5)
plt.imshow(manual_threshold_result, cmap='gray')
plt.title('手动分割结果')
plt.axis('off')

# Otsu方法分割结果
plt.subplot(2, 3, 6)
plt.imshow(otsu_result, cmap='gray')
plt.title('Otsu方法分割结果')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果图像
cv2.imwrite('3拉普拉斯绝对值.png', laplacian_abs)
cv2.imwrite('3阈值图像.png', g1)
cv2.imwrite('3模板图像.png', template_image)
cv2.imwrite('3手动分割结果.png', manual_threshold_result)
cv2.imwrite('3分割结果_otsu.png', otsu_result)
