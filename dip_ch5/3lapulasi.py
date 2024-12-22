"""
对灰度图片进行频率域上的拉普拉斯算子处理进行图像增强，并与空间域拉普拉斯算子进行比较，给出对比结果分析。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)
image_copy = image.copy()

# 将图像转换为浮点数
image_float = np.float32(image)

# 进行傅里叶变换
dft = np.fft.fft2(image_float)
dft_shifted = np.fft.fftshift(dft)

# 创建拉普拉斯算子
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
laplacian_filter = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        laplacian_filter[i, j] = -4 * np.pi**2 * ((i - crow)**2 + (j - ccol)**2)

# 在频率域应用拉普拉斯算子
filtered_dft = dft_shifted * laplacian_filter

# 进行反傅里叶变换
filtered_image = np.fft.ifftshift(filtered_dft)
inverse_dft = np.fft.ifft2(filtered_image)
frequency_laplacian = np.abs(inverse_dft)

# 归一化到0-255范围
frequency_laplacian_normalized = cv2.normalize(frequency_laplacian, None, 0, 255, cv2.NORM_MINMAX)
frequency_laplacian_normalized = np.uint8(frequency_laplacian_normalized)

# 将频率域拉普拉斯边缘图和原图相加（可以设置合适的权重）
processed_image = cv2.addWeighted(image, 1.5, frequency_laplacian_normalized, -0.5, 0)

# 绘制结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

# 显示原图
plt.subplot(1, 3, 1)
plt.title('原图')
plt.imshow(image, cmap='gray')
plt.axis('off')

# 显示处理后的图像
plt.subplot(1, 3, 2)
plt.title('处理后的图像')
plt.imshow(processed_image, cmap='gray')
plt.axis('off')

# 显示频率域拉普拉斯边缘图
plt.subplot(1, 3, 3)
plt.title('频率域拉普拉斯边缘图')
plt.imshow(frequency_laplacian_normalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Laplacian 二阶锐化
def laplacian_operator(img):
    laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(laplacian)

# 计算空间域拉普拉斯边缘图
spatial_laplacian_img = laplacian_operator(image)

# 合成处理后的空间域图像
spatial_processed_image = cv2.addWeighted(image, 1.5, spatial_laplacian_img, -0.5, 0)

# 保存处理后的边缘图和合成图像
cv2.imwrite('3频率域边缘图.png', frequency_laplacian_normalized)
cv2.imwrite('3频率域处理后的图像.png', processed_image)

# 保存空间域处理后的边缘图和合成图像
cv2.imwrite('3空间域边缘图.png', spatial_laplacian_img)
cv2.imwrite('3空间域处理后的图像.png', spatial_processed_image)