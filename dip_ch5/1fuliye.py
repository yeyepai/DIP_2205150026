"""
给出一张灰度图片的离散傅立叶变换频谱图，再进行逆变换。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 对图像进行傅立叶变换
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# 计算频谱图
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# 显示原图和频谱图
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('原图')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('离散傅里叶变换频谱图')
plt.axis('off')
plt.show()

# 使用 OpenCV 保存频谱图
cv2.imwrite('1离散傅里叶频谱图.png', magnitude_spectrum)

# 对傅立叶变换结果进行逆变换
idft_shift = np.fft.ifftshift(dft_shift)
img_back = np.fft.ifft2(idft_shift)
img_back = np.abs(img_back)

# 显示逆变换后的图像
plt.figure(figsize=(5,5))
plt.imshow(img_back, cmap='gray')
plt.title('逆变换后图')
plt.axis('off')
plt.show()

# 使用 OpenCV 保存逆变换后的图像
cv2.imwrite('1逆变换后图.png', img_back)
