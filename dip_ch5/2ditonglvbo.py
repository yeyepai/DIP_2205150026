"""
对灰度图片进行理想、巴特沃思以及高斯低通滤波处理，给出对比结果并分析，并和之前的空间的平滑滤波进行比较。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 获取图像的大小
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2  # 中心点

# 对图像进行傅立叶变换
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# 创建理想低通滤波器
def ideal_low_pass(shape, cutoff):
    filter = np.zeros(shape, np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff:
                filter[i, j] = 1
    return filter

# 创建巴特沃思低通滤波器
def butterworth_low_pass(shape, cutoff, order):
    filter = np.zeros(shape, np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            filter[i, j] = 1 / (1 + (distance / cutoff) ** (2 * order))
    return filter

# 创建高斯低通滤波器
def gaussian_low_pass(shape, cutoff):
    filter = np.zeros(shape, np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            filter[i, j] = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    return filter

# 设置截止频率和滤波器参数
cutoff = 30  # 截止频率
butterworth_order = 2  # 巴特沃思滤波器阶数

# 生成三种滤波器
ideal_filter = ideal_low_pass((rows, cols), cutoff)
butterworth_filter = butterworth_low_pass((rows, cols), cutoff, butterworth_order)
gaussian_filter = gaussian_low_pass((rows, cols), cutoff)

# 应用理想低通滤波器
ideal_dft_shift = dft_shift * ideal_filter
ideal_img_back = np.fft.ifft2(np.fft.ifftshift(ideal_dft_shift))
ideal_img_back = np.abs(ideal_img_back)

# 应用巴特沃思低通滤波器
butterworth_dft_shift = dft_shift * butterworth_filter
butterworth_img_back = np.fft.ifft2(np.fft.ifftshift(butterworth_dft_shift))
butterworth_img_back = np.abs(butterworth_img_back)

# 应用高斯低通滤波器
gaussian_dft_shift = dft_shift * gaussian_filter
gaussian_img_back = np.fft.ifft2(np.fft.ifftshift(gaussian_dft_shift))
gaussian_img_back = np.abs(gaussian_img_back)

# 显示原图和滤波后的图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('原图')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(ideal_img_back, cmap='gray')
plt.title('理想低通滤波')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(butterworth_img_back, cmap='gray')
plt.title('巴特沃思低通滤波')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(gaussian_img_back, cmap='gray')
plt.title('高斯低通滤波')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果图
cv2.imwrite('2理想低通滤波.png', ideal_img_back)
cv2.imwrite('2巴特沃思低通滤波.png', butterworth_img_back)
cv2.imwrite('2高斯低通滤波.png', gaussian_img_back)
