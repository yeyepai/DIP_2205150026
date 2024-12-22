"""
选择合适的滤波器对以上三张噪声污染图片进行噪声清除，并给出前后对比图。
"""

import cv2
import matplotlib.pyplot as plt

# 读取带噪声的图像
def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 高斯滤波器去噪（适用于高斯噪声）
def gaussian_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 均值滤波器去噪（适用于均匀噪声）
def mean_filter(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))

# 中值滤波器去噪（适用于椒盐噪声）
def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# 显示前后对比图像
def show_comparison(original, denoised, title):
    plt.figure(figsize=(12, 6))

    # 显示原图和去噪后的图像
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.title(f'{title} - 原图')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')
    plt.title(f'{title} - 去噪后')

    plt.tight_layout()
    plt.show()

# 读取三张带噪声的图像
gaussian_noisy_image = read_image('1高斯噪声.jpg')
uniform_noisy_image = read_image('1均匀噪声.jpg')
salt_pepper_noisy_image = read_image('1椒盐噪声.jpg')

# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'

# 高斯噪声图像去噪
gaussian_denoised_image = gaussian_filter(gaussian_noisy_image)

# 均匀噪声图像去噪
uniform_denoised_image = mean_filter(uniform_noisy_image)

# 椒盐噪声图像去噪
salt_pepper_denoised_image = median_filter(salt_pepper_noisy_image)

# 显示去噪后的图像前后对比
show_comparison(gaussian_noisy_image, gaussian_denoised_image, '对高斯噪声使用高斯滤波')
show_comparison(uniform_noisy_image, uniform_denoised_image, '对均匀噪声使用均值滤波')
show_comparison(salt_pepper_noisy_image, salt_pepper_denoised_image, '对椒盐噪声使用中值滤波')

# 保存去噪后的图像
cv2.imwrite('2高斯滤波.jpg', gaussian_denoised_image)
cv2.imwrite('2均匀滤波.jpg', uniform_denoised_image)
cv2.imwrite('2椒盐滤波.jpg', salt_pepper_denoised_image)
