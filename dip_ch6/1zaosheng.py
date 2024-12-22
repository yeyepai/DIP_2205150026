"""
在灰度图片上加上高斯噪声、均匀噪声以及椒盐噪声，分别给出原图加上噪声污染后的图片，并给出对应的四张直方图。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# 添加均匀噪声
def add_uniform_noise(image, low=0, high=50):
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# 添加椒盐噪声
def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    total_pixels = image.size
    salt = int(salt_prob * total_pixels)
    pepper = int(pepper_prob * total_pixels)

    # 添加盐噪声（白色像素）
    salt_coords = [np.random.randint(0, i - 1, salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # 添加椒噪声（黑色像素）
    pepper_coords = [np.random.randint(0, i - 1, pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


# 显示并保存直方图（仅保存直方图）
def save_histogram(image, title, filename):
    plt.figure(figsize=(6, 6))

    # 绘制直方图
    if title != '椒盐噪声':
        plt.hist(image.ravel(), bins=254, range=(1, 254), color='black', histtype='step')
    else:
        plt.hist(image.ravel(), bins=256, range=(0, 255), color='black', histtype='step')
    plt.title(f'{title}直方图')
    plt.xlabel('像素值')
    plt.ylabel('Frequency')

    # 关闭y轴坐标轴
    plt.gca().get_yaxis().set_visible(False)

    # 保存直方图为图片
    plt.tight_layout()
    plt.savefig(f'1{filename}直方图.jpg')  # 保存为直方图文件
    plt.show()


# 读取原始灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'

# 检查图像是否成功加载
if image is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 保存原图直方图
    save_histogram(image, '原图', '原图')

    # 添加不同的噪声
    gaussian_noisy_image = add_gaussian_noise(image)
    uniform_noisy_image = add_uniform_noise(image)
    salt_pepper_noisy_image = add_salt_pepper_noise(image)

    # 保存带噪声的图像
    cv2.imwrite('1高斯噪声.jpg', gaussian_noisy_image)
    cv2.imwrite('1均匀噪声.jpg', uniform_noisy_image)
    cv2.imwrite('1椒盐噪声.jpg', salt_pepper_noisy_image)

    # 仅保存带噪声图像的直方图
    save_histogram(gaussian_noisy_image, '高斯噪声', '高斯噪声')
    save_histogram(uniform_noisy_image, '均匀噪声', '均匀噪声')
    save_histogram(salt_pepper_noisy_image, '椒盐噪声', '椒盐噪声')
