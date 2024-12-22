"""
灰度级切片
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 灰度级切片：将灰度为100到200的全部设置为150，其他灰度全部设置为25

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)


# 灰度级切片函数
def gray_level_slicing(img, r1, r2, target_level, low_level):
    # 创建输出图像，并初始化为低级别
    sliced_image = np.full_like(img, low_level)

    # 使用灰度级范围进行切片，保留在范围内的像素值并设置为目标级别
    sliced_image[(img >= r1) & (img <= r2)] = target_level

    return sliced_image


# 绘制转换函数 T(r)
def plot_transformation_function(r1, r2, target_level, low_level):
    r_values = np.arange(0, 256)
    T = np.full_like(r_values, low_level)

    # 设置 T(r) 的值
    T[(r_values >= r1) & (r_values <= r2)] = target_level

    plt.figure()
    plt.plot(r_values, T, color='black')
    plt.title(f'转化函数：将 [{r1},{r2}] 转化为 {target_level}')
    plt.xlabel('输入灰度 (r)')
    plt.ylabel('输出灰度 (T(r))')
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.grid(False)
    plt.show()


# 设置灰度级切片范围和目标级别
r1, r2 = 100, 200
target_level = 150
low_level = 25
sliced_image = gray_level_slicing(image, r1, r2, target_level, low_level)

# 显示原始图像和切片后的图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# 切片后的图像
plt.subplot(1, 2, 2)
plt.imshow(sliced_image, cmap='gray')
plt.title('灰度级切片后')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存切片后的图像
cv2.imwrite('1灰度级切片.png', sliced_image)

# 绘制转换函数
plot_transformation_function(r1, r2, target_level, low_level)
