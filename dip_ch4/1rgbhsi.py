"""
给出一张彩色图片的RGB以及HSI分量图
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('rgb.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 使用 BGR 读取图片，所以需要转换为 RGB

# 分离 RGB 分量
R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]


# 将 RGB 转换为 HSI
def rgb_to_hsi(image):
    # 归一化 RGB 值
    img = image.astype(float) / 255
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 计算 I 分量
    I = (R + G + B) / 3

    # 计算 S 分量
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb  # 加 1e-6 防止除以零

    # 计算 H 分量
    theta = np.arccos(0.5 * ((R - G) + (R - B)) / np.sqrt((R - G) ** 2 + (R - B) * (G - B) + 1e-6))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # 归一化到 [0, 1] 范围

    return H, S, I


H, S, I = rgb_to_hsi(image)

# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'

# 显示 RGB 和 HSI 分量图
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 显示 RGB 分量
axes[0, 0].imshow(R, cmap='Reds')
axes[0, 0].set_title('RGB R分量')
axes[0, 0].axis('off')

axes[0, 1].imshow(G, cmap='Greens')
axes[0, 1].set_title('RGB G分量')
axes[0, 1].axis('off')

axes[0, 2].imshow(B, cmap='Blues')
axes[0, 2].set_title('RGB B分量')
axes[0, 2].axis('off')

# 显示 HSI 分量
axes[1, 0].imshow(H, cmap='hsv')
axes[1, 0].set_title('HSI H分量')
axes[1, 0].axis('off')

axes[1, 1].imshow(S, cmap='gray')
axes[1, 1].set_title('HSI S分量')
axes[1, 1].axis('off')

axes[1, 2].imshow(I, cmap='gray')
axes[1, 2].set_title('HSI I分量')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# 保存 RGB 分量图
plt.imsave('1RGB_R分量.png', R, cmap='Reds')
plt.imsave('1RGB_G分量.png', G, cmap='Greens')
plt.imsave('1RGB_B分量.png', B, cmap='Blues')

# 保存 HSI 分量图
plt.imsave('1HSI_H分量.png', H, cmap='hsv')
plt.imsave('1HSI_S分量.png', S, cmap='gray')
plt.imsave('1HSI_I分量.png', I, cmap='gray')
