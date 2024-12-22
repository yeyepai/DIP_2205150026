"""
分别在RGB和HSI空间上进行直方图均衡化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('rgb.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'

# 在 RGB 空间上进行直方图均衡化
equalized_rgb = image.copy()
for i in range(3):  # 对 R、G、B 三个通道分别进行处理
    equalized_rgb[:, :, i] = cv2.equalizeHist(image[:, :, i])

# 将 RGB 图像转换为 HSI
def rgb_to_hsi(image):
    img = image.astype(float) / 255
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    I = (R + G + B) / 3
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb  # 加 1e-6 防止除以零

    theta = np.arccos(0.5 * ((R - G) + (R - B)) / np.sqrt((R - G) ** 2 + (R - B) * (G - B) + 1e-6))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # 归一化到 [0, 1] 范围

    return H, S, I

# 将 HSI 图像转换回 RGB
def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    idx = (H >= 0) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / (np.cos(np.pi / 3 - H[idx])))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx] - 2 * np.pi / 3)) / (np.cos(np.pi - H[idx])))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx] - 4 * np.pi / 3)) / (np.cos(5 * np.pi / 3 - H[idx])))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb_image = np.stack([R, G, B], axis=-1)
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    return rgb_image

# 对 HSI 空间中的 H、S、I 分量分别进行直方图均衡化
H, S, I = rgb_to_hsi(image)
H_equalized = cv2.equalizeHist((H * 255).astype(np.uint8)) / 255.0
S_equalized = cv2.equalizeHist((S * 255).astype(np.uint8)) / 255.0
I_equalized = cv2.equalizeHist((I * 255).astype(np.uint8)) / 255.0

# 转换回 RGB 空间
equalized_hsi_rgb = hsi_to_rgb(H_equalized, S_equalized, I_equalized)

# 保存均衡化后的 RGB 图像
cv2.imwrite('2RGB均衡化.png', cv2.cvtColor(equalized_rgb, cv2.COLOR_RGB2BGR))

# 保存均衡化后的 HSI 图像
cv2.imwrite('2HSI均衡化.png', cv2.cvtColor(equalized_hsi_rgb, cv2.COLOR_RGB2BGR))

# 绘制 RGB 空间中 R、G、B 分量的直方图均衡化前后的对比
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    axes[0, i].hist(image[:, :, i].ravel(), bins=256, color=color, alpha=0.6)
    axes[0, i].set_title(f'原{color.upper()}分量直方图')
    axes[0, i].get_yaxis().set_visible(False)  # 隐藏纵轴坐标
    axes[0, i].get_xaxis().set_visible(False)  # 隐藏横轴坐标
    axes[1, i].hist(equalized_rgb[:, :, i].ravel(), bins=256, color=color, alpha=0.6)
    axes[1, i].set_title(f'均衡化后{color.upper()}分量直方图')
    axes[1, i].get_yaxis().set_visible(False)  # 隐藏纵轴坐标
    axes[0, i].get_xaxis().set_visible(False)  # 隐藏横轴坐标


plt.tight_layout()
plt.savefig('2RGB分量直方图.png')
plt.show()

# 绘制 HSI 空间中 H、S、I 分量的直方图均衡化前后的对比
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
components = [('H', H, H_equalized), ('S', S, S_equalized), ('I', I, I_equalized)]
for i, (name, original, equalized) in enumerate(components):
    axes[0, i].hist(original.ravel(), bins=256, color=colors[i], alpha=0.6)
    axes[0, i].set_title(f'原{name}分量直方图')
    axes[0, i].get_yaxis().set_visible(False)  # 隐藏纵轴坐标
    axes[0, i].get_xaxis().set_visible(False)  # 隐藏横轴坐标
    axes[1, i].hist(equalized.ravel(), bins=256, color=colors[i], alpha=0.6)
    axes[1, i].set_title(f'均衡化后{name}分量直方图')
    axes[1, i].get_yaxis().set_visible(False)  # 隐藏纵轴坐标
    axes[0, i].get_xaxis().set_visible(False)  # 隐藏横轴坐标


plt.tight_layout()
plt.savefig('2HSI分量直方图.png')
plt.show()
