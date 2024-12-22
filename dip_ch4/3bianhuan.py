"""
RGB上进行均值滤波以及拉普拉斯变换，仅在HSI的强度分量上进行相同的操作，比较两者的结果。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('rgb.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB

# 定义均值滤波和拉普拉斯锐化函数
def apply_mean_filter(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_laplacian_sharpen(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened_image = image - laplacian  # 将拉普拉斯结果减回原图实现锐化
    sharpened_image = np.clip(sharpened_image, 0, 255)  # 确保值在 [0, 255] 范围内
    return sharpened_image.astype(np.uint8)

# 在 RGB 空间上进行均值滤波和拉普拉斯锐化
mean_filtered_rgb = apply_mean_filter(image)
laplacian_sharpened_rgb = apply_laplacian_sharpen(image)

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

    # 扇区 1：0 <= H < 2π/3
    idx = (H >= 0) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / (np.cos(np.pi / 3 - H[idx])))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # 扇区 2：2π/3 <= H < 4π/3
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx] - 2 * np.pi / 3)) / (np.cos(np.pi - H[idx])))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # 扇区 3：4π/3 <= H < 2π
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx] - 4 * np.pi / 3)) / (np.cos(5 * np.pi / 3 - H[idx])))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb_image = np.stack([R, G, B], axis=-1)
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    return rgb_image

# 对 HSI 空间中的强度分量进行均值滤波和拉普拉斯锐化
H, S, I = rgb_to_hsi(image)
mean_filtered_I = apply_mean_filter((I * 255).astype(np.uint8))
laplacian_sharpened_I = apply_laplacian_sharpen((I * 255).astype(np.uint8))

# 计算 RGB 锐化结果与 HSI 强度分量锐化结果的差异
# difference_image = cv2.absdiff(mean_filtered_rgb, hsi_to_rgb(H, S, mean_filtered_I / 255.0))
difference_image = cv2.absdiff(laplacian_sharpened_rgb, hsi_to_rgb(H, S, laplacian_sharpened_I / 255.0))

# 将差异图像转换为灰度图像
difference_image_gray = cv2.cvtColor(difference_image, cv2.COLOR_RGB2GRAY)

# 转换回 RGB 空间以便显示
mean_filtered_hsi_rgb = hsi_to_rgb(H, S, mean_filtered_I / 255.0)
laplacian_sharpened_hsi_rgb = hsi_to_rgb(H, S, np.clip(laplacian_sharpened_I / 255.0, 0, 1))  # 确保范围在[0, 1]

# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'

# 显示并保存结果
plt.imsave('3RGB均值滤波.png', mean_filtered_rgb)
plt.imsave('3RGB拉普拉斯变换.png', laplacian_sharpened_rgb)
plt.imsave('3HSI强度分量均值滤波.png', mean_filtered_hsi_rgb)
plt.imsave('3HSI强度分量拉普拉斯变换.png', laplacian_sharpened_hsi_rgb)
plt.imsave('3拉普拉斯变换差异.png', difference_image_gray, cmap='gray')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 显示 RGB 空间处理结果
axes[0, 0].imshow(mean_filtered_rgb)
axes[0, 0].set_title('RGB均值滤波')
axes[0, 0].axis('off')

axes[0, 1].imshow(laplacian_sharpened_rgb)
axes[0, 1].set_title('RGB拉普拉斯变换')
axes[0, 1].axis('off')

# 显示 HSI 强度分量处理结果
axes[1, 0].imshow(mean_filtered_hsi_rgb)
axes[1, 0].set_title('HSI强度分量均值滤波')
axes[1, 0].axis('off')

axes[1, 1].imshow(laplacian_sharpened_hsi_rgb)
axes[1, 1].set_title('HSI强度分量拉普拉斯变换')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
