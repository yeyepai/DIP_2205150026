"""
对图片进行Harris角点检测。
"""

import matplotlib.pyplot as plt
from skimage.feature import corner_harris, corner_peaks
from skimage.io import imread

# 加载灰度图像
image = imread('gray.png', as_gray=True)

# 进行 Harris 角点检测
corners = corner_harris(image)

# 设置角点响应的阈值（较大的响应值表示角点较强）
threshold = 0.05 * corners.max()  # 阈值设置为最大值的 5%
corner_positions = corner_peaks(corners, min_distance=5, threshold_abs=threshold)

# 可视化结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))
plt.imshow(image, cmap='gray')
plt.scatter(corner_positions[:, 1], corner_positions[:, 0], color='red', marker='x')

# 去除标题和坐标轴
plt.axis('off')

# 保存图像到本地
plt.savefig('3harris角点.png', bbox_inches='tight', pad_inches=0)

# 显示图像
plt.show()
