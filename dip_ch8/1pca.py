"""
对图片进行主成分提取，选择合适的特征值个数，并对图片进行恢复，比较两张图片的效果，进行说明。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# 加载灰度图片
image = imread('gray.png', as_gray=True)

# 将图片转换为二维矩阵，记录原始形状
original_shape = image.shape
image_matrix = image.reshape(-1, original_shape[1])

# 标准化处理（零均值化）
mean_vector = np.mean(image_matrix, axis=0)
image_centered = image_matrix - mean_vector

# 计算协方差矩阵
cov_matrix = np.cov(image_centered, rowvar=False)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 特征值排序（降序）
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 选择保留的主成分数
num_components = 100  # 可调整此值，选取合适的主成分个数
selected_eigenvectors = eigenvectors[:, :num_components]

# 降维
low_dim_data = np.dot(image_centered, selected_eigenvectors)

# 恢复图片
reconstructed_image = np.dot(low_dim_data, selected_eigenvectors.T) + mean_vector
reconstructed_image = reconstructed_image.reshape(original_shape)

# 比较两张图片
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("原图")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"恢复后图 ({num_components} 主成分)")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.show()
cv2.imwrite("1恢复后图100.jpg", reconstructed_image)

# 输出信息比例
explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
print(f"{num_components} 个主成分的信息保留率：{explained_variance_ratio[num_components-1]:.2%}")
