"""
对图片进行HOG特征提取，画出HOG归一化之后的直方图。
"""

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, exposure

# 读取图像（假设已经是灰度图）
image = io.imread('gray.png')

# 提取HOG特征和HOG图像
fd, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    visualize=True)

# 归一化HOG图像
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 1. 保存HOG图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(6, 8))
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('2hog图.png', bbox_inches='tight', pad_inches=0)  # 保存HOG图像
plt.show()

# 2. 保存HOG特征直方图
plt.figure(figsize=(8, 6))
plt.hist(fd, bins=50, color='orange')
plt.savefig('2hog直方图.png')  # 保存直方图
plt.show()
