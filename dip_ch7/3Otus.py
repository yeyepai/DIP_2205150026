"""
使用Otsu方法直接进行图像分割，再按照指定的方法进行图像分割，将两者进行比较分析。
（Otsu方法）
"""

import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 使用Otsu方法进行图像分割
# 0 表示自动选择阈值，cv2.THRESH_BINARY表示进行二值化
_, otsu_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示原图和Otsu分割结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(10, 5))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# 显示Otsu分割结果
plt.subplot(1, 2, 2)
plt.imshow(otsu_result, cmap='gray')
plt.title('Otsu分割结果')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果图像
cv2.imwrite('3Otus结果.png', otsu_result)
