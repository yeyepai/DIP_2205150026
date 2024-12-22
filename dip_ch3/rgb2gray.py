"""
工具：将RGB图像rgb.jpg转为灰度图gray.png，非作业内容
"""

import cv2
import matplotlib.pyplot as plt

# 读取彩色图像
image_color = cv2.imread('rgb.jpg')

# 将彩色图像转换为灰度图像
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# 保存灰度图像
cv2.imwrite('gray.png', image_gray)

# 显示彩色图像和灰度图像
plt.figure(figsize=(10, 5))

# 显示彩色图像
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.title('Color Image')
plt.axis('off')

# 显示灰度图像
plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()
