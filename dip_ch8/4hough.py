"""
利用Hough Transform对灰度图像进行检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 对图像进行边缘检测（使用Canny边缘检测）
edges = cv2.Canny(image, 50, 150)

# 使用Hough Transform检测直线
# 降低阈值，阈值设置为50（可以根据需要进行调整）
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# 将原图复制一份来绘制检测到的直线
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 绘制检测到的直线，颜色设置为红色 (0, 0, 255)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.imshow(output_image)
plt.axis('off')  # 关闭坐标轴
plt.show()

cv2.imwrite('4hough变换.png', output_image)
