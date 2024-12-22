"""
自拍一张，用Viola Jones进行人脸检测
"""

import cv2
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('face.jpg')

# 转换为灰度图（Viola-Jones算法通常需要灰度图像作为输入）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 加载预训练的人脸检测分类器（Haar Cascade）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 降低 minNeighbors 值，以降低检测阈值
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

# 在检测到的人脸上绘制更粗的矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 50)  # 设置厚度为 4

# 将图像从 BGR 转换为 RGB，以便使用 plt 显示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用 plt 显示图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
plt.axis('off')  # 不显示坐标轴
plt.title("Viola-Jones人脸检测")
plt.show()

# 保存结果图像到本地
cv2.imwrite('5人脸检测.png', image)
