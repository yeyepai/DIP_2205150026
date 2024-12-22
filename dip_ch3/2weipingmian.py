"""
位平面切片
"""

import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# 创建一个列表来存储8个位平面图像
bit_planes = []

# 逐位提取位平面
for i in range(8):
    # 使用位移和按位与操作提取每个位平面
    bit_plane = (image >> i) & 1
    # 将位平面放大到 0-255 范围以便可视化
    bit_plane = bit_plane * 255
    bit_planes.append(bit_plane)

# 颠倒位平面顺序
bit_planes.reverse()

# 显示原图和颠倒顺序后的位平面图像
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 12))

# 显示原图
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原图')
plt.axis('off')

# 显示颠倒顺序后的位平面图像
for i in range(8):
    plt.subplot(3, 3, i + 2)
    plt.imshow(bit_planes[i], cmap='gray')
    plt.title(f'位平面切片后 {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 保存颠倒顺序后的位平面图像
for i in range(8):
    cv2.imwrite(f'2位平面切片_{i+1}.png', bit_planes[i])
