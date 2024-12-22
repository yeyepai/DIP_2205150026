"""
对原图进行一阶锐化处理，从Roberts算子、Sobel算子、Prewitt算子以及Kirsch算子进行选择；对原图进行二阶锐化处理，即拉普拉斯算子；
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转为灰度图
image = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

# Roberts 算子
def roberts_operator(img):
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Sobel 算子
def sobel_operator(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Prewitt 算子
def prewitt_operator(img):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Kirsch 算子
def kirsch_operator(img):
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=int),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=int),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=int),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=int),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=int)
    ]
    result = np.zeros_like(img, dtype=np.float32)
    for kernel in kernels:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        result = np.maximum(result, filtered)
    return cv2.convertScaleAbs(result)

# Laplacian 二阶锐化
def laplacian_operator(img):
    laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(laplacian)

# 显示图像的函数
def show_images(images, titles, rows, cols, fig_num):
    plt.figure(figsize=(12, 8))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 保存图像的函数
def save_images(images, filenames):
    for img, filename in zip(images, filenames):
        cv2.imwrite(filename, img)

# 一阶锐化结果
roberts_img = roberts_operator(image)
sobel_img = sobel_operator(image)
prewitt_img = prewitt_operator(image)
kirsch_img = kirsch_operator(image)

# 二阶锐化结果
laplacian_img = laplacian_operator(image)

# 原图锐化后的结果
roberts_sharpened = cv2.addWeighted(image, 1.5, roberts_img, -0.5, 0)
sobel_sharpened = cv2.addWeighted(image, 1.5, sobel_img, -0.5, 0)
prewitt_sharpened = cv2.addWeighted(image, 1.5, prewitt_img, -0.5, 0)
kirsch_sharpened = cv2.addWeighted(image, 1.5, kirsch_img, -0.5, 0)
laplacian_sharpened = cv2.addWeighted(image, 1.5, laplacian_img, -0.5, 0)

# 显示锐化结果
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'


show_images(
    [image, roberts_sharpened, sobel_sharpened, prewitt_sharpened, kirsch_sharpened, laplacian_sharpened],
    ['原图', 'Roberts锐化', 'Sobel锐化', 'Prewitt锐化', 'Kirsch锐化', 'Laplacian锐化'],
    rows=2,
    cols=3,
    fig_num=1
)

# 显示边缘检测结果
show_images(
    [image, roberts_img, sobel_img, prewitt_img, kirsch_img, laplacian_img],
    ['原图', 'Roberts边缘', 'Sobel边缘', 'Prewitt边缘', 'Kirsch边缘', 'Laplacian边缘'],
    rows=2,
    cols=3,
    fig_num=2
)

# 保存图像到本地
save_images(
    [roberts_img, sobel_img, prewitt_img, kirsch_img, laplacian_img,
     roberts_sharpened, sobel_sharpened, prewitt_sharpened, kirsch_sharpened, laplacian_sharpened],
    ['5Roberts边缘.png', '5Sobel边缘.png', '5Prewitt边缘.png', '5Kirsch边缘.png', '5Laplacian边缘.png',
     '5Roberts锐化.png', '5Sobel锐化.png', '5Prewitt锐化.png', '5Kirsch锐化.png', '5Laplacian锐化.png']
)
