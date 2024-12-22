"""
灰度图片运动模糊并加上高斯噪声后，分别用维纳滤波以及约束最小二乘方滤波进行恢复。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def getMotionDsf(shape, angle, dist):
    xCenter = (shape[0] - 1) / 2
    yCenter = (shape[1] - 1) / 2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    PSF = np.zeros(shape)  # 点扩散函数
    for i in range(dist):  # 将对应角度上motion_dis个点置成1
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
    return PSF / PSF.sum()  # 归一化

def makeBlurred(image, PSF, eps):  # 对图片进行运动模糊
    fftImg = np.fft.fft2(image)  # 进行二维数组的傅里叶变换
    fftPSF = np.fft.fft2(PSF) + eps
    fftBlur = np.fft.ifft2(fftImg * fftPSF)
    fftBlur = np.abs(np.fft.fftshift(fftBlur))
    return fftBlur

def wienerFilter(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    fftImg = np.fft.fft2(input)
    fftPSF = np.fft.fft2(PSF) + eps
    fftWiener = np.conj(fftPSF) / (np.abs(fftPSF)**2 + K)
    imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
    imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
    return imgWienerFilter

def getPuv(image):
    h, w = image.shape[:2]
    hPad, wPad = h - 3, w - 3
    pxy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    pxyPad = np.pad(pxy, ((hPad//2, hPad - hPad//2), (wPad//2, wPad - wPad//2)), mode='constant')
    fftPuv = np.fft.fft2(pxyPad)
    return fftPuv

def leastSquareFilter(image, PSF, eps, gamma=0.01):  # 约束最小二乘方滤波
    fftImg = np.fft.fft2(image)
    fftPSF = np.fft.fft2(PSF)
    conj = fftPSF.conj()
    fftPuv = getPuv(image)
    # absConj = np.abs(fftPSF) ** 2
    Huv = conj / (np.abs(fftPSF)**2 + gamma * (np.abs(fftPuv)**2))
    ifftImg = np.fft.ifft2(fftImg * Huv)
    ifftShift = np.abs(np.fft.fftshift(ifftImg))
    imgLSFilter = np.uint8(cv2.normalize(np.abs(ifftShift), None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgLSFilter

# 读取原始图像
img = cv2.imread("gray.png", 0)  # flags=0 读取为灰度图像
hImg, wImg = img.shape[:2]

# 带有噪声的运动模糊
PSF = getMotionDsf((hImg, wImg), 45, 100)  # 运动模糊函数
imgBlurred = np.abs(makeBlurred(img, PSF, 1e-6))  # 生成不含噪声的运动模糊图像

# 添加高斯噪声
scale = 0.01  # 噪声方差
noisy = imgBlurred.std() * np.random.normal(loc=0.0, scale=scale, size=imgBlurred.shape)  # 添加高斯噪声
imgBlurNoisy = imgBlurred + noisy  # 带有噪声的运动模糊

# 维纳滤波恢复
imgWienerFilter = wienerFilter(imgBlurNoisy, PSF, scale, K=0.01)

# 约束最小二乘方滤波恢复
imgLSFilter = leastSquareFilter(imgBlurNoisy, PSF, scale, gamma=0.01)

# 保存三张图像
cv2.imwrite("3模糊+噪声.png", np.uint8(imgBlurNoisy))  # 保存模糊+噪声图
cv2.imwrite("3维纳滤波恢复.png", np.uint8(imgWienerFilter))  # 保存维纳滤波恢复图
cv2.imwrite("3CLS恢复.png", imgLSFilter)  # 保存约束最小二乘方恢复图

# 绘制原图、模糊+噪声图、维纳恢复图、约束最小二乘恢复图
# plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(10, 10))

# 第一行：原图，模糊图（带噪声）
plt.subplot(221), plt.title("原图"), plt.axis('off'), plt.imshow(img, cmap='gray')
plt.subplot(222), plt.title("模糊+噪声"), plt.axis('off'), plt.imshow(imgBlurNoisy, cmap='gray')

# 第二行：维纳滤波恢复图，约束最小二乘方滤波恢复图
plt.subplot(223), plt.title("维纳滤波恢复图"), plt.axis('off'), plt.imshow(imgWienerFilter, cmap='gray')
plt.subplot(224), plt.title("约束最小二乘方滤波恢复图"), plt.axis('off'), plt.imshow(imgLSFilter, cmap='gray')

plt.tight_layout()
plt.show()
