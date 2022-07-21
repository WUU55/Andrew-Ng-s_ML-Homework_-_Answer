# -*- coding: utf-8 -*-
"""
Created on Mon June 27 18:12:45 2022
@author: wzj
python version: python 3.9

Title: K-means聚类与主成分分析

案例：使用kmeans对图片颜色进行聚类

## 原始图片为RGB图像，每个像素点范围0~255
数据集：bird_small.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


data = sio.loadmat('bird_small.mat')
print('data.keys():', data.keys())
# 会发现数据集里只有A，没有标签X了

A = data['A']
print('A.shape', A.shape)  # X是一个有300行（样本），两列（特征）的数据
# ---------------------------
from skimage import io
from ex7 import run_kmeans
from ex7 import init_centros
image = io.imread('bird_small.png')  # 读取png格式图片
plt.imshow(image)   # 显示image
plt.show()

A = A/255  # 让所有像素点的值都变成0~1
A = A.reshape(-1, 3)

k = 16
idx, centros_all = run_kmeans(A, init_centros(A, k=16), iters=20)
centros = centros_all[-1]
im = np.zeros(A.shape)

for i in range(k):
    im[idx == i] = centros[i]

im = im.reshape(128, 128, 3)
plt.imshow(im)
plt.show()





