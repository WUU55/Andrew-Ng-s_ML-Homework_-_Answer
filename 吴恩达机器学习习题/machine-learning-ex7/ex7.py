# -*- coding: utf-8 -*-
"""
Created on Mon June 27 13:15:53 2022
@author: wzj
python version: python 3.9

Title: K-means聚类与主成分分析

案例：给定一个二维数据集，使用K-means算法进行聚类

数据集：ex7data2.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data1 = sio.loadmat('ex7data2.mat')
print('data1.keys():', data1.keys())
# 会发现数据集里只有X，没有标签y了

X = data1['X']
print('X.shape', X.shape)  # X是一个有300行（样本），两列（特征）的数据

plt.scatter(X[:, 0], X[:, 1])
plt.show()
# ---------------------------

# ---------------------------
# 获取每个样本所属的类别
def find_centrolds(X, centros):
    idx = []

    for i in range(len(X)):
        # X是一个一维数组，centros是一个二维数组（k,2），k表示类别数；
        # print('X[i]:', X[i])
        dist = np.linalg.norm((X[i] - centros), axis=1)  # axis=1是按列；# np.linalg.norm线性带代数库的norm二范数，其实就是开方；
        # 得到每一个i下对应得有三个dist
        # print('dist:', dist)
        # print('dist.shape:', dist.shape)
        id_i = np.argmin(dist)  # 找到dist里当前i时最小的,让它等于id_i
        idx.append(id_i)  # 再将id_i给append到idx里

    return np.array(idx)   # 将idx以数组形式返回

centros = np.array([[3, 3],[6, 2], [8, 5]])  # 给定一个三行两列的centros，表示为将我们的数组化为一个3维的标签，标签值分别为0，1，2；
# 其实就是数据会聚成三个类，这三个数据是up主自己随便选择的
idx = find_centrolds(X, centros)  # 调用find_centrolds函数
print('idx[:3]:', idx[:3])
# ---------------------------

# ---------------------------
# 计算聚类中心点
def compute_centros(X, idx, k):
    centros = []

    for i in range(k):
        centros_i = np.mean(X[idx == i], axis=0)  # aixs=0按行
        centros.append(centros_i)

    return np.array(centros)

print('compute_centros(X, idx, k=3)', compute_centros(X, idx, k=3)) # 调用程序，打印出来
# ---------------------------

# ---------------------------
# 运行kmeans,重复执行1和2
def run_kmeans(X, centros, iters):

    k = len(centros)
    centros_all = []
    centros_all.append(centros)
    centros_i = centros

    for i in range(iters):
        idx = find_centrolds(X, centros_i)
        centros_i =compute_centros(X, idx, k)
        centros_all.append(centros_i)

    return idx, np.array(centros_all)
# ---------------------------

# ---------------------------
# 为了更直观的看到变化过程，绘制数据集和聚类中心的移动轨迹
def plot_data(X, centros_all, idx):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='rainbow')  # cmap是配色盘
    plt.plot(centros_all[:, :, 0], centros_all[:, :, 1], 'kx--')  # centros_all是一个三维数组，第一维度是迭代次数，然后是类别数，特征数

idx, centros_all = run_kmeans(X, centros, iters=10)
plot_data(X, centros_all, idx)
plt.show()

# ---------------------------

# ---------------------------
# 观察初始聚类点的位置对聚类效果的影响
def init_centros(X, k):
    index = np.random.choice(len(X), k)  # 从len(X)个数据里，随机取k（3）个整数，作为index
    return X[index]  # 将随机取的3个index用于给X随机取数，就得到了从X里随机取出的3个初始聚类点

print('init_centros(X,k=3):', init_centros(X,k=3))  # 调用init_centros函数，打印三个初始聚类点

for i in range(4):
    idx, centros_all = run_kmeans(X, init_centros(X, k=3), iters=10)
    plot_data(X, centros_all, idx)
    plt.show()
