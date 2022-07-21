# -*- coding: utf-8 -*-
"""
Created on Sat June 23 15:06:11 2022
@author: wzj
python version: python 3.9

Title: 支持向量机（Support Vector Machines）——线性可分案例

案例：使用支持向量机(svm)构建一个垃圾邮件分类器。

数据集：数据文件是ex6data1.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('ex6data1.mat')
print('data.keys():', data.keys())


X, y = data['X'], data['y']
print('X.shape, y.shape:', X.shape, y.shape)
print('X:', X)
print('y:', y)

# ---------------------------
# 画出数据的散点图来看看分布状况
def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')  #　cmap相当于是配色盘的意思，这次选择了jet这一套颜色。y.flatten()
    # 是将y拉伸成一列， 这样每一个X对应一个y,而y只有0，1两种，c是给数据点颜色,把0和1的数据点给出不同的颜色
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

# plot_data()  # 调用函数，画出数据的散点图来看看分布状况
# ---------------------------

# ---------------------------
from sklearn.svm import SVC

svc1 = SVC(C=1, kernel='linear')  # 设置SVC模型参数；C是误差惩罚系数，代替之前使用lamda的方式，用来调节模型方差与偏差的问题; kernel我们
# 暂就用linear
svc1.fit(X, y.flatten())  # 使用SVC对X，y进行拟合预测操作
print(svc1)
print(svc1.predict(X))  # 打印预测结果
print(svc1.score(X, y.flatten()))  # 显示分数，当前预测的准确率为0.90039
# ---------------------------

# ---------------------------
# 绘制决策边界
def plot_boundary(model):
    x_min, x_max = -0.5, 4.5  # 根据散点图看出数据点的x范围
    y_min, y_max = 1.3, 5  # 根据散点图看出数据点的y范围
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))  # np.meshgrid是画格子，这里xx和yy
    # 的shape为(500,500)，意思就是画个均匀的500*500的格子
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])  # 将xx,和yy都从500*500降成一维，再用np.c_[]合并成shape为（250000，2）

    zz = z.reshape(xx.shape)  # (500,500)
    plt.contour(xx, yy, zz)   # 绘制等高线 # 这个等高线暂时不是很理解，这个画法摘自网络

plot_boundary(svc1)  # 调用边界函数画出决策边界
plot_data()    # 调用plot_data() 画出数据的散点图
plt.show()    # 上面两行只是调用，还没有画出来，有了这行之后是把上面两行画出的决策边界和散点图都显示到同一张图上来
# 可以看出C=1时有一个样本点是被错分的
# ---------------------------

# ---------------------------
# 接下来我们换一个C的值来看看预测效果
svc100 = SVC(C=100, kernel='linear')  # C改为100
svc100.fit(X, y.flatten())
print(svc100.predict(X))  # 打印C=100时的预测结果
print(svc100.score(X, y.flatten()))  # 显示分数，当前预测的准确率为1.0

plot_boundary(svc100)  # 调用边界函数画出C=100时的决策边界
plot_data()
plt.show()





