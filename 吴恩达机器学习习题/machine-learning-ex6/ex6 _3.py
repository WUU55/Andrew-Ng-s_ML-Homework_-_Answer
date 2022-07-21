# -*- coding: utf-8 -*-
"""
Created on Sat June 23 18:01:23 2022
@author: wzj
python version: python 3.9

Title: 支持向量机（Support Vector Machines）——寻找最优参数C和gamma

案例：使用支持向量机(svm)构建一个垃圾邮件分类器。

数据集：数据文件是ex6data3.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # 导入sklearn.svm的库

data = sio.loadmat('ex6data3.mat')
print('data.keys():', data.keys())


X, y = data['X'], data['y']
Xval, yval = data['Xval'], data['yval']
print('X.shape, y.shape:', X.shape, y.shape)

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
#　寻找准确率最高时候的最优参数C和gamma
Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # 设置9个误差惩罚系数的候选值C
gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # 设置9个高斯核的参数候选值gamma

best_score = 0  # 设置初始得分（即预测准确率）
best_params = (0, 0)  # 设置初始参数

for c in Cvalues:  # 遍历Cvalues中的候选值
    for gamma in gammas:  # 遍历gammas中的候选值
        svc = SVC(C=c, kernel='rbf', gamma=gamma)   # 将候选值一个一个代入SVC中
        svc.fit(X, y.flatten())
        score = svc.score(Xval, yval.flatten())   # 将svc后的结果代入验证集中，对Xval,yval进行验证，显示预测准确率，
        if score > best_score:  # 如果当前的分数score大于之前的历史最好分数best_score
            best_score = score  # 就将当前的分数score赋值成历史最好分数best_score
            best_params = (c, gamma)  # 并且把当前的参数c和gamma赋值成历史最好参数best_params

print('best_score, best_params:', best_score, best_params)
# ---------------------------

# ---------------------------
# 将最优参数代回去，得到最后的最优分类图像
svc2 = SVC(C=best_params[0], kernel='rbf', gamma=best_params[1])  # 其实C和gamma分别就是0.3和100
svc2.fit(X, y.flatten())

# 绘制决策边界
def plot_boundary(model):
    x_min, x_max = -0.6, 0.4  # 根据散点图看出数据点的x范围
    y_min, y_max = -0.7, 0.7 # 根据散点图看出数据点的y范围
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))  # np.meshgrid是画格子，这里xx和yy
    # 的shape为(500,500)，意思就是画个均匀的500*500的格子
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])  # 将xx,和yy都从500*500降成一维，再用np.c_[]合并成shape为（250000，2）

    zz = z.reshape(xx.shape)  # (500,500)
    plt.contour(xx, yy, zz)   # 绘制等高线 # 这个等高线暂时不是很理解，这个画法摘自网络

plot_boundary(svc2)  # 调用边界函数画出决策边界
plot_data()    # 调用plot_data() 画出数据的散点图
plt.show()    # 上面两行只是调用，还没有画出来，有了这行之后是把上面两行画出的决策边界和散点图都显示到同一张图上来

