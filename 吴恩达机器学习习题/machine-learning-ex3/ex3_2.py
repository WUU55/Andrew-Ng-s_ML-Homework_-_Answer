# -*- coding: utf-8 -*-
"""
Created on Wed June 8 14:23:14 2022
@author: wzj
python version: python 3.9

Title: 利用神经网络进行多层次分类（Multi-class Classification with the neural network）
对于此练习，我们将使用逻辑回归来识别手写数字(0到9)。我们将扩展我们在练习2中写的逻辑回归的实现，并将其应用于一对一的分类。

数据集：和之前的数据集一样，数据文件是ex3data1.mat
权重集：theta的权重，数据文件是ex3weights.mat
"""

import numpy as np
import scipy.io as sio

data = sio.loadmat('ex3data1.mat')
raw_X = data['X']
raw_y = data['y']

X = np.insert(raw_X, 0, values=1, axis=1)  # 为了让X能与下面的theta1矩阵相乘，给X插入一列，loc=0,值全是1，这样X的列数就与theta1的行数
# 相同，两矩阵就能做乘法运算了
print('X.shape:', X.shape)
y = raw_y.flatten()  # 把y去掉一个维度,原本y=([[1],[2],[3]...]).T的5000行1列的的样式，现在变成了y=([1,2,3....])的数组样式，从2维到1维
print('y.shape:', y.shape)

theta = sio.loadmat('ex3weights.mat')  # 这个文件数据格式是一个字典dict
print('theta.keys():', theta.keys())
theta1 = theta['Theta1']
theta2 = theta['Theta2']
print('theta1.shape:', theta1.shape)
print('theta2.shape:', theta2.shape)
# ---------------------------


# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# ---------------------------


# ---------------------------
a1 = X

# 计算z2, a2
z2 = X @ theta1.T
a2 = sigmoid(z2)
print('a2.shape:', a2.shape)

a2 = np.insert(a2, 0, values=1, axis=1)  # 对a2添加一列全为1的值,(X, 索引=0， 值=0， axis是按行按行插入（0：行、1：列）)
print('a2.shape:', a2.shape)


# 计算z3, a3
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
print('a3.shape:', a3.shape)
# ---------------------------


# ---------------------------
# 求准确率
y_pred = np.argmax(a3, axis=1)  # arg是np中的比较函数，max是选大的意思；axis=0是行，axis=1就是将对同一行的每一列去比较，哪一个比较
    # 大我们就将它的索引返回回来，比如第一个位置的结果最大，就返回它的索引，但结果是0，所以要对返回的索引加1
y_pred = y_pred + 1
acc = np.mean(y_pred == y)  # 计算准确率accuracy; 求预测值y_pred等于真实值y的个数，再求平均值
print('acc:', acc)






