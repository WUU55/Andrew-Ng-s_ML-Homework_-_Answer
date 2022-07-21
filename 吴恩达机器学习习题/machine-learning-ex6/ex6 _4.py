# -*- coding: utf-8 -*-
"""
Created on Sat June 23 18:01:23 2022
@author: wzj
python version: python 3.9

Title: 支持向量机（Support Vector Machines）——判断一封邮件是否是垃圾邮件

案例：使用支持向量机(svm)构建一个垃圾邮件分类器。

训练集是：spamTrain.mat
测试集是：spamTest.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # 导入sklearn.svm的库

# Training data
data1 = sio.loadmat('spamTrain.mat')
print('data1.keys():', data1.keys())


# Testing data
data2 = sio.loadmat('spamTest.mat')
print('data2.keys():', data2.keys())

X, y = data1['X'], data1['y']
Xtest, ytest = data2['Xtest'], data2['ytest']
print('X.shape, y.shape:', X.shape, y.shape)
print('X:', X)  #　X由1899种特征来表示，这些特征由0和1组成，0表示语义库不能找到该单词，1表示语义库可以找到该单词
print('y:', y)  #　y只有0和1两种形式，1表示当前邮件为垃圾邮件，0表示不是垃圾邮件
# ---------------------------

# ---------------------------
# 调用SVC进行邮件分类
Cvalues = [3, 10, 30, 100, 0.01, 0.03, 0.1, 0.3, 1]  # 设置9个误差惩罚系数的候选值C
best_score = 0  # 设置初始得分（即预测准确率）
best_param = 0  # 设置初始参数

for c in Cvalues:  # 遍历Cvalues中的候选值
    svc = SVC(C=c, kernel='linear')   # 将候选值一个一个代入SVC中,kernel采用线性分类linear
    svc.fit(X, y.flatten())
    score = svc.score(Xtest, ytest.flatten())   # 将svc后的结果代入验证集中，对Xval,yval进行验证，显示预测准确率，
    if score > best_score:  # 如果当前的分数score大于之前的历史最好分数best_score
        best_score = score  # 就将当前的分数score赋值成历史最好分数best_score
        best_param = c  # 并且把当前的参数c和gamma赋值成历史最好参数best_params

print('best_score, best_param:', best_score, best_param)


# 将最优的参数best_param分别带入训练集和测试集，得到在各自数据下的最好分数（预测准确率）
svc = SVC(C= best_param, kernel='linear')
svc.fit(X, y.flatten())
score_train = svc.score(X, y.flatten())
score_test = svc.score(Xtest, ytest.flatten())
print('score_train, score_test:', best_score, best_param)