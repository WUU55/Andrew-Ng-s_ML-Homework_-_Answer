# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:53:02 2022
@author: wzj
python version: python 3.9

Title: logistic regression with Binary classification
建立一个逻辑回归模型来预测一个学生是否被大学录取。根据两次考试的结果来决定每个申请人的录取机会。有以前的申请人的历史数据，可以用它作为透辑回归的训练集

python实现逻辑回归目标:建立分类器(求解出三个参数theta0 theta1 theta2)即得出分界线,备注:theta1对应'Exam 1';成绩theta2对应'Exam 2'设定阈
值，根据阈值判断录取结果;备注:阈值指的是最终得到的概率值.将概率值转化成一个类别.一般是>0.5是被录取了,<0.5未被录取.实现内容:
sigmoid :映射到概率的函数; model :返回预测结果值; cost:根据参数计算损失; gradient:计算每个参数的梯度方向; descent:进行参数更新;
acuracy:计算精度.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn与matplotlib在功能上是互补关系
plt.style.use('fivethirtyeight')  # 样式美化
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report  # 这个包是评价报告

# 照常，先读取数据观察一下
path = 'ex2data1.txt'
data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])
print(data.head())
print(data.describe())


# # ---------------------------
'''第一种画图的方式'''
# sns.set(context='notebook', style='darkgrid', palette=sns.color_palette('RdBu', 2))  # 设置样式参数，默认主题darkgrid(灰色背景+白网格)，调色板
#
# sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data,
#            height=6,     # height参数是每个构面的高度(以英寸为单位)
#            fit_reg=False,  # fit_reg参数，控制是否显示拟合的直线
#            scatter_kws={'s': 50}  # hue参数是将name所指定的不同类型的数据（即admitted下有0有1两种不同数据）叠加在一张图中显示
#            )  # scatter_kws不知道是干嘛的，设置为 scatter_kws=None感觉画出的图也没啥变化
# plt.show()
# # ---------------------------


# '''第二种画图的方式'''
# # 将label为0和1的放在两个dataframe中，由于这里只有0和1，所以这里用==1或者==2也行
# # 这里positive和negative并没有重置dataframe的索引，索引还是根据原始的data变量里的
# positive = data[data['admitted'].isin([1])]
# negative = data[data['admitted'].isin([0])]
# # positive = positive.reset_index(drop=True)
# # iloc索引以行号为索引，含头不含尾，loc索引以index或者说标签为索引，含头含尾！
# # print(positive.iloc[0:1, :])
# # 画个图看一下
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='admitted')  # s=数量, c=颜色,
# #marker=点的形状
# ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not admitted')
# ax.legend()
# ax.set_xlabel('exam1 Score')
# ax.set_ylabel('exam2 Score')
# # plt.show()
# # ---------------------------


# ---------------------------
'''第三种画图的方式'''
# fig, ax = plt.subplots(figsize=(12, 8))  # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
# ax.scatter(data[data['admitted'] == 0]['exam1'], data[data['admitted'] == 0]['exam2'], c='r', marker='x', label='y=0')
# 将'adimitted'这列是0的点拎出来，画一个散点图
# ax.scatter(data[data['admitted'] == 1]['exam1'], data[data['admitted'] == 1]['exam2'], c='b', marker='o', label='y=1')
# 将'adimitted'这列是1的点拎出来，画一个散点图
# ax.legend(loc=1) # 标签位置默认
# ax.set(xlabel='exam1',
#        ylabel='exam2')
# plt.show()
# ---------------------------


# ---------------------------
# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))

# 实现逻辑回归的代价函数，两个部分，-y(log(hx)和-(1-y)log(1-hx)
def computeCost(X, y, theta):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T) + 1e-5))  # 结果是个（100，1）矩阵
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T) + 1e-5))  # 结果也是个（100，1）矩阵
    return np.sum(first - second) / (len(X))
# ---------------------------


# ---------------------------
# 初始化theta、X、y
data.insert(0, 'ones', 1)# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。在训练集的左侧插入一列全为1的列，以便
# 计算即x0=1 loc为O, name为ones,value为1.
# 看下data共有多少列
cols = data.shape[1]  # 表示得到data一共有4列，即cols=4
print("看看有多少列:",  cols)
# pandas中利用.iloc选取数据loc; ','前的部分标明选取的行，','后的部分标明选取的列。
X = data.iloc[:, : cols-1]
y = data.iloc[:, cols-1: cols]
theta = np.full((1, X.shape[1]), 0)    # np.full((行数，列数)， 想要填充的数字)
print(theta, "theta01")
print(theta.shape, 'theta.shape01')

# 打印表头，观察下X（训练集）和y(目标变量)是否正确
print(X.head())
print(y.head())

# 将X和y转换为矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)

print(computeCost(X, y, theta), "cost1")
print(theta, "theta")
# ---------------------------


# ---------------------------
# 2.batch gradient decent(梯度下降函数)
# 关键点在于theta0和thata1要同时更新，需要用temp进行临时存储
def gradientDescent(X, y, theta, alpha, iters):
    # print(X, 'X')
    # print(X * theta.T, "X * theta.T")
    # print(y, 'y')
    # print(sigmoid(X @ theta.T) - y, 'error')
    # print(X[:, 1], "X[:, 1]")
    # print(np.multiply(sigmoid(X @ theta.T) - y, X[:, 1]), 'term')
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = sigmoid(X @ theta.T) - y  # 计算出每一组数据的误差，每行都是样本的误差

        for j in range(parameters):
            term = np.multiply(error, X[:, j])   # X[ : , j]的意思是X[所有行，第j列]
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))   # 前面转不转置和上课学的公式不太一样，这没关系，因为书
            # 上的转置是考虑了求和，我们这里的转置是为了变成可以相乘的方块，至于求和最后用了sum直接求

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
# ---------------------------


# ---------------------------
# 初始化学习速率和迭代次数
alpha = 0.004  # 看up主ladykaka007设置的
iters = 200000  # 看up主ladykaka007设置的; 设置为400000迭代曲线效果就很明显

g, cost = gradientDescent(X, y, theta, alpha, iters)

print("g:", g)
print("theta:", theta)
print("cost", cost)

minCost = computeCost(X, y, g)
print("minCost:", minCost)   # 和up主ladykaka007的结果差不多，她是minCost0.26，我是0.25839
# ---------------------------


# 预测函数
# ---------------------------
def predict(X, theta):
    prob = sigmoid(X @ theta.T)  # 结果prob是一个（100，1）矩阵
    return [1 if x >= 0.5 else 0 for x in prob]  # 遍历prob这个（100，1）矩阵，如果其中数大于0.5，就预测于1，即为通过exam;否则为0，
    # 即未通过exam

y_=np.array(predict(X, g))  # 将最终的g(即最优值theta)代入X*theta.T,让矩阵X与最优的theta挨个乘，得到一个(100, 1)的矩阵
y_pre = y_.reshape(len(y_), 1)  # .reshape(a,b)表示以a行b列的数组形式显示，即将y_表示为(100，1)矩阵形式（虽然本来就长这样,但将它从数组变成了矩阵？）
# print("y_", y_)
# print("y_.shape:", y_.shape)
# print("y_pre:", y_pre)
# print("y_pre.shape:", y_pre.shape)
admi = np.mean(y_pre == y)  # x == y表示两个数组中的值相同时，输出True；否则输出False，其中True=1,False=0;即可得到一个（100，1）的矩阵，
# 其中预测值的y_pre与真实y之间吻合的那行就是1，不吻合的那行就是0。np.mean()为计算矩阵的均值，即为正确率

print("预测正确的概率：", admi)
# ---------------------------

# ---------------------------
# 画决策边界
coef1 = -g[0, 0] / g[0, 2]
coef2 = -g[0, 1] / g[0, 2]

x = np.linspace(20, 100, 100) #  # 从最小数据到最大数据（即所有数据）中随机抽100个数据点，用np.linspace可以将这些点用等差数列排列出来，即
# 得到一个个间隔均匀的x点
f = coef1 + coef2 * x
fig, ax = plt.subplots(figsize=(12, 8))  # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
ax.scatter(data[data['admitted'] == 0]['exam1'], data[data['admitted'] == 0]['exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['admitted'] == 1]['exam1'], data[data['admitted'] == 1]['exam2'], c='b', marker='o', label='y=1')
ax.legend(loc=1) # 标签位置默认
ax.set(xlabel='exam1',
       ylabel='exam2')
ax.plot(x, f, c='g')
plt.show()  # 数图上分错的点有9个，总共100个点，说明正确率有91%，和弹幕大家一样


# ---------------------------
# 画迭代曲线，但是此次的曲线是震荡的，很密集的上下的，所以看起来是一大块红色，当迭代次数为400000时曲线比较明显
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')  # np.arange()自动返回等差数组
# print(np.arange(iters), "np.arange(iters)")
# print(cost, "cost_pic")
# print(cost.shape[0], "cost_pic")
ax.set_xlabel('Iterations')  # 设定x轴名称
ax.set_ylabel('Cost')  # 设定y轴名称
ax.set_title('Error vs. Training Epoch')  # 设定整个表的名称
plt.show()














