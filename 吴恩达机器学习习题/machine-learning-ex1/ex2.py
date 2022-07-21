# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:53:02 2022
@author: wzj
python version: python 3.9

Title: linear regression with multiple variable
在这一部分中我们有带有两个变量的值（房屋大小和卧室数量）来估计房屋价格。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None) # 删除names=['Size', 'Bedrooms', 'Price']之后，它会自动给每列命名为0，1，2....
print(data2.head())  # 预览数据

# 此时由于特征值大小不同需要进行均值归一化
# 如果这个房子价格不归一化，它的数量级和你输入值规一化数量级差别太大，几十万的数量级和个位小数做回归，就不能保证收敛了预测
# 的y和实际上y几十万差的太多了
data2 = (data2 - data2.mean()) / data2.std()  # 这是标准通用的均值归一化公式
print(data2.head())

# print(data.describe())  # 对于数值数据，结果的索引将包括计数，平均值，标准差，最小值，最大值以及较低的百分位数和5O。默认情况下，
# 较低的百分位数为25，较高的百分位数为75.50百分位数与中位数相同。

# print("运行结束!")


# ---------------------------
# 现在让我们使用梯度下降来实现线性回归。以最小化成本函数
def computeCost(X2, y2, theta):
    inner = np.power(((X2 * theta.T) - y2), 2)  # 即成本函数公式求和符号右边；.T是转置矩阵的意思
    return np.sum(inner) / (2 * len(X2))  # np.sum()可以将矩阵里的元素全部加起来，即成本函数公式求和符号与（1/2m）
# return将它后面的值/结果返回给定义的函数，不然打印函数名出来是空值；有了return之后，所以下面打印函数名才能出现值/结果


# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。在训练集的左侧插入一列全为1的列，以便计算即x0=1 loc为O, name
# 为ones,value为1.
data2.insert(0, 'ones', 1)

# 现在我们来做一些变量初始化。.shope会得到该矩阵共有（几行，几列）.shape[0]为第一维的长度，shape[1]为第二维的长度，即列，
# pandas中利用.iloc选取数据loc; ','前的部分标明选取的行，','后的部分标明选取的列， 此时三列了
# set X (training data) and y (target variable)
cols = data2.shape[1]  # 表示得到第二维（行）一共有3列，即cols=3
X2 = data2.iloc[ : , 0:cols-1]  # [X是所有行， y是从0到cols-1列（即第一、二列）]
y2 = data2.iloc[ : , cols-1: cols]  # [X是所有行， y是从cols-1到cols列（即第三列）]

# 打印表头，观察下X（训练集）和y(目标变量)是否正确
print(X2.head())
print(y2.head())

# 代价函数是应该是numpy矩阵，所以我们需要转换X和y，然后才能使用它们。我们还需要初始化theta，即把theta所有元素都设置为0.
# pandas读取的数据是dataframe的形式，优势是可以对数据进行很多操作,但是要想进行矩阵运算，要将dataframe形式转换为矩阵，例如x = x.values
X2 = np.matrix(X2.values)  # 将X转换为矩阵
y2 = np.matrix(y2.values)  # 将y转换为矩阵
# theta = np.matrix(np.array([0, 0, 0]))  # 因为X变成了三列，所以theta自然也要有三列
theta = np.full((1, X2.shape[1]), 0)   # np.full()是填充0的函数，与np.zeros()差不多; 79行的.ravel去掉了我觉得用它没啥用
# np.full((行数，列数)， 想要填充的数字)

# 打印theta看看，应该是个一行3列（1，3）的矩阵
print(theta, 'theta')

# 看下维度
print('X2.shape:', X2.shape)
print('theta.shape', theta.shape)
print('y2.shape', y2.shape)
print(np.full(((1, X2.shape[1])), 0))

# 计算代价函数（theta初始值为0）
computeCost(X2, y2, theta)
print('Cost_init:', computeCost(X2, y2, theta))  #得到初始的成本代价（还未开始迭代）
# ---------------------------


# ---------------------------
# 2.batch gradient decent(批量梯度下降)
# 关键点在于theta0和thata1要同时更新，需要用temp进行临时存储
def gradientDescent(X2, y2, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))   # 按照theta的shape建立一个（1，3）的零值矩阵temp，为了临时存储theta，方便进行迭代更新。
    parameters = int(theta.shape[1])  # theta.shape[1]就是3; ravel计算需要求解的参数个数，功能将多维数组降至一维
    print(parameters, 'parameters')
    cost = np.zeros(iters) # 构建iters个0的数组，用来存放cost

    for i in range(iters):
        error = (X2 * theta.T) - y2  # 计算出每一组数据的误差，每行都是样本的误差

        for j in range(parameters):
            # 线性回归方程对theta求导后的公式为(theta.T*X-y)*xj的平方的sum
            # 所以这里用X*theta.T，即列向保存，再乘以X中对应的每一列，方便保存求和。
            # multiply为数组对应元素相乘，所以这个term就是还未相加的对theta求导后的数据
            term = np.multiply(error, X2[ : , j])
            temp[0, j] = theta[0, j] - ((alpha / len(X2)) * np.sum(term))  # 前面转不转置和上课学的公式不太一样，这没关系，因为书
            # 上的转置是考虑了求和，我们这里的转置是为了变成可以相乘的方块，至于求和最后用了sum直接求

        theta = temp
        cost[i] = computeCost(X2, y2, theta)  # 每迭代一次就调用一次成本函数，计算目标方程的值；并存为cost数组里面的第i个值

    return theta, cost

# 初始化学习速率和迭代次数
alpha = 0.01
iters = 1000

g2, cost2 = gradientDescent(X2, y2, theta, alpha, iters)
# g为迭代完之后的theta;（ 按return的theta， cost排序）
print(g2, "g2")  # 此时的g为满足使成本函数最小的最优值theta
print(cost2, "cost2")

minCost = computeCost(X2, y2, g2)  # 代入最优值theta，计算最小成本函数
print(minCost, "minCost")


# ---------------------------
'''因为现在是两个x变量，一个y变量，所以画不了图像了（按理说三维图像还是可以画的，懒得学了），三维以上都没图像的，现在开始习惯'''
# # 现在我们来绘制纯属模型以及数据，直观地看出它的拟合。fig代表整个图像，ax代表实例
# x = np.linspace(data2.Population.min(), data2.Population.max(), 100)  # 从最小数据到最大数据（即所有数据）中随机抽100个数据点，
# # 用np.linspace可以将这些点用等差数列排列出来，即得到一个个间隔均匀的x点
# # 线性回归的最终方程
# f = g2[0, 0] + (g2[0, 1] * x)  # 将一个个间隔均匀的x点带入，得到一个个f值
#
# fig, ax = plt.subplots(figsize=(12, 8))  # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作。figsize用来设置图形的大小
# ax.plot(x, f, 'r', label='Prediction')  # 设置横纵坐标的函数，并设置颜色为红色，在图标上标上'Prediction'标签
# ax.scatter(data2.Population, data2.Profit, label='Training Data')  # 画散点图，设置横纵坐标的函数，并设置颜色为红色，在图上标上'Training Data'标签
# ax.legend(loc=4)  # 给标签选个位置，1表示在第一象限（右上角），2表示在左上角，3表示在左下角，4表示在第四象限
# ax.set_xlabel('Population')  # 设定x轴名称
# ax.set_ylabel('Profit')  # 设定y轴名称
# ax.set_title('Predicted Profit vs. Population Size')  # 设定整个表的名称
# plt.show()




# ---------------------------
# 由于梯度方程函数在每个训练迭代中输出一个代价的向量，所以我们也可以绘制代价函数。请注意，代价总是在降低的，这是凸优化问题的一个特点。
# 代价函数的曲线
# fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')  # np.arange()自动返回等差数组
# print(np.arange(iters), "np.arange(iters)")
# print(cost2, "cost_pic")
# print(cost2.shape[0], "cost_pic")
ax.set_xlabel('Iterations')  # 设定x轴名称
ax.set_ylabel('Cost')  # 设定y轴名称
ax.set_title('Error vs. Training Epoch')  # 设定整个表的名称
plt.show()










