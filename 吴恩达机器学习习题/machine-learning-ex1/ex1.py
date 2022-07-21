# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:53:02 2022
@author: wzj
python version: python 3.9

Title: Linear regression with one variable
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())  # 预览数据

print(data.describe())  # 对于数值数据，结果的索引将包括计数，平均值，标准差，最小值，最大值以及较低的百分位数和5O。默认情况下，较低的百分位数
# 为25，较高的百分位数为75百分位数与中位数相同。

# print("运行结束!")

# 数据可视化，绘制散点图kind:取值为line或者scatter,后者为默认值图像大小
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()


# ---------------------------
# 现在让我们使用梯度下降来实现线性回归。以最小化成本函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)  # 即成本函数公式求和符号右边；.T是转置矩阵的意思
    return np.sum(inner) / (2 * len(X))  # np.sum()可以将矩阵里的元素全部加起来，即成本函数公式求和符号与（1/2m）
# return将它后面的值/结果返回给定义的函数，不然打印函数名出来是空值；有了return之后，所以下面打印函数名才能出现值/结果


# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。在训练集的左侧插入一列全为1的列，以便计算
data.insert(0, 'ones', 1)  # 即x0=1 索引index为O, name为ones,value为1.

# 现在我们来做一些变量初始化。.shape会得到该矩阵共有（几行，几列）.shape[0]为第一维的长度，shape[1]为第二维的长度，即列，
# pandas中利用.iloc选取数据loc; ','前的部分标明选取的行，','后的部分标明选取的列， 此时三列了
# set X (training data) and y (target variable)
cols = data.shape[1]  # 表示得到第二维（行）一共有3列，即cols=3
X = data.iloc[ : , 0:cols-1]  # [X是所有行， y是从0到cols-1列（即第一、二列）]
y = data.iloc[ : , cols-1: cols]  # [X是所有行， y是从cols-1到cols列（即第三列）]

# 打印表头，观察下X（训练集）和y(目标变量)是否正确
print(X.head())
print(y.head())

# 代价函数是应该是numpy矩阵，所以我们需要转换x和y成矩阵，然后才能使用它们。 我们还需要初始化theta，即把theta所有元素都设置为0。
# pandas读取的数据是DataFrame形式，优势是能对数据进行很多操作,但要想进行矩阵运算，要将DataFrame形式转换为矩阵，例如x=np.matrix(X.values)
X = np.matrix(X.values)  # 将X转换为矩阵
y = np.matrix(y.values)  # 将y转换为矩阵
theta = np.matrix(np.array([0, 0]))  # 因为X是两列，所以theta自然也要有两列

#打印theta检查，应该是个一行两列的零矩阵
print('theta:', theta, )

# 看下维度
print('X.shape:', X.shape)
print('theta.shape', theta.shape)
print('y.shape', y.shape)

# 计算代价函数（theta初始值为0）
computeCost(X, y, theta)
print('Cost_init:', computeCost(X, y, theta))  #得到初始的成本代价（还未开始迭代）


# ---------------------------
# 2.batch gradient decent(批量梯度下降)
# 关键点在于theta0和thata1要同时更新，需要用temp进行临时存储
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))   # 按照theta的shape建立一个（1，2）的零值矩阵temp，为了临时存储theta，方便进行迭代更新。
    parameters = int(theta.ravel().shape[1])  # theta.shape[1]就是2; ravel计算需要求解的参数个数，功能将多维数组降至一维
    cost = np.zeros(iters) # 构建iters个0的数组，用来存放cost

    for i in range(iters):
        error = (X * theta.T) - y  # 计算出每一组数据的误差，每行都是样本的误差

        for j in range(parameters):
            # 线性回归方程对theta求导后的公式为(theta.T*X-y)*xj的平方的sum
            # 所以这里用X*theta.T，即列向保存，再乘以X中对应的每一列，方便保存求和。
            # multiply为数组对应元素相乘，所以这个term就是还未相加的对theta求导后的数据
            term = np.multiply(error, X[ : , j])  # X[ : , j]的意思是X[所有行，第j列]
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  # 前面转不转置和上课学的公式不太一样，这没关系，因为书
            # 上的转置是考虑了求和，我们这里的转置是为了变成可以相乘的方块，至于求和最后用了sum直接求

        theta = temp
        cost[i] = computeCost(X, y, theta)  # 每迭代一次就调用一次成本函数，计算目标方程的值；并存为cost数组里面的第i个值

    return theta, cost

# ---------------------------
# 初始化学习速率和迭代次数
alpha = 0.01  # 0.005，0.001一个比一个迭代得快
iters = 1000  # 10000迭代也曲线也挺好的

# ---------------------------
# 调用梯度下降函数，计算迭代后的最优值
g, cost = gradientDescent(X, y, theta, alpha, iters)
# g为迭代完之后的theta;（ 按return的theta， cost排序）
print('g:', g)  # 此时的g为满足使成本函数最小的最优值theta

minCost = computeCost(X, y, g)  # 代入最优值theta，计算最小成本函数
print('minCost:', minCost)


# ---------------------------
# 现在我们来绘制纯属模型以及数据，以便直观地看出它的拟合。fig代表整个图像，ax代表实例
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 从最小数据到最大数据（即所有数据）中随机抽100个数据点，
# 用np.linspace可以将这些点用等差数列排列出来，即得到一个个间隔均匀的x点
# 线性回归的最终方程
f = g[0, 0] + (g[0, 1] * x)  # 将一个个间隔均匀的x点带入h(\theta)函数，得到一个个f值,也就是我们的预测值

fig, ax = plt.subplots(figsize=(12, 8))  # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作。figsize用来设置图形的大小
ax.plot(x, f, 'r', label='Prediction')  # 设置横纵坐标的函数，并设置颜色为红色，在图标上标上'Prediction'标签
ax.scatter(data.Population, data.Profit, label='Training Data')  # 画散点图，设置横纵坐标的函数，并设置颜色为红色，在图上标上'Training Data'标签
ax.legend(loc=4)  # 给标签选个位置，1表示在第一象限（右上角），2表示在左上角，3表示在左下角，4表示在第四象限
ax.set_xlabel('Population')  # 设定x轴名称
ax.set_ylabel('Profit')  # 设定y轴名称
ax.set_title('Predicted Profit vs. Population Size')  # 设定整个表的名称
plt.show()


# ---------------------------
# 由于梯度方程函数在每个训练迭代中输出一个代价的向量，所以我们也可以绘制代价函数。请注意，代价总是在降低的，这是凸优化问题的一个特点。
# 代价函数的曲线
# fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')  # np.arange()自动返回等差数组
ax.set_xlabel('Iterations')  # 设定x轴名称
ax.set_ylabel('Cost')  # 设定y轴名称
ax.set_title('Error vs. Training Epoch')  # 设定整个表的名称
plt.show()










