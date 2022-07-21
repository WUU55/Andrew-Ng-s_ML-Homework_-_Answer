# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:53:02 2022
@author: wzj
python version: python 3.9

Title: 线性不可分案例
案例:设想你是工厂的生产主管，你要决定是否芯片要被接受或抛弃
数据集:ex2data2.txt,芯片在两次测试中的测试结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # seaborn与matplotlib在功能上是互补关系
# plt.style.use('fivethirtyeight')  # 样式美化
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report  # 这个包是评价报告

# 照常，先读取数据观察一下
path = 'ex2data2.txt'
data = pd.read_csv(path, names=['Test1', 'Test2', 'Accepted'])
print(data.head())
print(data.describe())

# ---------------------------
'''第三种画图的方式'''
fig, ax = plt.subplots(figsize=(12, 8))  # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
# 将'adimitted'这列是0的点拎出来，画一个散点图
ax.scatter(data[data['Accepted'] == 0]['Test1'], data[data['Accepted'] == 0]['Test2'], c='r', marker='x', label='y=0')
''''''
# print(data['Accepted'], "data['Accepted']")  # data[a,b]就是找到a行b列，这里直接指出的'Accepted'，那就直接打印Accepted这一列
# print("# ---------------------------")
# print(data['Accepted'] == 0)  # 布尔类型，判断并打印Accepted这一列，是0的话就打印True，是1的话就打印False
# print("# ---------------------------")
# print(data[data['Accepted'] == 0])  # 逐行判断Accepted这列的每一行，Accepted是0的话，就把这一行的Test1 Test2 accepted全都给打印出来
# print(data[data['Accepted'] == 0]['Test1'])  # 将前面的"data[data['Accepted'] == 0]" 整体就看做一个（117，3）矩阵，
# # "矩阵['Test1']"即为找到该矩阵中'Test1'这一列
''''''
# 将'adimitted'这列是0的点拎出来，画一个散点图
ax.scatter(data[data['Accepted'] == 1]['Test1'], data[data['Accepted'] == 1]['Test2'], c='b', marker='o', label='y=1')
ax.legend(loc=1)  # 标签位置默认
ax.set(xlabel='Test1',
       ylabel='Test2')
# plt.show()
# ---------------------------


# ---------------------------
#　特征映射,把x1,x2两列特征扩展成28个不同次数的x1x2之积，相当于增加样本的丰富（维）度，这样才能画出更具细节的曲线
def feature_mapping(x1, x2, power):
       data = {}   # 设置一个空字典
       for i in np.arange(power+1):
              # print(i)
              for j in np.arange(i+1):
                     data['F{}{}'.format(i-j, j)] = np.power(x1, i-j) * np.power(x2, j)

       return pd.DataFrame(data)  # 将data的字典格式转换为Dataframe的格式（可看作矩阵格式）

x1 = data['Test1']
x2 = data['Test2']

data2 = feature_mapping(x1, x2, 6)
print(data2.head(), 'data2')
# ---------------------------


# ---------------------------
X = data2.values
print('X.shape:', X.shape)
y = data.iloc[:, -1].values  # 此处的'-1'为序号，意思是倒着数第一个
# print(data.iloc[:, -1])
y = y.reshape(len(y), 1)  # .reshape(a,b)表示以a行b列的数组形式显示，即将y表示为(118，1)矩阵形式
# ---------------------------


# ---------------------------
# 损失函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))

# 实现逻辑回归的代价函数，除了上一题的两个部分，-y(log(hx)和-(1-y)log(1-hx)，还多了一项lamda/(2m) * sum(theta^2),维数高了，要防止过拟合
def computeCost(X, y, theta, lamda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T) + 1e-5))  # 结果是个（118，1）矩阵
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T) + 1e-5))  # 结果也是个（118，1）矩阵
    reg = -np.sum(np.power(theta.T[1:], 2)) * (lamda / (2 * len(X)))    # reg就是正则化项，lamda/2m,乘以所有theta的平方的和，但
    # 是不包括theta0 这就是防止过拟合的那一项；theta.T[1:]是将输出一个（1，27）的矩阵(因为公式中theta_j的j要>=1,故theta0是不参与正则化的)
    # print(theta.T[1:], "theta[1:]")

    return np.sum(first - second) / (len(X)) + reg
# ---------------------------


# ---------------------------
# 初始化theta、X、y
# theta = np.zeros((28, 1))  # 将初始theta设置为（28，1）的0矩阵  # 如果用这行那上面的theta就全都不用转置了
theta = np.full((1, X.shape[1]), 0)    # np.full((行数，列数)，想要填充的数字)， 将初始theta设置为（1， 28）的0矩阵
print('theta.shape:', theta.shape)
lamda = 1  # lamda设置为1
print(computeCost(X, y, theta, lamda), "cost_init")
# ---------------------------


# ---------------------------
# 2.batch gradient decent(梯度下降函数)
def gradientDescent(X, y, theta, alpha, iters, lamda):
    # print(X, 'X')
    # print(X @ theta.T, "X * theta.T")
    # print(y, 'y')
    # print(y.shape, "y.shape()")
    # print(sigmoid(X @ theta.T), 'sigmoid(X @ theta.T)')
    # print(sigmoid(X @ theta.T).shape, 'sigmoid(X @ theta.T).shape')
    #
    # print(sigmoid(X @ theta.T) - y, 'error')
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = sigmoid(X @ theta.T) - y  # 计算出每一组数据的误差，每行都是样本的误差

        for j in range(parameters):
            reg = theta.T[1:] * (lamda / len(X))  # 因为theta.T[1:]是没有theta0的，所以它是（27，1）矩阵，比28少了一行
            reg = np.insert(reg, 0, values=0, axis=0)  # 所以人为给它插入一行0，(reg, 索引=0， 值=0， 坐标轴中位置=0)
            term = np.multiply(error, X[:, j])   # X[ : , j]的意思是X[所有行，第j列]
            # print(term, 'term')
            # print(theta.T, 'theta.T')
            # print(reg.shape, 'reg.shape')
            # print(len(X), 'len(X)')
            # print(alpha / len(X), 'alpha / len(X)')
            # print(np.sum(term), 'np.sum(term)')
            # print((alpha / len(X)) * np.sum(term), "(alpha / len(X)) * np.sum(term) + reg")

            temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term) - (alpha * np.sum(reg))  # 前面转不转置和上课学的公式不太一样，这没关系，
            # 因为书上的转置是考虑了求和，我们这里的转置是为了变成可以相乘的方块，至于求和最后用了sum直接求

        theta = temp
        cost[i] = computeCost(X, y, theta, lamda)

    return theta, cost
# ---------------------------


# ---------------------------
# 初始化学习速率和迭代次数
alpha = 0.005  # 看up主ladykaka007设置的
iters = 200000  # 看up主ladykaka007设置的;
lamda = 0.0001

g, cost = gradientDescent(X, y, theta, alpha, iters, lamda)
print("g:", g)
print("cost", cost)

minCost = computeCost(X, y, g, lamda)
print("minCost:", minCost)   # 和up主ladykaka007的结果差很多多，她是minCost0.46，我是0.69


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

print("预测正确的概率：", admi)  # 和up主ladykaka007的结果差很多，她是minCost0.8几，我是0.508
# ---------------------------


# ---------------------------
# x = np.linspace(-1.2, 1.2, 200)
# xx, yy = np.meshgrid(x, x)  # meshgrid() 用于生成网格采样点矩阵。画出xx,yy就是一个200*200的网格
# z = feature_mapping(xx.ravel(), yy.ravel(), 6).values  # 调用feature,即得到通过特征映射把x1,x2两列特征扩展成的28个不同次数的X和y
#
# zz = z @ g  # g=最终最优的theta；相当于把最优值theta与z(即28个x与y)带回目标成本函数中
# zz = zz.reshape(xx.shape)  # 把最终的成本函数值zz按28个特征映射的几行几列形式表达出来
#
# fig, ax = plt.subplots(figsize=(12, 8))  # fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
# # 将'adimitted'这列是0的点拎出来，画一个散点图
# ax.scatter(data[data['Accepted'] == 0]['Test1'], data[data['Accepted'] == 0]['Test2'], c='r', marker='x', label='y=0')
# # 将'adimitted'这列是0的点拎出来，画一个散点图
# ax.scatter(data[data['Accepted'] == 1]['Test1'], data[data['Accepted'] == 1]['Test2'], c='b', marker='o', label='y=1')
# ax.legend(loc=1)  # 标签位置默认
# ax.set(xlabel='Test1',
#        ylabel='Test2')
#
# plt.contour(xx, yy, zz, 0)  # 画等高线，实际就是一个圈的形式，X, Y表示的是坐标位置，Z代表每个坐标对应的高度值（只是图是二维的看不出高度）
# plt.show()



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
























