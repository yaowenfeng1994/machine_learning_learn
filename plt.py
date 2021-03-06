import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import adalineGD
import math

df = pd.read_csv("iris.data")
y = df.iloc[0:100, 4].values

y = np.where(y == "Iris-setosa", 0, 1)

x = df.iloc[0:100, [0, 2]].values


# print(x)
# plt.scatter(x[:50, 0], x[:50, 1], color="red", marker="o", label="setosa")
# plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="X", label="versicolor")
# plt.xlabel("petal length")
# plt.ylabel("setal length")
# plt.legend(loc="upper left")
# plt.show()

# 随机梯度下降法SGD
class Perceptron(object):
    """
    原始形态感知机
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.wb = []
        self.errors_ = []

    def fit(self, x, y):
        """
        拟合函数，使用训练集来拟合模型
        :param x:training sets
        :param y:training labels
        :return:self
        """
        # X's each col represent a feature
        # initialization wb(weight plus bias)
        self.wb = np.zeros(1 + x.shape[1])
        # the main process of fitting
        self.errors_ = []  # store the errors for each iteration
        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.eta * (yi - self.predict(xi))
                self.wb[1:] += update * xi
                self.wb[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def logistics_fit(self, x, y):
        self.wb = np.zeros(1 + x.shape[1])
        self.errors_ = []
        for idx in range(self.n_iter):
            for xi, yi in zip(x, y):
                update = (yi - self.logistics(self.net_input(xi))) * xi  # 这条是 李航统计学习求导 得出
                self.wb[1:] += update
                self.wb[0] += update.sum()
                yp = self.logistics(self.net_input(xi))
                # print("样本结果：", yi, "预测结果:", yp)
            # print("第 %d 次迭代" % idx)
        print(self.wb)
        return self

    def predict_v2(self, x):
        l = list()
        for xi in x:
            a = np.dot(xi, np.array([-5.83, 10.64])) + np.array([4.81])
            l.append(self.logistics(a))
        return l

    @staticmethod
    def logistics(s):
        return 1.0 / (1 + math.exp(-s))

    def net_input(self, xi):
        """
        计算净输入
        :param xi:
        :return:净输入
        """
        return np.dot(xi, self.wb[1:]) + self.wb[0]

    def predict(self, xi):
        """
        计算预测值
        :param xi:
        :return:-1 or 1
        """
        return np.where(self.net_input(xi) <= 0.0, -1, 1)


ppn = Perceptron(eta=0.001, n_iter=20)
print(x.shape, y)
# ppn.logistics_fit(x, y)
# print(ppn.log())
print(ppn.predict_v2(x))
# ppn = adalineGD.AdalineSGD(eta=0.1, n_iter=10, shuffle=False, random_state=1)
# ppn.fit(x, y)

# print(np.dot([[1, 2], [3, 4]], [2, 2])+1)
# a = np.array([[7, 8, 9]])
# b = np.array([[1, 2, 3], [4, 5, 6]])
# print(a ** 2)

# r = np.random.permutation(len(y))
# print(y[r])
