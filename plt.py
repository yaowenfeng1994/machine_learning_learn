import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import adalineGD

df = pd.read_csv("iris.data")
y = df.iloc[0:100, 4].values

y = np.where(y == "Iris-setosa", -1, 1)


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


ppn = adalineGD.AdalineGD(eta=0.0001, n_iter=100)
ppn.fit(x, y)
# print(ppn.log())

# ppn = adalineGD.AdalineSGD(eta=0.1, n_iter=10, shuffle=False, random_state=1)
# ppn.fit(x, y)

# print(np.dot([[1, 2], [3, 4]], [2, 2])+1)
# a = np.array([[7, 8, 9]])
# b = np.array([[1, 2, 3], [4, 5, 6]])
# print(a ** 2)

# r = np.random.permutation(len(y))
# print(y[r])
