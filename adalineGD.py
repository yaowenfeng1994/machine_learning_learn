import numpy as np
from numpy.random import seed
import math


# 批量梯度下降法BGD
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.cost_ = []
        self.w_ = []
        self.activation_func = lambda s: 1.0 / (1 + math.exp(-s))

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = y - output
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    # def activation(self, x):
    #     return self.net_input(x)

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def log(self):
        self.activation_func(1)

# 随机梯度下降法SGD
class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.w_ = []
        self.cost_ = []
        if random_state:
            seed(random_state)

    def fit(self, x, y):
        self._initialize_weights(x.shape[1])
        for _ in range(self.n_iter):
            if self.shuffle:
                # 打乱样本顺序
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            print(sum(cost))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        print(self.cost_, self.w_)
        return self

    def partial_fix(self, x, y):
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self

    @staticmethod
    def _shuffle(x, y):
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.predict(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)
