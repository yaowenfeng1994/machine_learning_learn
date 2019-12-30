import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class LineRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
        self.cost_ = []
        self.w_ = []

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
        print(self.cost_)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return self.net_input(x)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="blue")
    plt.plot(X, model.predict(X), color="red")
    return None


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None,
                 sep="\s+")
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
# print(df.head())

# x = df[["RM"]].values
# y = df["MEDV"].values
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# x_std = sc_x.fit_transform(x)
# y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# lr = LineRegressionGD()
# lr.fit(x_std, y_std)
# print("Slope: %.3f" % lr.w_[1])
# print("Intercept: %.3f" % lr.w_[0])
#
# plt.plot(range(1, lr.n_iter + 1), lr.cost_)
# plt.ylabel("SSE")
# plt.xlabel("Epoch")
# plt.show()

# lin_regplot(x_std, y_std, lr)
# plt.xlabel("Average number of rooms [RM] (standardized)")
# plt.ylabel("Price in $1000\'s [MEDV] (standardized)")
# plt.show()

# num_rooms_std = sc_x.transform([[5.0]])
# price_std = lr.predict(num_rooms_std)
# print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))

# slr = LinearRegression()
# slr.fit(x, y)
# print("Slope: %.3f" % slr.coef_[0])
# print("Intercept: %.3f" % slr.intercept_)
# lin_regplot(x, y, slr)
# plt.xlabel("Average number of rooms [RM] (standardized)")
# plt.ylabel("Price in $1000\'s [MEDV] (standardized)")
# plt.show()

# ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, loss="absolute_loss",
#                          residual_threshold=5.0, random_state=0)
# ransac.fit(x, y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
# line_x = np.arange(3, 10, 1)
# # print(line_x, line_x[:, np.newaxis], line_x[:, np.newaxis].flatten())
# line_y_ransac = ransac.predict(line_x[:, np.newaxis])
# plt.scatter(x[inlier_mask], y[inlier_mask], c="steelblue", edgecolors="white", marker="o", label="Inliers")
# plt.scatter(x[outlier_mask], y[outlier_mask], c="limegreen", edgecolors="white", marker="s", label="Outliers")
# plt.plot(line_x, line_y_ransac, color="black", lw=2)
# plt.xlabel("Average number of rooms [RM]")
# plt.ylabel("Price in $1000s [MEDV]")
# plt.legend(loc="upper left")
# # plt.show()
#
# print("Slope: %.3f" % ransac.estimator_.coef_[0])
# print("Intercept: %.3f" % ransac.estimator_.intercept_)

x = df.iloc[:, :-1].values
y = df["MEDV"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
slr = LinearRegression()
slr.fit(x_train, y_train)
y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c="steelblue", edgecolors="white", marker="o", label="Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c="limegreen", edgecolors="white", marker="s", label="Test data")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, color="black", lw=2)
plt.xlim([-10, 50])
plt.show()
