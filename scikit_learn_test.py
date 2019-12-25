from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier, SGDRegressor
import pandas as pd
import numpy as np

iris = pd.read_csv("iris.data")
y = iris.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
x = iris.iloc[0:100, [2, 3]].values

# iris = datasets.load_iris()
# x = iris.data[:, [2, 3]]
# y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

sc = StandardScaler()
a = sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
print(111, len(x), len(x_train_std), len(y_train))
# ppn = Perceptron(eta0=0.1, random_state=3)
# ppn.fit(x_train_std, y_train)
# y_pred = ppn.predict(x_test_std)
# print(y_test,y_pred)
# print((y_test != y_pred))

lr = SGDClassifier(random_state=0)
lr.fit(x_train_std, y_train)
# print(x_test_std, y_test)
print(222, lr.coef_, lr.intercept_)
print(111, lr.predict(x_test_std), y_test)
