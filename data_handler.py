import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# print(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.fit_transform(x_test)

lr = LogisticRegression(penalty="l1", C=0.1)
lr.fit(x_train_std, y_train)
print(lr.score(x_train_std, y_train))
print(lr.intercept_, lr.coef_)