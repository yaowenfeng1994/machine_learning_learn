import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sbs import SBS

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# print(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.fit_transform(x_test)

# lr = LogisticRegression()
# lr.fit(x_train_std, y_train)
# print(lr.score(x_test_std, y_test))
# print(lr.coef_, lr.intercept_)

# sbs = SBS(lr, k_features=8)
# sbs.fit(x_train_std, y_train)
# print(sbs.scores_)
# print(sbs.estimator.intercept_, sbs.estimator.coef_)
# feat_labels = df_wine.columns[1:]
# print(feat_labels)
# forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
# forest.fit(x_train, y_train)
# importance = forest.feature_importances_
# indices = np.argsort(importance)[::-1]
# print(importance)
# for f in range(x_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labelsdata_handler.py:37[f], importance[indices[f]]))

# for idx, a in enumerate(x_train_std.T):
    # print(a)
    # print(idx)
cov_mat = np.cov(x_train_std.T)
# print(cov_mat)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)
print(eigen_vecs)




