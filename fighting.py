import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import warnings

warnings.filterwarnings('ignore')  # 忽略warning

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                 header=None)
x = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)
# print(le.transform(["M", "B"]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

########################################################################################################################
pipe_lr = Pipeline([("scl", StandardScaler()), ("pca", PCA(n_components=2)),
                    ("clf", LogisticRegression(random_state=1))])  # 数据缩放，降维，线性算法预测模型
# pipe_lr.fit(x_train, y_train)
# print(pipe_lr.score(x_test, y_test))

kfold = StratifiedKFold(n_splits=10, random_state=1)

scores = []
for train, test in kfold.split(x_train, y_train):  # 返回的是下标
    pipe_lr.fit(x_train[train], y_train[train])
    score = pipe_lr.score(x_train[test], y_train[test])
    scores.append(score)
print(scores)

########################################################################################################################
pipe_svc = Pipeline([("scl", StandardScaler()), ("clf", SVC(random_state=1))])
# param_range = [0.0001, 0.001, 0.01, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{"clf__C": param_range, "clf__kernel": ["linear"]},
#               {"clf__C": param_range, "clf__gamma": param_range, "clf__kernel": ["rbf"]}]
# # param_grid 是一个参数列表，GridSearchCV会从这个列表里穷举出最佳的参数选项
# gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)
# gs = gs.fit(x_train, y_train)
# print(gs.best_score_)
#
# print(gs.best_params_)
# clf = gs.best_estimator_
# clf.fit(x_train, y_train)
# print(clf.score(x_test, y_test))

########################################################################################################################

# pipe_svc.fit(x_train, y_train)
# y_pred = pipe_svc.predict(x_test)
# print(y_pred)
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(confmat)

