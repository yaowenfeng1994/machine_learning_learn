import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                 header=None)
x = df.loc[:, 2:].values
y = df.loc[:, 1].values
# print(x)
print(len(y))
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# print(le.transform(["M", "B"]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

pipe_lr = Pipeline([("scl", StandardScaler()), ("pca", PCA(n_components=2)),
                    ("clf", LogisticRegression(random_state=1))])  # 数据缩放，降维，线性算法预测模型
pipe_lr.fit(x_train, y_train)
print(pipe_lr.score(x_test, y_test))

kfold = StratifiedKFold(n_splits=10, random_state=1)
print(kfold)
