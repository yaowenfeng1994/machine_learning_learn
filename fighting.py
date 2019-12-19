import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                 header=None)
x = df.loc[:, 2:].values
y = df.loc[:, 1].values
print(x)
print(y)
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
