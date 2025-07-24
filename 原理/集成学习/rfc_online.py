# 导入必要的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集data.csv
data = pd.read_csv("data.csv")

# 获取X，y
X = data.iloc[:, data.columns != "Survived"]
y = data.iloc[:, data.columns == "Survived"]


# 数据预处理
# 删除Name列与Cabin列、PassengerId、Ticket	列
X = X.drop(["Name", "Cabin", "PassengerId", "Ticket"], axis=1)

# 用平均值填充Age列的缺失值
X["Age"] = X["Age"].fillna(X["Age"].mean())

# 删除Embarked列的缺失值
X = X.dropna()

# 调整y
y = y.loc[X.index]

# 重建索引
X.index = range(X.shape[0])
y.index = range(y.shape[0])

# 合并SibSp和Parch列
X["Family"] = X["SibSp"] + X["Parch"]
X = X.drop(["SibSp", "Parch"], axis=1)

# 如果Family列的值大于0，将其设置为1
X["Family"] = X["Family"].map(lambda x: 1 if x > 0 else 0)

# 将Embarked列的值转换为数值
Embarked_index = list(X['Embarked'].unique())
X["Embarked"] = X["Embarked"].map(lambda x: Embarked_index.index(x))

# 将Sex列的值转换为数值
X['Sex'] = X['Sex'].map(lambda x: ['male', 'female'].index(x))


# 构建模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 网格搜索
# 已知n_estimators=156
param_grid = {'max_depth': np.arange(1, 15, 1), 'min_samples_leaf': np.arange(1, 20, 1), 
              'min_samples_split': np.arange(1, 20, 1), 'max_features': np.arange(1, 10, 1)}

rfc = RandomForestClassifier(n_estimators=156, random_state=90)

GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(X, y.values.ravel())

# 储存模型
import joblib
joblib.dump(GS, "model.pkl")