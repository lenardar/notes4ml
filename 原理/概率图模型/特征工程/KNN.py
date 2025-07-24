from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import time

def log_result(message):
    with open('result.txt', 'a') as f:
        f.write(message + '\n')

# 读取数据
# /home/ubuntu/project/ML/digit.csv
try:
    data = pd.read_csv('notes4ml/特征工程/digit.csv')
    log_result('读取数据成功！')
except Exception as e:
    log_result(f'读取数据失败：{e}')

try:
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    log_result('X,y 提取成功！')
except Exception as e:
    log_result(f'提取X,y失败：{e}')

try:
    X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
    log_result('X_fsvar生成成功！')
except Exception as e:
    log_result(f'生成X_fsvar失败：{e}')

# KNN未处理数据
try:
    t1 = time.time()
    score = cross_val_score(KNN(), X, y, cv=5).mean()
    t2 = time.time()
    log_result(f'KNN未处理数据的准确率为：{score:.2f}, 用时{t2-t1:.2f}')
except Exception as e:
    log_result(f'KNN未处理数据失败：{e}')

# KNN处理数据
try:
    t1 = time.time()
    score = cross_val_score(KNN(), X_fsvar, y, cv=5).mean()
    t2 = time.time()
    log_result(f'KNN处理数据的准确率为：{score:.2f}, 用时{t2-t1:.2f}')
except Exception as e:
    log_result(f'KNN处理数据失败：{e}')