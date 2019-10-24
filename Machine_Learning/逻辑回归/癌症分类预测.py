# -*- coding:utf-8 -*-
# 2019-10-24 10：33
# 癌症分类预测-逻辑回归

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def load_data():
    names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
             'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=names)
    return data


def cancer_predict():
    # 1. 加载数据
    data = load_data()

    # 2. 基本数据处理
    # 2.1 缺失值处理
    data = data.replace(to_replace="?", value=np.NaN)
    data = data.dropna()
    # 2.2 确定特征值、目标值
    x = data.iloc[:, 1:10]
    y = data['Class']
    # 2.3 数据集分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3. 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 机器学习-逻辑回归
    estimator = LogisticRegression()
    # 4.1 模型保存
    joblib.dump(estimator, './data/breast-cancer-wisconsin.pkl')
    # 4.2 加载模型
    # estimator = joblib.load('./data/breast-cancer-wisconsin.pkl')
    # 4.3 模型训练
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    y_predict = estimator.predict(x_test)
    print("预测结果为：\n", y_predict)
    print("对比真实值与预测值：\n", y_predict == y_test)

    # 5.1 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)


def main():
    cancer_predict()


if __name__ == '__main__':
    main()
    # 初次运行结果
    #     预测结果为：
    #     [2 4 4 2 2 2 2 2 2 2 2 2 2 4 2 2 4 4 4 2 4 2 4 4 4 2 4 2 2 2 2 2 4 2 2 2 4
    #      2 2 2 2 4 2 4 4 4 4 2 4 4 2 2 2 2 2 4 2 2 2 2 4 4 4 4 2 4 2 2 4 2 2 2 2 4
    #      2 2 2 2 2 2 4 4 4 2 4 4 4 4 2 2 2 4 2 4 2 2 2 2 2 2 4 2 2 4 2 2 4 2 4 4 2
    #      2 2 2 4 2 2 2 2 2 2 4 2 4 2 2 2 4 2 4 2 2 2 4 2 2 2 2 2 2 2 2 2 2 2 4 2 4
    #      2 2 4 4 4 2 2 4 4 2 4 4 2 2 2 2 2 4 4 2 2 2 4]
    # 对比真实值与预测值：
    # 389
    # True
    # 32
    # True
    # 272
    # True
    # 655
    # True
    # 271
    # True
    # ...
    # 296
    # False
    # 585
    # True
    # 256
    # True
    # 43
    # False
    # 691
    # True
    # Name: Class, Length: 171, dtype: bool
    # 准确率： 0.9766081871345029
