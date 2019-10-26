# -*- coding:utf-8 -*-
# 2019-10-26 10:51
# 泰坦尼克号乘客生存预测-决策树


import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals import joblib


def titan():
    data_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt'
    # 1. 获取数据
    data = pd.read_csv(data_url)

    # 2. 确定特征值、目标值
    x = data[["pclass", "age", "sex"]]
    y = data["survived"]

    # 2.1 缺失值处理
    # 2.2 将特征中有类别的特征进行字典特征提取
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 2.3 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3. 特征工程-字典特征提取
    # 将数组特征转换为字典数据
    transfer = DictVectorizer(sparse=False)
    x_train = transfer.fit_transform(x_train.to_dict(orient='records'))
    x_test = transfer.transform(x_test.to_dict(orient='records'))

    # 4. 机器学习-决策树算法
    estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    estimator.fit(x_train, y_train)
    # 4.2 模型保存与加载
    joblib.dump(estimator, './data/titan.pkl')
    # estimator = joblib.load('./data/titan.pkl')
    # 保存树的结构
    export_graphviz(estimator, out_file="./data/tree.dot",
                    feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 5. 模型评估
    score = estimator.score(x_test, y_test)
    print("预测评分：", score)
    y_predict = estimator.predict(x_test)
    print("预测值：\n", y_predict)
    print("预测值与真实值对比：", y_test == y_predict)


def main():
    titan()


if __name__ == '__main__':
    main()


