# -*- coding:utf-8 -*-
# 2019-10-23 22:04
# k-近邻算法预测鸢尾花种类

from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


def load_data():
    iris = load_iris()  # 加载鸢尾花数据
    # print("鸢尾花数据集的返回值：\n", iris)
    # # 返回值是一个继承自字典的Bench
    # print("鸢尾花的特征值:\n", iris["data"])
    # print("鸢尾花的目标值：\n", iris.target)
    # print("鸢尾花特征的名字：\n", iris.feature_names)
    # print("鸢尾花目标值的名字：\n", iris.target_names)
    # print("鸢尾花的描述：\n", iris.DESCR)
    return iris


def plot_iris():
    """
    鸢尾花数据分布图绘制函数
    :return: None
    """
    iris = load_data()
    iris_df = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
    iris_df['Species'] = iris.target
    sns.lmplot(data=iris_df, x='Petal_Width', y='Sepal_Length', hue="Species", fit_reg=False)
    plt.xlabel('Petal_Width')
    plt.ylabel('Sepal_Length')
    plt.title('鸢尾花种类分布图')
    plt.savefig('鸢尾花种类分布图.png')
    plt.show()


def iris_predict():
    """
    K-近邻算法
    预测鸢尾花种类
    :return: None
    """
    iris = load_data()
    # 对鸢尾花数据集进行分割

    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 特征工程-特征值标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 机器学习（模型训练）
    estimator = KNeighborsClassifier(n_neighbors=9)  # 指定k的个数为9
    estimator.fit(x_train, y_train)

    # 模型保存
    joblib.dump(estimator, './data/iris.pkl')
    # 加载模型
    estimator = joblib.load('./data/iris.pkl')

    # 模型评估
    # 对比真实值与预测值
    y_predict = estimator.predict(x_test)
    print("预测结果为：\n", y_predict)
    print("对比真实值与预测值：\n", y_predict == y_test)

    # 直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)


def main():
    iris_predict()


if __name__ == '__main__':
    main()
    # plot_iris()

    # 运行结果
    #     预测结果为：
    #     [0 2 1 2 1 1 1 1 1 0 2 1 2 2 0 2 1 1 1 1 0 2 0 1 2 0 2 2 2 2 0 0 1 1 1 0 0
    #      0]
    # 对比真实值与预测值：
    # [True  True  True  True  True  True  True False  True  True  True  True
    #  True  True  True  True  True  True False  True  True  True  True  True
    #  True  True  True  True  True  True  True  True  True  True  True  True
    #  True  True]
    # 准确率： 0.9473684210526315
