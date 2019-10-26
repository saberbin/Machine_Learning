# -*- coding:utf-8 -*-
# 2019-10-24 19:00
# 波士顿房价预测-岭回归

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.datasets import load_boston
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error


def load_data():
    """
    产生（加载）波士顿房价数据并返回
    :return: 波士顿房价数据
    """
    data = load_boston()
    return data


def boston_predict_model_3(data):
    """
    波士顿房价预测-岭回归
    :param data: boston 房价数据
    :return: None
    """
    # 1. 加载数据
    # data = load_data()

    # 2. 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=22)

    # 3. 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 机器学习-线性回归（正规方程）
    estimator = Ridge(alpha=1)
    # estimator = RidgeCV(alphas=(0.1, 1, 10))  # 岭回归与交叉验证

    # 4.1 模型训练
    estimator.fit(x_train, y_train)

    # 4.2 模型保存
    joblib.dump(estimator, './data/boston_predict_model_3.pkl')
    # 4.3 模型加载
    # estimator = joblib.load('./data/boston_predict_model_3.pkl')


    # 5. 模型评估
    # 5.1 获取模型系数
    y_predict = estimator.predict(x_test)
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)
    # 5.2 评价-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)


def main():
    data = load_data()
    boston_predict_model_3(data)


if __name__ == '__main__':
    main()
    # 岭回归预测结果：
    #     预测值为:
    #     [28.22119941 31.49858594 21.14690941 32.64962343 20.03976087 19.07187629
    #      21.11827061 19.61935024 19.64669848 32.83666525 21.01034708 27.47939935
    #      15.55875601 19.80406014 36.86415472 18.79442579  9.42343608 18.5205955
    #      30.67129766 24.30659711 19.07820077 34.08772738 29.77396117 17.50394928
    #      34.87750492 26.52508961 34.65566473 27.42939944 19.08639183 15.04854291
    #      30.84974343 15.76894723 37.18814441  7.81864035 16.27847433 17.15510852
    #      7.46590141 19.98474662 40.55565604 28.96103939 25.25570196 17.7598197
    #      38.78171653  6.87935126 21.76805062 25.25888823 20.47319256 20.48808719
    #      17.24949519 26.11755181  8.61005188 27.47070495 30.57806886 16.57080888
    #      9.42312214 35.50731907 32.20467352 21.93128073 17.62011278 22.08454636
    #      23.50121152 24.08248876 20.16840581 38.47001591 24.69276673 19.7638548
    #      13.96547058  6.76070715 42.04033544 21.9237625  16.88030656 22.60637682
    #      40.74664535 21.44631815 36.86936185 27.17135794 21.09470367 20.40689317
    #      25.35934079 22.35676321 31.1513028  20.39303322 23.99948991 31.54251155
    #      26.77734347 20.89368871 29.05880401 22.00850263 26.31965286 20.04852734
    #      25.46476799 24.08084537 19.90846889 16.47030743 15.27936372 18.39475348
    #      24.80822272 16.62280764 20.86393724 26.70418608 20.74534996 17.89544942
    #      24.25949423 23.35743497 21.51817773 36.76202304 15.90293344 21.52915882
    #      32.78684766 33.68666117 20.61700911 26.78345059 22.72685584 17.40478038
    #      21.67136433 21.6912557  27.66684993 25.08825085 23.72539867 14.64260535
    #      15.21105331  3.81916568 29.16662813 20.67913144 22.33386579 28.01241753
    #      28.531445]
    # 模型中的系数为:
    # [-0.63591916  1.12109181 - 0.09319611  0.74628129 - 1.91888749  2.71927719
    #  - 0.08590464 - 3.25882705  2.41315949 - 1.76930347 - 1.74279405  0.87205004
    #  - 3.89758657]
    # 模型中的偏置为:
    # 22.62137203166228
    # 误差为:
    # 20.65644821435496


