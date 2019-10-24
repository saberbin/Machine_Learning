# -*- coding:utf-8 -*-
# 2019-10-23 22:04
# k-近邻算法预测鸢尾花种类-交叉验证和网络搜索

from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


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
    K-近邻算法+网络搜索和交叉验证
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
    # 准备超参数
    param_dict = {"n_neighbors": [3, 5, 7, 9]}
    estimator = KNeighborsClassifier(n_neighbors=9)  # 指定k的个数为9
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)  # 网格搜索和交叉验证
    estimator.fit(x_train, y_train)  # 进行训练

    # 模型保存
    joblib.dump(estimator, './data/iris_grid_search.pkl')
    # 加载模型
    estimator = joblib.load('./data/iris_grid_search.pkl')

    # 模型评估
    # 对比真实值与预测值
    y_predict = estimator.predict(x_test)
    print("预测结果为：\n", y_predict)
    print("对比真实值与预测值：\n", y_predict == y_test)

    # 直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)

    # 交叉验证的结果
    print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
    print("最好的参数模型：\n", estimator.best_estimator_)
    print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)


def main():
    iris_predict()


if __name__ == '__main__':
    main()
    # plot_iris()

    # 运行结果
    # 预测结果为：
    #     [0 2 1 2 1 1 1 1 1 0 2 1 2 2 0 2 1 1 1 1 0 2 0 1 2 0 2 2 2 2 0 0 1 1 1 0 0
    #      0]
    # 对比真实值与预测值：
    # [True  True  True  True  True  True  True False  True  True  True  True
    #  True  True  True  True  True  True False  True  True  True  True  True
    #  True  True  True  True  True  True  True  True  True  True  True  True
    #  True  True]
    # 准确率： 0.9473684210526315
    # 在交叉验证中验证的最好结果：
    # 0.9732142857142857
    # 最好的参数模型：
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #                      metric_params=None, n_jobs=None, n_neighbors=5, p=2,
    #                      weights='uniform')
    # 每次交叉验证后的准确率结果：
    # {'mean_fit_time': array([0.00116173, 0., 0.00083582, 0.00062831]),
    #  'std_fit_time': array([0.00062384, 0., 0.00026008, 0.00044663]),
    #  'mean_score_time': array([0.00183018, 0.00197395, 0.00150021, 0.00169023]),
    #  'std_score_time': array([2.33890881e-04, 1.58248804e-05, 4.24534451e-04, 4.12225394e-04]),
    #  'param_n_neighbors': masked_array(data=[3, 5, 7, 9],
    #                                    mask=[False, False, False, False],
    #                                    fill_value='?',
    #                                    dtype=object),
    #  'params': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}, {'n_neighbors': 9}],
    #  'split0_test_score': array([0.97368421, 0.97368421, 0.97368421, 0.94736842]),
    #  'split1_test_score': array([0.97297297, 0.97297297, 0.97297297, 0.97297297]),
    #  'split2_test_score': array([0.89189189, 0.97297297, 0.97297297, 0.91891892]),
    #  'mean_test_score': array([0.94642857, 0.97321429, 0.97321429, 0.94642857]),
    #  'std_test_score': array([0.03830641, 0.00033675, 0.00033675, 0.02197906]),
    #  'rank_test_score': array([3, 1, 1, 3])}

