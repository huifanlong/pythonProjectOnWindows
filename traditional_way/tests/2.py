import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            # 计算每个测试样本和训练集之间的距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # 找到距离最近的k个邻居
            k_nearest_neighbors = np.argsort(distances)[:self.k]
            # 投票决定类别
            y_pred[i] = np.argmax(np.bincount(self.y_train[k_nearest_neighbors]))
        return y_pred


data = pd.read_csv("iris_training.csv", encoding="gbk")  # 读取数据
X = data.iloc[:, 1:22]  # 读取训练数据，
Y = data.iloc[:, 22].apply(lambda e: 0 if e == "无毒" else 1)  # 读取标签类别，并进行0、1转化
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


def compare_knn(n_neighbors):
    # 初始化自己构造的knn模型，指定k值为3（不指定的话，默认为5）
    knn = KNN(n_neighbors)
    # 给knn模型传入训练数据和标签
    knn.fit(x_train.to_numpy(), y_train.to_numpy())

    # 使用模型预测测试集的类别，传入的是一组样本
    y_pred = knn.predict(x_test.to_numpy())
    # 计算测试集上的准确率
    accuracy = accuracy_score(y_test.to_numpy(), y_pred)

    # 预测给定的样本的类别
    a_pred = knn.predict(a)

    print("临近数：{}时，模型在测试集上准确率为{}，给定样本的预测类别为{}".format(n_neighbors, accuracy, a_pred))


# 随机构造一个训练数据样本
a = np.random.rand(21).reshape(1, -1)
compare_knn(n_neighbors=1)
compare_knn(n_neighbors=2)
compare_knn(n_neighbors=3)
compare_knn(n_neighbors=4)
compare_knn(n_neighbors=5)
compare_knn(n_neighbors=6)
compare_knn(n_neighbors=7)
compare_knn(n_neighbors=8)
compare_knn(n_neighbors=9)
compare_knn(n_neighbors=10)
