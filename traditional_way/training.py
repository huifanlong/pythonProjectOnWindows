from dataprocessing import df_grade, df_time_record
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def convert_time_record(data):
    list_1 = [("o" + e if len(e) == 1 else e) for e in data.split(" ")]
    return " ".join(list_1)


# 整合获得位置点击流和成绩
df_merge = pd.merge(df_grade, df_time_record, on=["uid", "vid"], how="inner")[["timerecord", "grade"]]
# 处理成绩为0、1的标签
df_merge["grade"] = df_merge["grade"].apply(lambda x: 1 if x == 100 else 0)
# 处理点击流数据，给其中小于10的数字加上符号o,这样再使用文本向量化工具时才能被当作一个feature
df_merge["timerecord"] = df_merge["timerecord"].apply(convert_time_record)

# 将文本数据转换为数字矩阵

# # 使用CountVectorizer
# vectorizer = CountVectorizer(ngram_range=(2, 2))

# 使用TfidfVectorizer
vectorizer = TfidfVectorizer()

# # 使用HashingVectorizer
# vectorizer = HashingVectorizer()

# 文本向量化转换
X = vectorizer.fit_transform(df_merge["timerecord"])
names = vectorizer.get_feature_names()
# 将X转为成数组在转化成numpy ndarray
data = np.array(X.toarray())

# 数据归一化
data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 使用pca降维
pca = PCA(n_components=20)
pca.fit(data_normalized)
reduced_data = pca.transform(data_normalized)


# 朴素贝叶斯模型
bys = GaussianNB()
# K-邻近模型
knn = KNeighborsClassifier(n_neighbors=10)
# 决策树
tree = DecisionTreeClassifier()
# svm
s_vm = svm.SVC()
# 记录每次准确率的列表
acc_list_bys, acc_list_knn, acc_list_tree, acc_list_svm = [], [], [], []

for i in range(1000):
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, df_merge["grade"], test_size=0.3)

    # 使用朴素贝叶斯模型
    bys.fit(x_train, y_train)
    # 计算模型在测试集上的准确率
    accuracy = bys.score(x_test, y_test)
    acc_list_bys.append(accuracy)

    # 使用K-邻近模型
    knn.fit(x_train, y_train)
    # 计算模型在测试集上的准确率
    accuracy = knn.score(x_test, y_test)
    acc_list_knn.append(accuracy)

    # 使用决策树模型
    tree.fit(x_train, y_train)
    # 计算模型在测试集上的准确率
    accuracy = tree.score(x_test, y_test)
    acc_list_tree.append(accuracy)

    # 使用svm模型
    s_vm.fit(x_train, y_train)
    # 计算模型在测试集上的准确率
    accuracy = s_vm.score(x_test, y_test)
    acc_list_svm.append(accuracy)

avg_bys = np.mean(np.array(acc_list_bys), axis=0)
avg_knn = np.mean(np.array(acc_list_knn), axis=0)
avg_tree = np.mean(np.array(acc_list_tree), axis=0)
avg_svm = np.mean(np.array(acc_list_svm), axis=0)

print("Accuracy:", accuracy)
