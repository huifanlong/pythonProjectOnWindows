import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from traditional_way.data_reader.notebook_reader import df_notes_edu
from traditional_way.data_reader.like_collection_list_reader import df_likes_edu, df_collections_edu

"""
项目介绍：
使用四种传统的机器学习（bskt，贝叶斯、svm、knn、决策树）模型，预测学生观看视频后的答题准确率，称之为FCA(First Chance Accuracy)任务
数据集：教育技术2022
特征：['visit_num', 'interval', 'online', 'notebooks', 'is_week', 'is_like', 'hour', 'completion', 'is_collect']以及视频位置点击流（当做文本进行PCA）
"""


def convert_time_record(data):
    list_1 = [("o" + e if len(e) == 1 else e) for e in data.split(" ")]
    return " ".join(list_1)


# 从reader保存的数据文件中读取
df_grade = pd.read_csv("../../mydata/processed/quiz_grade_edutec.csv")
df_quiz_inf = pd.read_csv("../../mydata/processed/quiz_inf.csv")
df_time_record = pd.read_csv("../../mydata/processed/record.csv")

# 将时间字符串转换为pandas支持的datetime数据类型
df_grade["quiz_time"] = pd.to_datetime(df_grade["quiz_time"])

# 整合获得位置点击流record和成绩grade
df_merge = pd.merge(df_grade, df_time_record, on=["uid", "vid"], how="inner")[["uid", "vid", "timerecord", "completion", "grade"]]
# 处理成绩为0、1的标签
df_merge["grade"] = df_merge["grade"].apply(lambda x: 1 if x == 100 else 0)
# 处理点击流数据，给其中小于10的数字加上符号o,这样再使用文本向量化工具时才能被当作一个feature
df_merge["timerecord"] = df_merge["timerecord"].apply(convert_time_record)
# 合并trace数据
df_merge = pd.merge(df_merge, df_quiz_inf, on=["uid", "vid"], how="inner")

# 合并notebook数据
df_merge = pd.merge(df_merge, df_notes_edu, on=["uid", "vid"], how="left")
df_merge["notebooks"] = df_merge["notebooks"].fillna(0).astype("int64")
# 合并like_list
df_merge = pd.merge(df_merge, df_likes_edu, on=["uid", "vid"], how="left")
df_merge["is_like"] = df_merge["is_like"].fillna(0).astype("int64")
# 合并collection_list
df_merge = pd.merge(df_merge, df_collections_edu, on=["uid", "vid"], how="left")
df_merge["is_collect"] = df_merge["is_collect"].fillna(0).astype("int64")

# dis = df_merge.describe()
# 计算属性之间的相关系数,去掉uid，vid
corr = df_merge.iloc[:, 2:].corr()
corr_to_grade = corr.loc[:, "grade"].abs().sort_values(ascending=False).iloc[1:11].index.to_list()
# 将文本数据转换为数字矩阵

# 使用CountVectorizer
vectorizer = CountVectorizer()

# 使用TfidfVectorizer
# vectorizer = TfidfVectorizer()

# 使用HashingVectorizer
# vectorizer = HashingVectorizer()

# 文本向量化转换
X = vectorizer.fit_transform(df_merge["timerecord"])
# names = vectorizer.get_feature_names_out()
# 将X转为成数组在转化成numpy ndarray
data = np.array(X.toarray())


# 数据归一化,如果某列方差已经为0，则不需要归一化
data_normalized = np.where(data.std(axis=0) == 0, data, (data - np.mean(data, axis=0)) / np.std(data, axis=0))

# 使用pca降维，PCA的维度最高为min(n_samples, n_features)，
n_samples, n_features = data_normalized.shape
pca = PCA(n_components=200)
pca.fit(data_normalized)
reduced_data = pca.transform(data_normalized)

# 取df_merge中除了timerecord之外的数据，转成numpy，归一化之后并与处理好的timerecord进行拼接
x = df_merge.loc[:, corr_to_grade].to_numpy()
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
data_all = np.concatenate([reduced_data, x], axis=1)

bys_vid, svm_vid, tree_vid, knn_vid = [], [], [], []  # 记录四种算法，在每个特征下的表现
plot_x = corr_to_grade
plot_x.insert(0, "record")
# 共有是个特征分别来进行测试
for epoch in range(0, 9):
    data = data_all[:, :200] if epoch == 0 else data_all[:, 200+epoch-1:200+epoch+1]
    # 朴素贝叶斯模型
    bys = GaussianNB()
    # K-邻近模型, n_neighbors <= n_samples
    n_neighbors = 10 if n_samples >= 10 else n_samples-3
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # 决策树
    tree = DecisionTreeClassifier()
    # svm
    s_vm = svm.SVC()

    # 记录每次准确率的列表
    acc_list_bys, acc_list_knn, acc_list_tree, acc_list_svm = [], [], [], []

    for i in range(1000):
        # 划分数据集
        x_train, x_test, y_train, y_test = train_test_split(data, df_merge["grade"], test_size=0.3)

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

    bys_vid.append(avg_bys)
    knn_vid.append(avg_knn)
    tree_vid.append(avg_tree)
    svm_vid.append(avg_svm)


bys_vid.append(0.5)
knn_vid.append(0.5)
tree_vid.append(0.5)
svm_vid.append(0.5)

# 绘制轮廓系数图像
plt.scatter(plot_x, bys_vid, label="bys")
plt.scatter(plot_x, knn_vid, label="knn")
plt.scatter(plot_x, tree_vid, label="tree")
plt.scatter(plot_x, svm_vid, label="svm")


# 添加图例
plt.legend()

plt.title('ACC BY Feature')
plt.xlabel('Feature name')
plt.ylabel('avg acc (1000s)')

plt.show()
