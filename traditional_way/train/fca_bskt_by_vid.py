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
区别于fca_bskt项目，本项目在每个视频下单独划分训练集和测试集，即按照视频进行预测；
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
df_merge_all = pd.merge(df_merge, df_quiz_inf, on=["uid", "vid"], how="inner")
# 合并notebook数据
df_merge_all = pd.merge(df_merge_all, df_notes_edu, on=["uid", "vid"], how="left")
df_merge_all["notebooks"] = df_merge_all["notebooks"].fillna(0).astype("int64")
# 合并like_list
df_merge_all = pd.merge(df_merge_all, df_likes_edu, on=["uid", "vid"], how="left")
df_merge_all["is_like"] = df_merge_all["is_like"].fillna(0).astype("int64")
# 合并collection_list
df_merge_all = pd.merge(df_merge_all, df_collections_edu, on=["uid", "vid"], how="left")
df_merge_all["is_collect"] = df_merge_all["is_collect"].fillna(0).astype("int64")

bys_vid, svm_vid, tree_vid, knn_vid = [], [], [], []  # 记录四种算法，在每个vid下的表现
x_plot = []  # vid中记录条数太少的不参数，用于记录参与的vids，用作绘图的横轴
data_balance = []  # 用于保存每个vid的正负例比例（均大于0.5，越大越不均衡）
vid_len = []  # 用于保存每个vid的记录数
# 分视频id进行训练，视频id范围为2-25
for vid in range(2, 26):
    # 获取vid的所有记录，并删除uid，vid这两列
    df_merge = df_merge_all[df_merge_all["vid"] == vid].iloc[:, 2:]
    if len(df_merge) > 10 and 0 < len(df_merge[df_merge["grade"] == 0]) < len(df_merge):
        x_plot.append(vid)  # 记录训练的vid
        b = len(df_merge[df_merge["grade"] == 0])/len(df_merge)  # 正负例比例
        data_balance.append(b if b > 0.5 else 1-b)  # 设置正负例比例均大于0.5
        vid_len.append(len(df_merge))  # 记录vid的记录数

        # dis = df_merge.describe()
        # 计算属性之间的相关系数
        # corr = df_merge.corr()

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
        pca = PCA(n_components=min(n_samples, n_features))
        pca.fit(data_normalized)
        reduced_data = pca.transform(data_normalized)

        # 取df_merge中除了timerecord之外的数据，转成numpy，归一化之后并与处理好的timerecord进行拼接
        x = df_merge.loc[:, ["completion", "visit_num", "online", "interval", "hour", "is_week", "notebooks", "is_like", "is_collect"]].to_numpy()
        x = np.where(x.std(axis=0) == 0, x, (x - np.mean(x, axis=0)) / np.std(x, axis=0))
        data = np.concatenate([reduced_data, x], axis=1)

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


# 绘制轮廓系数图像
plt.scatter(x_plot, bys_vid, label="bys")
plt.scatter(x_plot, knn_vid, label="knn")
plt.scatter(x_plot, tree_vid, label="tree")
plt.scatter(x_plot, svm_vid, label="svm")
plt.plot(x_plot, data_balance, label="data_portion")
plt.plot(x_plot, list(np.array(vid_len)/np.array(vid_len).max()), label="data_size")
plt.text(0.95, 0.01, 'max video size:'+str(np.array(vid_len).max()),
        verticalalignment='bottom', horizontalalignment='right')
# 添加图例
plt.legend()

plt.title('ACC BY VID')
plt.xlabel('Number of vid')
plt.ylabel('avg acc (1000s)')

plt.show()


