import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from traditional_way.data_reader.notebook_reader import df_total_notes
from traditional_way.data_reader.like_collection_list_reader import df_collections_edu, df_likes_edu, df_collections_db, df_likes_db
import matplotlib.pyplot as plt

"""
项目介绍：
通过学生的学期平均特征，采用简单的逻辑回归模型，进行Final Grade Prediction
数据集包括：教育技术学2022，数据库2022
特征包括：反思总条数，反思平均字数；点击流播放事件数、快进事件数、快退事件数、更改速率数，以及视频播放总数；笔记总数
"""

"""
反思数据处理：获取【反思字数】和【反思总数】
"""
# 教育技术课程
df_ref_edu = pd.read_csv("../../mydata/processed/processed_reflection.csv")
df_ref_edu = df_ref_edu.drop(columns="ref_diff")
df_ref_edu = df_ref_edu.groupby("uid").agg({"cid": "size", "ref_len": "mean"}).rename(columns={"cid": "ref_num"}).reset_index()

# 数据库课程
# 读取数据库课程反思
df_ref_db = pd.read_csv("../../mydata/processed/sorted_reflection_database_2022.csv")
# 从反思内容中，获取字数
df_ref_db["ref_len"] = df_ref_db["content"].apply(
    lambda e: len(re.findall(u'[\u4e00-\u9fff]', e)) + len(re.findall(r'\b\w+\b', e)))
# 按照学生进行聚类，获取反思条数和平均反思字数
df_ref_db = df_ref_db.groupby("uid").agg({"content": "size", "ref_len": "mean"}).reset_index().rename(columns={"content": "ref_num"})

# 合并两门课程的反思数据
df_ref = pd.concat([df_ref_db, df_ref_edu], axis=0)


"""
点击流数据数据处理：获取【播放事件数】、【快进事件数】、【快退事件数】、【更改速率事件数】、【视频观看总数】
"""
# 教育技术课程
df_record_edu = pd.read_csv("../../mydata/processed/processed_time_record.csv")
# 获取数据库课程
df_record_db = pd.read_csv("../../mydata/processed/processed_time_record_db.csv")
# 合并两门课程
df_record = pd.concat([df_record_db, df_record_edu], axis=0)
# 通过学生id计算课程平均事件数
df_record = df_record.groupby("uid").agg({"paused": "mean", "skip_back": "mean", "skip_forward": "mean", "rate_change": "mean", "watch_num": "mean"}).reset_index()

"""
笔记处理：获取【笔记总数】,直接导入notebook_reader包下的df_total_notes
"""
"""
视屏后测验处理：获取视屏后的【平均答题时长】，【测验平均成绩】,【测验答题顺序】
"""
# 读取数据
df_quiz_edu = pd.read_csv("../../mydata/processed/quiz_grade_edutec.csv")
# 获取每次测验的测验顺序
df_quiz_edu["q_order"] = df_quiz_edu[["uid", "vid", "quiz_time"]].sort_values(["vid", "quiz_time"]).groupby(["vid"]).cumcount()
# 读取数据
df_quiz_db = pd.read_csv("../../mydata/processed/quiz_grade_database_2022.csv")
# 获取每次测验的测验顺序
df_quiz_db["q_order"] = df_quiz_db[["uid", "vid", "quiz_time"]].sort_values(["vid", "quiz_time"]).groupby(["vid"]).cumcount()
# 合并
df_quiz = pd.concat([df_quiz_edu, df_quiz_db], axis=0).reset_index(drop=True)
# 按学生id聚类，计算平均答题时长和平均测验成绩
df_quiz = df_quiz.groupby("uid").agg({"used_time(s)": "mean", "grade": "mean", "q_order": "mean"}).rename(
    columns={"used_time(s)": "q_time", "grade": "q_grade"}).reset_index()
# df_quiz_db = df_quiz_db.groupby("uid").agg({"used_time(s)": "mean", "grade": "mean", "q_order": "mean"}).rename(
#     columns={"used_time(s)": "q_time", "grade": "q_grade"}).reset_index()
df_quiz = df_quiz.groupby("uid").agg({"used_time(s)": "mean", "grade": "mean", "q_order": "mean"}).rename(columns={"used_time(s)": "q_time", "grade": "q_grade"}).reset_index()

"""
点赞收藏数据处理：读取数据的任务由like_collection_list_reader处理，直接导入即可
"""
df_likes = pd.concat([df_likes_db.drop("vid", axis=1), df_likes_edu.drop("vid", axis=1)], axis=0)
df_likes = df_likes.groupby("uid")["is_like"].sum().reset_index().rename(columns={"is_like": "likes"})
df_collections = pd.concat([df_collections_db.drop("vid", axis=1), df_collections_edu.drop("vid", axis=1)], axis=0)
df_collections = df_collections.groupby("uid")["is_collect"].sum().reset_index().rename(columns={"is_collect": "collections"})
# 合并点赞和收藏，其实可以将这两个特征进行合并
df_likes_collections = pd.merge(df_collections, df_likes, on="uid", how="outer").fillna(0)

"""
trace和quiz特征提取：
"""


"""
课程最终成绩处理：预测任务的Label
"""
# 获取教育技术课程成绩
df_grade_edu = pd.read_excel("../../mydata/row/成绩登记表.xlsx", usecols=[2, 6], names=["uid", "grade"])
# 获取数据库课程成绩
df_grade_db = pd.read_excel("../../mydata/row/数据库成绩表.xlsx", usecols=[2, 6], names=["uid", "grade"])
# 合并两门课程的成绩
df_grade = pd.concat([df_grade_db, df_grade_edu], axis=0)
# 将成绩进行0，1类别划分，划分，取分数为88
df_grade["grade"] = df_grade["grade"].apply(lambda e: 1 if e >= 88 else 0)

"""
特征合并：
特征之间的合并都采用outer方式，特征与成绩之间的合并采用left方式
"""
# 合并反思特征和点击流特征，采用outer方式
df_data = pd.merge(df_record, df_ref, on="uid", how="outer").fillna(0)
# 合并笔记特征
df_data = pd.merge(df_data, df_total_notes, on="uid", how="outer").fillna(0)
# 合并视频测验特征

df_data = pd.merge(df_data, df_quiz, on="uid", how="outer")
# 给‘q_order’为nan的值填充对应课程人数值
df_data["q_order"] = df_data.groupby("uid")["q_order"].transform(
    lambda x: len(df_grade_edu) if (pd.isnull(x.iloc[0]) and df_data.iloc[x.index.to_list()[0], 0] in df_grade_edu["uid"].to_list())
    else len(df_grade_db) if(pd.isnull(x.iloc[0])) else x.iloc[0])
# 其他两个字段q_time,q_grade填充为0
df_data = df_data.fillna(0)

df_data = pd.merge(df_data, df_quiz, on="uid", how="outer").fillna(0)

# 合并点赞收藏也正
df_data = pd.merge(df_data, df_likes_collections, on="uid", how="outer").fillna(0)

# 合并数据和成绩标签
df_data = pd.merge(df_data, df_grade, on="uid", how="left").fillna(0)

"""
训练前数据预处理
"""
# 获取样本数据，并转化为numpy类型
data = df_data.iloc[:, 1:14].to_numpy().astype("float32")
# 数据归一化,如果某列方差已经为0，则不需要归一化
data_normalized = np.where(data.std(axis=0) == 0, data, (data - np.mean(data, axis=0)) / np.std(data, axis=0))

# 给numpy数据中增加一列，作为每行数据的标记，在划分数据集时拿出来，但是实现查看哪些数据在测试集效果更差的目的
data_normalized = np.concatenate((data_normalized, np.arange(0, len(df_data)).reshape(-1, 1)), axis=1).astype("float32")
# 获取标签数据，并转化为numpy类型
label = df_data.iloc[:, 14].to_numpy().astype("float32")[:, np.newaxis]

# 准备数据
# dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).to(torch.float32), torch.from_numpy(label))
# train_data, test_data = train_test_split(dataset, test_size=0.15, random_state=1)


# 定义模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=13, out_features=1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


model = LogisticRegression()

# 定义损失函数和优化器
criterion = nn.BCELoss()
# nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(epoch, iters):
    # 训练模型
    num_epochs = epoch
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, label, test_size=0.2)
    # 获取参与测试的样本id
    index = x_train[:, -1]
    for epoch in range(num_epochs):
        # 将数据转换为张量
        # 截取数据部分，而忽略最后一列的样本id记录
        inputs = torch.from_numpy(x_train[:, 0:-1])
        labels = torch.from_numpy(y_train)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每100个epoch输出一次损失值
        # if (epoch+1) % 100 == 0:
            # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


    # 测试模型
    with torch.no_grad():
        # 将数据转换为张量
        # 截取数据部分，而忽略最后一列的样本id记录
        inputs = torch.from_numpy(x_test[:, 0:-1])
        labels = torch.from_numpy(y_test)
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == labels).float().mean()
        # 添加当前训练轮次的准确率到准确率list
        accuracy_list.append(accuracy)
        # 记录数据准备率矩阵
        train_item_acc_2d[np.setdiff1d(np.arange(0, len(df_data)), index), iters] = accuracy
        print('Test Accuracy: {:.2f}%'.format(accuracy.item() * 100))


# 记录每次训练的测试集准确率
accuracy_list = []
# 构建一个二维数组，存放每条数据若参数测试，其在测试上的准确率
train_item_acc_2d = np.zeros((len(df_data), 500))
for iters in range(500):
    train(epoch=1000, iters=iters)
# 根据每条数据参与测试时的准确率数组，来获取其参与测试的平均准确率
train_item_acc = np.mean(train_item_acc_2d, axis=1, where=train_item_acc_2d > 0.1)
# 填充nan值
train_item_acc = np.nan_to_num(train_item_acc, nan=0)

print("平均准确率{}，最低准确率{}, 最高准确率{}".format(np.mean(np.array(accuracy_list)), np.min(np.array(accuracy_list)), np.max(np.array(accuracy_list))))

"""
绘图
"""
fig, ax = plt.subplots()
ax.scatter(list(np.arange(0, len(df_data))), list(train_item_acc))
# 添加图例
ax.legend()

ax.set_title('Item Acc')
ax.set_xlabel('item')
ax.set_ylabel('acc')

ax.set_xticks(list(np.where(train_item_acc < 0.72)[0]),
              labels=list(np.where(train_item_acc < 0.72)[0]))

plt.show()

print("ok")

