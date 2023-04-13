import math

import pandas as pd
import numpy as np

# 读取数据
df_student = pd.read_csv("/home/hadoop/PycharmProjects/OULADdata/studentInfo.csv")
# 过滤数据：课程为BBB，且结果不为Withdrawn；并只截取两个需要的列
df_student = df_student[(df_student.code_module == "BBB") & (df_student.final_result != "Withdrawn")].iloc[:, [2, 11]]
# 将成绩转换成0、1；只有Fail为0，其他都为1
df_student["final_result"] = df_student["final_result"].apply(lambda s: 0 if s == "Fail" else 1)

# 读取数据
df_student_vle = pd.read_csv("/home/hadoop/PycharmProjects/OULADdata/studentVle.csv")
# 过滤数据：课程为BBB，且id_student在上面的df_student.id_student中；
# 这一步使用isin方式得到相同行列索引、相同大小的true、false布尔数据框；再使用any方法按行（axis=1）获取id_student满足条件为True的行；再与课程为BBB的这个筛选逻辑取逻辑与
df_student_vle = df_student_vle[(df_student_vle.code_module == "BBB") & (df_student_vle.isin({"id_student": df_student.id_student.to_list()}).any(axis=1))].iloc[:, 2:]
x = df_student_vle.id_student.unique()
# 过滤数据:只取有课程活动的学生，并且给学生成绩去重（有的学生会有多个成绩）
df_student = df_student[df_student.id_student.isin(df_student_vle.id_student.unique())].drop_duplicates(subset="id_student")

# 读取数据
df_vle = pd.read_csv("/home/hadoop/PycharmProjects/OULADdata/vle.csv")
# 过滤数据:课程为BBB，并选择需要的两个列
df_vle = df_vle[df_vle.code_module == "BBB"].iloc[:, [0, 3]]
activities = df_vle.drop_duplicates(subset="activity_type")  # 查看总共有多少个activity type

# 根据student_vle中的id_site设置其activity_type；使用merge方法，设定方式为“left”，得到该课程中总共有12个activity type
df_student_vle = pd.merge(df_student_vle, df_vle, on="id_site", how="left").drop("id_site", axis=1)


def group_week(index):
    a = df_student_vle.iat[index, 1]
    if a < 0:
        return 0
    else:
        return math.ceil(a/7)


# 按照日期按周分组（group_week方法），再在每个周内部按student id和activity type分组，并unstack，由此生成每个学生每周的各个活动点击数
df_student_vle_week = df_student_vle.groupby(group_week).apply(lambda s: s.groupby(["id_student", "activity_type"])["sum_click"].count()).unstack().reset_index().rename(columns={"level_0": "week"})
# 构建一个panel数据框，表示所有学生所有周的数据
df_panel_week = pd.merge(pd.DataFrame(df_student.id_student), pd.Series(range(0, 40), name="week"), how="cross")
# 根据panel数据框进行merge，并补充缺省值
df_student_vle_week = pd.merge(df_panel_week, df_student_vle_week, on=["week", "id_student"], how="left").fillna(0)
# 构建数据集，转成numpy数据
data_week = df_student_vle_week.iloc[:, 2:].to_numpy().reshape(len(df_student), -1, 12)
# 构建标签集合，二分类任务，是一个one-hot
label_week = pd.DataFrame({"0": df_student.final_result, "1": (~df_student.final_result.astype(bool)).astype(int)}).to_numpy()
# 数据归一化
data_normalized_week = np.where(data_week.std(axis=0) == 0, data_week, (data_week - np.mean(data_week, axis=0)) / np.std(data_week, axis=0))
# 保存数据
np.save('../result_data/data_week.npy', data_normalized_week)
np.save('../result_data/label_week.npy', label_week)

def group_month(index):
    a = df_student_vle.iat[index, 1]
    if a < 0:
        return 0
    else:
        return math.ceil(a/30)


# 按照日期按周分组（group_week方法），再在每个周内部按student id和activity type分组，并unstack，由此生成每个学生每周的各个活动点击数
df_student_vle_month = df_student_vle.groupby(group_month).apply(lambda s: s.groupby(["id_student", "activity_type"])["sum_click"].count()).unstack().reset_index().rename(columns={"level_0": "week"})
# 构建一个panel数据框，表示所有学生所有周的数据
df_panel_month = pd.merge(pd.DataFrame(df_student.id_student), pd.Series(range(0, 10), name="week"), how="cross")
# 根据panel数据框进行merge，并补充缺省值
df_student_vle_month = pd.merge(df_panel_week, df_student_vle_week, on=["week", "id_student"], how="left").fillna(0)
# 构建数据集，转成numpy数据
data_month = df_student_vle_week.iloc[:, 2:].to_numpy().reshape(len(df_student), -1, 12)
# 构建标签集合，二分类任务，是一个one-hot
label_month = pd.DataFrame({"0": df_student.final_result, "1": (~df_student.final_result.astype(bool)).astype(int)}).to_numpy()
# 数据归一化
data_normalized_month = np.where(data_month.std(axis=0) == 0, data_month, (data_month - np.mean(data_month, axis=0)) / np.std(data_month, axis=0))
# 保存数据
np.save('../result_data/data_month.npy', data_normalized_month)
np.save('../result_data/label_month.npy', label_month)

print("ok")