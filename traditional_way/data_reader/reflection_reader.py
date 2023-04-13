import pandas as pd
import re

# # 排序文件 再手动处理
# df_reflections = pd.read_excel("../../mydata/row/reflection_database.xls", header=None, usecols=[1, 3, 4, 5], names=["uid", "content", "title", "time"])
# # 通过cid来聚类，查看聚类后按照时间排序的列表
# df_reflections = df_reflections.sort_values(["uid", "time"]).reset_index(drop=True)
#
# df_reflections.to_csv("../../mydata/processed/sorted_reflection_database_2022.csv", index=None)


# 读取排序后处理了的文件
df_reflections = pd.read_csv("../../mydata/processed/sorted_reflection_database_2022.csv")

# 通过cid来聚类，查看每章最先写的反思时间；与教学时间对比，从而判断cid是否有误填；修正好的数据还是有反思填写时间在教学之间之前，因为学生提前批量完成多章反思
df_reflections = df_reflections.groupby("cid")["time"].apply(lambda s: s.sort_values().head(1)).reset_index()

df_processed_reflection = df_reflections.iloc[:, [0, 4]].rename(columns={0: "uid", 1: "cid"})
# 记录课程上课时间；第七、八次课都是讲的第七章的内容，所以第八次课的时间被移除
course_time = pd.date_range(start="2022-9-1 18:00:00", end="2022-12-1 18:00:00", freq="W-THU").to_series().reset_index(
    drop=True).drop(7).reset_index(drop=True)
# 构造【diff】，计算学生每章记笔记的时间与发布时间的差别
df_processed_reflection["ref_diff"] = df_reflections.groupby("cid")["time"].transform(
    lambda s: (s - course_time.iloc[s.name - 1]).dt.total_seconds().astype("int64").abs() if s.name > 0 else 0)


# 统计【反思内容中的字数】;直接用len进行计算的化，每个英文字母都会计算一个字符长度，因此使用正则表达式找出中文字符数+英文单词数；
# （这里的英文单词数其实也会记录标点符号，其逻辑主要是连续祖母以及标点等字符只记录一个长度；
df_processed_reflection["ref_len"] = df_reflections["reflection"].apply(
    lambda e: len(re.findall(u'[\u4e00-\u9fff]', e)) + len(re.findall(r'\b\w+\b', e)))

df_processed_reflection.to_csv("../../mydata/processed/processed_reflection_database.csv", index=False)
