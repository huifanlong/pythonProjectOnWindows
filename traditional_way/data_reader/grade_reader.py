import pandas as pd

"""
格式转换并保存答题记录数据
"""
# 1172rows
df_grade = pd.read_excel("../mydata/quiz_record_edutec_converted.xlsx", header=None, usecols=[1, 2, 6, 8, 9],
                         names=["uid", "vid", "quiz_time", "used_time(s)", "grade"])
# 将时间字符串转换为pandas支持的datetime数据类型
df_grade["quiz_time"] = pd.to_datetime(df_grade["quiz_time"])
# 将用时转换为timedelta类型
df_grade["used_time(s)"] = pd.to_timedelta(df_grade["used_time(s)"]).dt.total_seconds().astype("int64")
# 排序
df_grade = df_grade.sort_values(by=["uid", "vid"]).reset_index(drop=True)

# 保存
df_grade.to_csv("../mydata/processed/grade.csv", index=0)

"""
将答题成绩按章节分数来存储
"""