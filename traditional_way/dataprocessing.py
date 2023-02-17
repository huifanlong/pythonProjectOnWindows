import pandas as pd
"""
系统浏览轨迹数据trace
"""
# 读取数据  df_trace：9093rows
df_trace = pd.read_csv("../data/learning_trace_edutec.csv", header=None,
                       usecols=[1, 2, 3, 4], names=["vid", "uid", "trace", "datetime"])
# 将时间字符串转换为pandas支持的datetime数据类型
df_trace["datetime"] = pd.to_datetime(df_trace["datetime"], format="%Y-%m-%d %H:%M:%S")
# 按照学号、时间顺序排列,并重新整理index（drop是不将原来的index添加为一列）
df_trace = df_trace.sort_values(by=["uid", "datetime"]).reset_index(drop=True)
# 新增一列数据timediff，其是学生分组之后，计算的上下两次轨迹时间差
df_trace["timediff"] = df_trace.groupby("uid")["datetime"].transform(lambda x: x.diff())
# 将timediff的时间间隔类型转换为表示秒数（自动为float类型）
df_trace["timediff"] = df_trace["timediff"].dt.total_seconds()
# timediff中每个学生的第一行都是Nan，将Nan设置为-1，并整体转换成int类型
df_trace["timediff"] = df_trace["timediff"].fillna(-1).astype("int64")
visit_interval = 60 * 60  # 设置算一次新访问的最低间隔时间，单位是秒
# 新增一列数据visit_flag,其为1的话表示该次是一次新访问。其根据timediff的时间差，与visit_interval比较来设置
df_trace["visit_flag"] = df_trace["timediff"].apply(lambda x: 1 if x == -1 or x > 60 * 60 else 0)
# 计算每个学生的系统登录次数：按学生分组后计算每组的visit_flag值相加
df_visit_num = df_trace.groupby("uid")["visit_flag"].sum().reset_index()

"""
视频位置点击流数据
"""
# 读取数据  df_time_record:1017rows
df_time_record = pd.read_csv("C:/Users/DELL/PycharmProjects/pythonProject/data/time_record(1)_edutec.csv", header=None,
                             usecols=[1, 2, 3, 6], names=["uid", "vid", "timerecord", "num"])
# 排序
df_time_record = df_time_record.sort_values(by=["uid", "vid"]).reset_index(drop=True)
min_len = 30  # 设定用于过滤的数据最低长度
# 过滤长度低于min_len的数据，df_time_record:929rows
df_time_record = df_time_record[df_time_record["timerecord"].apply(len) >= min_len].reset_index(drop=True)

"""
答题记录数据
"""
df_grade = pd.read_excel("../data/quiz_record_edutec_converted.xlsx", header=None, usecols=[1, 2, 6, 8, 9],
                         names=["uid", "vid", "quiz_time", "used_time(s)", "grade"])
# 将时间字符串转换为pandas支持的datetime数据类型
df_grade["quiz_time"] = pd.to_datetime(df_grade["quiz_time"])
# 将用时转换为timedelta类型
df_grade["used_time(s)"] = pd.to_timedelta(df_grade["used_time(s)"]).dt.total_seconds().astype("int64")
# 排序
df_grade = df_grade.sort_values(by=["uid", "vid"]).reset_index(drop=True)

