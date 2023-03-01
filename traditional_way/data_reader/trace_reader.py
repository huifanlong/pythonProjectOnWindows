import pandas as pd
from grade_reader import df_grade

"""
读取并处理系统浏览轨迹数据trace
"""
# 读取数据  df_trace：9093rows
df_trace = pd.read_csv("../../mydata/row/learning_trace_edutec.csv", header=None,
                       usecols=[2, 3, 4], names=["uid", "trace", "datetime"])
# 将时间字符串转换为pandas支持的datetime数据类型
df_trace["datetime"] = pd.to_datetime(df_trace["datetime"], format="%Y-%m-%d %H:%M:%S")
# 按照学号、时间顺序排列,并重新整理index（drop是不将原来的index添加为一列）
df_trace = df_trace.sort_values(by=["uid", "datetime"]).reset_index(drop=True)

"""数据过滤，学生可能多次进入quiz页面，只留下产生答题记录的那一次"""
# 1.从df_trace中筛选出trace为quiz的行:1162rows，而真实的答题记录只有1080条，因为有学生可能重复进入页面
df_quiz_filter = df_trace[df_trace["trace"].str.startswith("quiz")]
# 把trace中的quiz*,转成vid
df_quiz_filter["vid"] = df_quiz_filter["trace"].apply(lambda x: int(x.replace("quiz", "")))
# 排序
df_quiz_filter = df_quiz_filter.sort_values(by=["uid", "vid"])


# 2.重复进入答题页面的学生，需要选择其数据库所记录的成绩的那一次，来计算其登陆次数和时间差,以及登录时长
def filter_quiz(x):
    # x是group之后的一个组的dataframe，apply方法会将其传入该方法
    # 当dataframe的长度大于1，说明有重复的trace/vid
    # 通过reset_index来记录filter之前的index（其会被当做一个新的列）
    x = x.reset_index()
    if len(x) > 1:
        # 此时取出grade的时间与其进行比较，隔得近的作为次grade的quiz，其他的就删除
        # 取出uid
        uid_from_quiz_inf = x.iloc[0, 1]
        # 取出vid
        vid_from_quiz_inf = x.iloc[0, 4]  # 如果df_quiz_inf列表变化，这里的参数需要修改
        # 根据uid和vid来取出grade时间
        grade_time = df_grade[(df_grade["uid"] == uid_from_quiz_inf) & (df_grade["vid"] == vid_from_quiz_inf)].iloc[
            0, 2]
        # 与grade时间比较
        x["interval"] = x["datetime"].apply(lambda ele: ele - grade_time if ele >= grade_time else grade_time - ele)
        # 排序，方便有多个纪录时取出第一个（第一个就是跟grade时间最近的）
        x = x.sort_values(by="interval")
        # 切割，只取第一条记录,并去掉建立的interval属性
        result = x.iloc[[0], 0:8]  # 如果df_quiz_inf列表变化，这里的参数需要修改
        return result
    else:
        return x


# 从df_quiz_inf中过滤掉多余的记录，得到1000rows；多余的记录，可能未答题的访问、或已经答题一次后的访问
# 记录数少于做题记录（1080条），可能是trace记录有误；
df_quiz_inf_filtered = df_quiz_filter.groupby(by=["uid", "trace"]).apply(filter_quiz)
# 取出filter之后的index列表，将之前的index和filter之后的index比较，构建一个true、false列；
df_quiz_filter["flag"] = df_quiz_filter.index.isin(df_quiz_inf_filtered["index"].array)
# 根据flag来得到False行的index，在原来的df_trace中直接删除
df_trace = df_trace[~df_trace.index.isin(df_quiz_filter[~df_quiz_filter["flag"]].index.array)].reset_index(drop=True)

"""构造timediff列，从datetime属性中计算两个访问动作之间的时间差；在后续处理会经常用到"""
df_trace["timediff"] = df_trace.groupby("uid")["datetime"].transform(lambda x: x.diff())
# 将timediff的时间间隔类型转换为表示秒数（自动为float类型）根据group、和time_diff列来transform出
df_trace["timediff"] = df_trace["timediff"].dt.total_seconds()
# timediff中每个学生的第一行都是Nan，将Nan设置为-1，并整体转换成int类型
df_trace["timediff"] = df_trace["timediff"].fillna(-1).astype("int64")

visit_interval = 60 * 60  # 设置算一次新访问的最低间隔时间，单位是秒
"""构造visit_flag列，从timediff比较时间差，来判断是不是新的访问；后续用于计算登陆次数"""
df_trace["visit_flag"] = df_trace["timediff"].apply(lambda x: 1 if x == -1 or x > 60 * 60 else 0)
# 利用好timediff中-1作为visit——flag判断的一部分之后，将其设置为0
df_trace["timediff"] = df_trace["timediff"].where(lambda s: s > 0, 0)

flag = 1


# 定义自定义函数：根据trace属性是否出现quiz以及uid属性来划分不同的组。一个组就相当于是两次答题之间。
def custom_func(x):
    global flag
    # x是dataframe的索引
    if x == 0:  # 第一行直接返回初始flag
        uid_now = df_trace.iloc[[x]]["uid"].iloc[0]
        return str(uid_now) + "/" + str(flag)
    else:
        # 根据索引取x的上一行，再取其trace列的值
        trace = df_trace.iloc[[x - 1]]["trace"].iloc[0]
        # 读取uid
        uid_before = df_trace.iloc[[x - 1]]["uid"].iloc[0]
        uid_now = df_trace.iloc[[x]]["uid"].iloc[0]
        if "quiz" in trace and uid_now == uid_before:  # 说明是同一个人，到了一个新quiz:
            flag = flag + 1
        elif uid_now != uid_before:  # 说明是不同的人
            flag = 1
        return str(uid_now) + "/" + str(flag)


def agv_visit(x):
    # 从传入的series取出类别标签“uid/【练习次数】”（custom_func定义的），并取出其【练习次数】
    log_n = int(x.name.split("/")[1])
    # 根据x的index的最大值，以及uid，来匹配用户当前练习为止的总登陆次数
    total_n = \
        df_trace[
            (df_trace["uid"] == df_trace.at[x.index.array[0], "uid"]) & (df_trace.index <= x.index.array[len(x) - 1])][
            "visit_flag"].sum()
    # 总登录次数除以练习次数，就得到当前练习的平均登陆次数。
    return total_n / log_n


"""给一次答题的访问序列分组：根据trace中是否是quiz，以及用户uid"""
group = df_trace.groupby(by=custom_func, axis=0)

"""获取每次答题的【平均登陆次数】，根据group、和visit_flag列按组来transform出"""
df_trace["visit_num"] = group["visit_flag"].transform(agv_visit)

"""获取每次答题距离上次答题的【时间间隔】，根据group、和time_diff列来按组transform出"""
df_trace["time_diff"] = group["timediff"].transform(sum)
# 大于七天的diff设置为7天，防止值差别过大
df_trace["time_diff"] = df_trace["time_diff"].apply(lambda x: 7 * 24 * 60 * 60 if x > 7 * 24 * 60 * 60 else x)

"""获取每次答题的【平均登录时长】，根据timediff来计算出"""
# 根据group、和timediff列来计算【登录时长】，与【timediff总时间】的区别是sum函数不计算visit_flag为1的timediff（以及每个组的第一个，这个不重要）
# 不算计的这两个，前者是说明这是一次退出到一次新的登录的时间差，自然不计入登录时长；第二个是上一次quiz的答题用时，一般都较短所以不重要
# 1.先计算第二个，这两个不能重复减，所以将判断每个组的第一个flag不等于1时才去掉（否则会被重复去掉）
df_trace["online"] = group["timediff"].transform(
    lambda x: x.sum() if df_trace.iloc[x.index.array[0], 4] == 1 else x.tail(len(x) - 1).sum())


def quiz2_online(x):
    # 从输入series中取出index的头和尾
    index_first, index_last = (x.index.array[0], x.index.array[len(x) - 1])
    # 根据index头和尾，来切分timediff和visit_flag,这二者相乘就是得到我们应该要去掉的timediff；然后用输入series的任意值做被减数都行
    quiz_interval = x.array[0] - (df_trace["timediff"].loc[index_first:index_last] * df_trace["visit_flag"].loc[
                                                                                     index_first:index_last]).sum()
    return quiz_interval


def avg_online(x):
    # 从传入的series取出类别标签“uid/【练习次数】”（custom_func定义的），并取出其【练习次数】
    log_n = int(x.name.split("/")[1])
    total_online = \
        df_trace[
            (df_trace["uid"] == df_trace.at[x.index.array[0], "uid"]) & (df_trace.index < x.index.array[len(x) - 1])][
            "online"].unique().sum()
    return total_online / log_n


# 第一次group+transform，是分组让online_q-(timediff*visit_flag).sum(),得到每个quiz距离上个quiz之间的登录时长
df_trace["online"] = group["online"].transform(quiz2_online)
# 第一次group+transform，是得到当前quiz的总时长/quiz次数，得到平均登录时长
df_trace["online"] = group["online"].transform(avg_online)

"""从trace中提取quiz"""
# 1.从df_trace中筛选出trace为quiz的行,并去掉不需要的列:1000rows，真实的答题记录只有1080条，因为有学生可能重复进入页面
df_quiz_inf = df_trace[df_trace["trace"].str.startswith("quiz")].loc[:,
              ["uid", "trace", "datetime", "visit_num", "time_diff", "online"]]
# 把trace中的quiz*,转成vid
df_quiz_inf["vid"] = df_quiz_inf["trace"].apply(lambda x: int(x.replace("quiz", "")))
# 排序
df_quiz_inf = df_quiz_inf.sort_values(by=["uid", "vid"]).reset_index(drop=True)


def get_interval(s):
    # 排序
    s = s.sort_values(ascending=True)
    # 计算与第一个的时间差
    s = s.apply(lambda x: x - s.iat[0])
    # 将时间差转换成秒数，并把大于七天的设置为7天
    return s.dt.total_seconds().astype("int64").apply(lambda e: 7 * 24 * 60 * 60 if e > 7 * 24 * 60 * 60 else e)


"""提取【答题距离】第一个同学答题的时间，当做一种答题积极性衡量"""
df_quiz_inf["interval"] = df_quiz_inf.groupby(["vid"])["datetime"].transform(get_interval)

"""获取答题时间，以小时计数，也可以记录属于上午下午晚上哪个类别"""
df_quiz_inf["hour"] = df_quiz_inf["datetime"].apply(lambda x: x.hour)

"""获取是否是周末"""
df_quiz_inf["is_week"] = df_quiz_inf["datetime"].apply(lambda x: 0 if x.dayofweek <= 4 else 1)

# 选择需要的列，并排序
df_quiz_inf = df_quiz_inf[["uid", "vid", "visit_num", "online", "interval", "hour", "is_week"]].sort_values(
    by=["uid", "vid"]).reset_index(drop=True)

# 保存
df_quiz_inf.to_csv("../mydata/processed/quiz_inf.csv", index=0)
