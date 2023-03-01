import pandas as pd

"""
将点击流数据排序后保存，用于做event——encoding
"""
# df_time_record = pd.read_csv("../mydata/time_record_edutec.csv", header=None,
#                              usecols=[1, 2, 3, 4, 5, 6], names=["uid", "vid", "timerecord", "rate", "time", "num"])
# df_time_record = df_time_record.sort_values(["uid", "vid"]).reset_index(drop=True)
# df = df_time_record.groupby("vid").size()
# df_time_record.to_csv("../mydata/processed/record_sorted.csv", index=None)

"""
读取并处理视频位置点击流数据
"""
# 读取数据  df_time_record:1017rows
df_time_record = pd.read_csv("../../mydata/row/time_record_edutec.csv", header=None,
                             usecols=[1, 2, 3, 6], names=["uid", "vid", "timerecord", "num"])

# 排序
df_time_record = df_time_record.sort_values(by=["uid", "vid"]).reset_index(drop=True)
min_len = 80  # 设定用于过滤的数据最低长度，正常速率下，每停留一秒至少有占两个len
# 过滤长度低于min_len的数据，df_time_record:929rows
df_time_record = df_time_record[df_time_record["timerecord"].apply(len) >= min_len].reset_index(drop=True)

# 计算每个视频的长度：尝试通过相同vid的record的最大秒数来确定
# 1.通过record来计算当前算出的最大的视频位置
df_time_record["v_len"] = df_time_record["timerecord"].apply(lambda x: sorted([int(e) for e in x.strip().split(" ")])[-1])
# 通过对vid进行聚合，并对e_len实行max的transform，就可以设定最大的v_len作为视频的最终v_len
df_time_record["v_len"] = df_time_record.groupby("vid")["v_len"].transform(max)

# 【观看完整度】获取
# 1.先根据record获取停留的位置数，把record变成list，再用set()来去重，在变成list的求len
df_time_record["completion"] = df_time_record["timerecord"].apply(lambda x: len(list(set([int(e) for e in x.strip().split(" ")])))-1)
# 2.上面的得到的位置停留数/之前得到的v_len就得到最终的观看完整率completion
df_time_record["completion"] = df_time_record["completion"] / df_time_record["v_len"]

# 认定completion小于0.3的都是脏数据，然后再除掉多余的列
df_time_record = df_time_record[df_time_record["completion"] > 0.3].iloc[:, [0, 1, 2, 5]]
# 保存
df_time_record.to_csv("../mydata/processed/record.csv", index=0)
