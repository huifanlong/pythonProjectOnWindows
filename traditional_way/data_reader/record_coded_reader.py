import pandas as pd

# 读取进行事件编码的点击流文件
df_coded_record = pd.read_csv("../../mydata/processed/coded_record_edutec.csv", index_col=[0, 1])
# 将两个列重新命名,第一个列其实是record_id,因为是从record_sorted.csv进行编码的，这两个id相同，也用于后续的拼接
df_coded_record = df_coded_record.reset_index(names=["record_id", "event_id"])
# 根据record_id来计算每条点击流数据的事件数
group = df_coded_record.groupby("record_id")

# 读取原始的点击流数据
df_row_record = pd.read_csv("../../mydata/processed/record_sorted.csv")
# 新建一个dataframe用于存储需要的数据列
df_processed_record = df_row_record.iloc[:, [0, 1]]
# 将点击流数据的四种事件数 作为列进行添加
df_processed_record = df_processed_record.assign(paused=group.apply(lambda d: len(d[d.event == 1]))).assign(skip_back=group.apply(lambda d: len(d[d.event == 2]))).assign(skip_forward=group.apply(lambda d: len(d[d.event == 3]))).assign(rate_change=group.apply(lambda d: len(d[(d.event == 4) | (d.event == 5) | (d.event == 6)])))
# 将nan值进行补0操作，因为每条点击流数据至少都有一个播放事件
df_processed_record[["paused", "skip_back", "skip_forward", "rate_change"]] = df_processed_record[["paused", "skip_back", "skip_forward", "rate_change"]].fillna(0).astype("int64")
# 记录视频id与章节之间的关系
vid_to_cid = [[1], [2, 3, 4], [23, 24, 25, 26], [5, 6], [8, 9], [22], [10, 11, 12], [0], [0], [14, 15], [16, 17], [18, 19], [20, 21]]
# 增加一列叫做章节id
# chatgpt完成的代码，使用了next函数，来迭代从()中的生成器取值，如果vid出现在列表所取的元素上，就返回其索引+1，也就是章节id；如果迭代器迭代完还是没有输出值的化，就返回-1
df_processed_record['cid'] = df_processed_record['vid'].apply(lambda x: next((i+1 for i, lst in enumerate(vid_to_cid) if x in lst), -1))

df_cid_filter_record = df_processed_record.groupby(["uid", "cid"]).agg({"paused": sum, "skip_back": sum, "skip_forward": sum, "rate_change": sum, "vid": len}).reset_index()
df_cid_filter_record.rename(columns={"vid": "watch_num"}, inplace=True)

df_cid_filter_record.to_csv("../mydata/processed/processed_time_record.csv", index=False)

