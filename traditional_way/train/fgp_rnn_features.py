import pandas as pd
import matplotlib.pyplot as plt


"""
项目介绍：
使用七个特征进行按章节的聚合，用于RNN模型中，进行Final Grade Prediction
数据集：教育技术学2022
特征：反思字数，反思总数；点击流快进、快退、暂停、倍速、观看总数
"""

"""
数据预处理
"""
plt.ioff()
df_record = pd.read_csv("../../mydata/processed/processed_time_record.csv")

df_reflection = pd.read_csv("../../mydata/processed/processed_reflection.csv")

df_grade = pd.read_excel("../../mydata/row/成绩登记表.xlsx", usecols=[2, 6], names=["uid", "grade"])

# # 给ref_len来划分区别，以100长度为区别，划分六个区间 a = df_reflection["ref_len"].groupby( by=lambda x: 1 if df_reflection["ref_len"][
# x] < 100 else 2 if df_reflection["ref_len"][x] < 200 else 3 if df_reflection["ref_len"][x] < 300 else 4 if
# df_reflection["ref_len"][x] < 400 else 5 if df_reflection["ref_len"][ x] < 500 else 6).count()


# 给ref_diff来排序，而不是计算具体时间差值
df_reflection["ref_diff"] = df_reflection[["uid", "ref_diff", "cid"]].sort_values(["cid", "ref_diff"]).groupby(
    "cid").cumcount() + 1
# 合并reflection和record两列
df_all = pd.merge(df_reflection, df_record, on=["uid", "cid"], how="outer")
# 构建每个学生×每个章节的数据框：使用merge的cross方式，所传入的两个都要求是dataframe而不能是series
df_test = pd.merge(df_grade["uid"], pd.Series(range(1, 14), name="cid").to_frame(), how="cross")
# 合并all和helper两列，得到所有学生的所有单元的数据，all中存在重复的uid、cid（第七章），所以使用drop_duplicates方法进行处理；也可以使用groupby进行处理；
# 最终得到结果就是学生人数×单元数，即59×13=767
df_all = pd.merge(df_all.drop_duplicates(subset=["uid", "cid"], keep="first"), df_test, on=["uid", "cid"], how="right")
# 对缺省值进行处理
df_all["ref_diff"] = df_all["ref_diff"].fillna(len(df_grade))  # ref_diff缺省值填写为学生人数，即默认为最后一名
df_all = df_all.fillna(0)  # 其他的缺省值都填充0

"""
模型构建
"""
# 数据集
data = df_all.iloc[:, 2:].to_numpy().reshape(59, -1, 7)
# 查看成绩分布，使用88分作为分界线
s = df_grade.describe()
# 构建0、1标签
label = df_grade["grade"].apply(lambda e: 1 if e >= 88 else 0)
# 在RNN中可能需要将二分类构造成one-hot表示，即两个列
label = pd.DataFrame({"0": label, "1": (~label.astype(bool)).astype(int)}).to_numpy()

