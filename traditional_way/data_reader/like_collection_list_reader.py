import pandas as pd

df_likes = pd.read_csv("../../mydata/row/like_list_edutec.csv", header=None, usecols=[1, 2, 3], names=["uid", "vid", "is_like"])

df_collections = pd.read_csv("../../mydata/row/collection_list_edutec.csv", header=None, usecols=[1, 2, 3], names=["uid", "vid", "is_collect"])
