import pandas as pd

df_likes_edu = pd.read_csv("../../mydata/row/like_list_edutec.csv", header=None, usecols=[1, 2, 3], names=["uid", "vid", "is_like"])

df_collections_edu = pd.read_csv("../../mydata/row/collection_list_edutec.csv", header=None, usecols=[1, 2, 3], names=["uid", "vid", "is_collect"])

df_likes_db = pd.read_csv("../../mydata/row/like_list_database_2022.csv", header=None, usecols=[1, 2, 3], names=["uid", "vid", "is_like"])

df_collections_db = pd.read_csv("../../mydata/row/collection_list_database_2022.csv", header=None, usecols=[1, 2, 3], names=["uid", "vid", "is_collect"])


# print("k")

