import pandas as pd

df_notes_edu = pd.read_csv("../../mydata/row/notebook_edutec.csv", header=None, usecols=[1, 2], names=["uid", "vid"])
df_notes_edu = df_notes_edu[~df_notes_edu.index.isin(pd.Series([17, 18]))]
df_notes_edu["uid"] = df_notes_edu["uid"].astype("int64")
df_notes_edu = df_notes_edu.groupby(["uid", "vid"]).size().reset_index(name="notebooks")

df_notes_db = pd.read_csv("../../mydata/row/notebook_database_2022.csv", header=None, usecols=[1, 2], names=["uid", "vid"])
df_notes_db = df_notes_db[~df_notes_db.index.isin(pd.Series([0, 1, 2]))]
df_notes_db["uid"] = df_notes_db["uid"].astype("int64")
df_notes_db = df_notes_db.groupby(["uid", "vid"]).size().reset_index(name="notebooks")

df_notes_all = pd.concat([df_notes_edu, df_notes_db], axis=0)
df_total_notes = df_notes_all.groupby("uid").agg({"notebooks": sum}).rename(columns={"notebooks": "notes_num"}).reset_index("uid")
# print("ok")
