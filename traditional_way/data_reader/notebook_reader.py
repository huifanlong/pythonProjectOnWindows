import pandas as pd

df_notes = pd.read_csv("../../mydata/row/notebook_edutec.csv", header=None, usecols=[1, 2], names=["uid", "vid"])
df_notes = df_notes[~df_notes.index.isin(pd.Series([17, 18]))]
df_notes["uid"] = df_notes["uid"].astype("int64")
df_notes = df_notes.groupby(["uid", "vid"]).size().reset_index(name="notebooks")
