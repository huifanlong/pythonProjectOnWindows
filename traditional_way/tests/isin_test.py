import pandas as pd

# 创建一个包含不需要的值的DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
a = df.isin({'B': [5, 6]})
b = df.isin({'B': [5, 6]}).any(axis=1)
c = df.isin({'B': [5, 6]}).any(axis=0)
d = df.A == 1
# 删除包含不需要的值的行
df = df.loc[~df.isin({'B': [5, 6]}).any(axis=1)]

# 删除包含不需要的值的列
df = df.loc[:, ~df.isin({'A': [1, 3]}).any(axis=0)]

print(df)