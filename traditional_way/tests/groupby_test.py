import pandas as pd

# 创建DataFrame
df = pd.DataFrame({'姓名': ['小明', '小红', '小张', '小明', '小红', '小张'],
                   '科目': ['语文', '语文', '语文', '数学', '数学', '数学'],
                   '分数': [80, 70, 90, 85, 75, 95]})

# 按科目和姓名分组，并计算平均分
df_grouped = df.groupby(['科目', '姓名']).mean()

# 将姓名转换为列索引
df_unstacked = df_grouped.unstack()

# 输出结果
print(df_unstacked)