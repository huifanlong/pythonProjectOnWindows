import pandas as pd

# # 创建两个DataFrame
# df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]})
# df2 = pd.DataFrame({'key': ['C', 'D', 'E', 'F'], 'value': [5, 6, 7, 8]})
#
# # 将df1和df2按key列进行merge，并只保留df1中重复的行
# result = pd.merge(df1, df2.drop_duplicates('key'), on='key', how='outer', suffixes=('_left', '_right'))
# result = result.fillna({'value_left': result['value_right'], 'value_right': result['value_left']})
# result = result.drop_duplicates('key')
# print(result)

import pandas as pd

# 创建两个DataFrame
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'A': [3, 4, 5], 'B': [6, 7, 8]}, index=['c', 'd', 'e'])

# 按照索引比较两个DataFrame，只找出索引相同的行
result = df1.merge(df2, how='inner', left_index=True, right_index=True, indicator=True)
result = result[result['_merge'] == 'both'].drop(columns='_merge')

print(result)
