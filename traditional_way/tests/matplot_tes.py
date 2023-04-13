import numpy as np

# 创建示例数据
my_array = np.array([1, 2, 3, 4, 5, 2, 6, 7, 8, 2])

# 获取所有2的索引
indices = np.where(my_array == 2)[0]

# 打印索引
print(indices)
