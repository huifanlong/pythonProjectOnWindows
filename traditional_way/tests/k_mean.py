# import pandas as pd
# from sklearn.cluster import KMeans
# import numpy as np
# from sklearn.cluster import SpectralClustering
#
# # 读取数据
# similarity_matrix = pd.read_excel("k_mean.xlsx", header=None)
# similarity_matrix = similarity_matrix.to_numpy()
#
#
# def k_means_clustering(X, k, max_iters=1000):
#     # 随机初始化质心
#     groups = np.random.choice(X.shape[0], k, replace=False)[:, np.newaxis]
#
#     for _ in range(max_iters):
#         clusters = np.argmax(np.array([np.array(X[:, groups[i]]).mean(axis=1) for i in range(k)]), axis=0)
#         groups = [[index for index, ele in enumerate(clusters) if ele == i] for i in range(k)]
#
#     return clusters
#
#
# kmean = k_means_clustering(np.array([[1, 0.8, 0.4], [0.8, 1, 0.2], [0.4, 0.2, 1]]), 2)
# print(kmean)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

# 读取数据余弦相似度
distances = pd.read_excel("k_mean.xlsx", header=None)

# 使用肘方法确定簇的数量
distortions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(1-np.array(distances))
    distortions.append(kmeans.inertia_)

# k-means聚类
kmeans = KMeans(n_clusters=4, random_state=0).fit(distances)

# 输出聚类结果
clusters = kmeans.labels_
for i, cluster in enumerate(clusters):
    print(f"Product {distances.index[i]} belongs to cluster {cluster+1}")

plt.plot(range(1, 10), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
