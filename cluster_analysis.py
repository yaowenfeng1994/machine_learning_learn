from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
# plt.scatter(x[:, 0], x[:, 1], c="white", marker="o", edgecolors="black", s=50)
# plt.grid()
# plt.show()

km = KMeans(n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(x)

# plt.scatter(x[y_km == 0, 0], x[y_km == 0, 1], c="green", marker="s", edgecolors="black", s=50, label="cluster 1")
# plt.scatter(x[y_km == 1, 0], x[y_km == 1, 1], c="orange", marker="o", edgecolors="black", s=50, label="cluster 2")
# plt.scatter(x[y_km == 2, 0], x[y_km == 2, 1], c="blue", marker="v", edgecolors="black", s=50, label="cluster 3")
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker="*", c="red", edgecolors="black",
#             label="centroids")
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()

