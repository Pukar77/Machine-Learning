import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ----- Generate Synthetic Dataset -----
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# ----- Apply KMeans++ -----
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

# ----- Plot Results -----
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("KMeans++ Clustering")
plt.legend()
plt.show()

# ----- Print Cluster Centers -----
print("Cluster Centers:\n", kmeans.cluster_centers_)
