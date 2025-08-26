import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic dataset
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)

# Convert to DataFrame
data = pd.DataFrame(X, columns=['X', 'Y'])

# Apply KMeans
k = 4   # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data[['X','Y']])

# Print sample output
print("First 10 rows of clustered dataset:")
print(data.head(10))

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(data['X'], data['Y'], c=data['Cluster'], cmap='rainbow', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            color='black', marker='X', s=200, label='Centroids')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
