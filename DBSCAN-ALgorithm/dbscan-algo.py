import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Generate synthetic dataset (3 clusters + noise)
X, _ = make_blobs(n_samples=50, centers=[(2, 2), (8, 8), (15, 2)], cluster_std=0.8, random_state=42)

data = pd.DataFrame(X, columns=['X', 'Y'])

# Apply DBSCAN
eps = 1.5   # neighborhood radius
min_samples = 3  # minimum points to form a core point
db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

# Assign cluster labels
data['Cluster'] = db.labels_

# Identify Core, Border, and Noise points
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

point_type = []
for i, label in enumerate(db.labels_):
    if label == -1:
        point_type.append("Noise")
    elif core_samples_mask[i]:
        point_type.append("Core")
    else:
        point_type.append("Border")

data['Point_Type'] = point_type

print("Dataset with Core, Border, and Noise Points:\n")
print(data.head(15))  # showing first 15 rows

# Visualization
colors = {'Core': 'blue', 'Border': 'green', 'Noise': 'red'}
plt.figure(figsize=(8, 6))
plt.scatter(data['X'], data['Y'], c=[colors[t] for t in data['Point_Type']], s=70, edgecolor='k')

# Annotate points
for i, row in data.iterrows():
    plt.text(row['X']+0.1, row['Y'], row['Point_Type'][0], fontsize=7)  # C=Core, B=Border, N=Noise

plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN: Core, Border, and Noise Points")
plt.show()
