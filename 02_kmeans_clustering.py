# KMeans clustering for EV segmentation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_excel("kmeansEVs.xlsx")
data = df[["Battery Capacity", "Driving Range"]]

# Scale
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow + silhouette
k_range = range(2, 11)
wcss = []
sil_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(data_scaled, labels))

# Plot elbow
plt.figure()
plt.plot(k_range, wcss, marker='o')
plt.xlabel("k")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.savefig("results/figures/elbow.png", dpi=300)
plt.show()

# Plot silhouette
plt.figure()
plt.bar(k_range, sil_scores)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.savefig("results/figures/silhouette.png", dpi=300)
plt.show()

# Final clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(data_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot clusters
plt.figure()
plt.scatter(df["Battery Capacity"], df["Driving Range"], c=df["Cluster"])
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
plt.xlabel("Battery Capacity")
plt.ylabel("Driving Range")
plt.title("EV Clustering")
plt.savefig("results/figures/clusters.png", dpi=300)
plt.show()

print("Centroids:", centroids)