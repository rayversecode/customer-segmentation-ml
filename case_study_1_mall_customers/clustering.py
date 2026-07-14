import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("mall_customers.csv")


# age group had a few missing values, so deriving it from age instead
def get_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return "51+"

df['Age Group'] = df['Age'].apply(get_age_group)


# select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Estimated Savings (k$)', 'Credit Score']
X = df[features]


# scale the features so they're all on comparable ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# elbow method to find a good number of clusters
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_plot.png')
print("Elbow plot saved as elbow_plot.png")
plt.show()




# building the final k-means model, picked k=5 based on the elbow plot
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(df['Cluster'].value_counts())

print("\nAverage feature values per cluster:")
print(df.groupby('Cluster')[features].mean())



# visualize clusters using income vs spending score
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                       c=df['Cluster'], cmap='viridis', s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'Customer Segments (k={optimal_k})')
plt.colorbar(scatter, label='Cluster')
plt.savefig('cluster_plot.png')
print("\nCluster plot saved as cluster_plot.png")
plt.show()




# silhouette score for the chosen k
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score for k={optimal_k}: {sil_score:.3f}")


# checking silhouette scores across a range of k values to confirm k=5 was a good pick
print("\nSilhouette scores for different k values:")
for k in range(2, 8):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}: silhouette score = {score:.3f}")




# hierarchical clustering as a second algorithm for comparison
plt.figure(figsize=(12, 6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, truncate_mode='lastp', p=20)
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Customers (grouped)')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')
print("\nDendrogram saved as dendrogram.png")
plt.show()




# fitting agglomerative clustering with the same k=5 for a fair comparison
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['HC_Cluster'] = hc.fit_predict(X_scaled)

print("\nHierarchical Clustering - cluster counts:")
print(df['HC_Cluster'].value_counts())

hc_silhouette = silhouette_score(X_scaled, df['HC_Cluster'])
print(f"\nHierarchical Clustering Silhouette Score (k=5): {hc_silhouette:.3f}")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                       c=df['HC_Cluster'], cmap='plasma', s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments - Hierarchical Clustering (k=5)')
plt.colorbar(scatter, label='Cluster')
plt.savefig('hc_cluster_plot.png')
print("\nHC cluster plot saved as hc_cluster_plot.png")
plt.show()




# dbscan as a third algorithm, density based instead of needing a fixed k
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN cluster counts (-1 = noise/outliers):")
print(df['DBSCAN_Cluster'].value_counts())


# only compute silhouette score if more than one real cluster was found, excluding noise
n_clusters_dbscan = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
print(f"\nNumber of clusters found by DBSCAN: {n_clusters_dbscan}")

if n_clusters_dbscan > 1:
    mask = df['DBSCAN_Cluster'] != -1
    dbscan_silhouette = silhouette_score(X_scaled[mask], df['DBSCAN_Cluster'][mask])
    print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette:.3f}")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                       c=df['DBSCAN_Cluster'], cmap='tab10', s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments - DBSCAN Clustering')
plt.colorbar(scatter, label='Cluster (-1 = noise)')
plt.savefig('dbscan_cluster_plot.png')
print("\nDBSCAN cluster plot saved as dbscan_cluster_plot.png")
plt.show()