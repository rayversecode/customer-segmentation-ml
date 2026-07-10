import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering



# Load data
df = pd.read_csv("mall_customers.csv")


# Fix missing Age Group
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

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Estimated Savings (k$)', 'Credit Score']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow Method: find optimal number of clusters ---
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_plot.png')
print("Elbow plot saved as elbow_plot.png")
plt.show()




# --- Build final K-Means model with chosen k ---
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(df['Cluster'].value_counts())

print("\nAverage feature values per cluster:")
print(df.groupby('Cluster')[features].mean())

# Visualize clusters using Income vs Spending Score
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





# --- Silhouette Score ---
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score for k={optimal_k}: {sil_score:.3f}")

# --- Compare silhouette scores across different k values ---
print("\nSilhouette scores for different k values:")
for k in range(2, 8):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}: silhouette score = {score:.3f}")









# --- Hierarchical Clustering ---

# Step 1: Plot dendrogram to visualize how points merge
plt.figure(figsize=(12, 6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, truncate_mode='lastp', p=20)
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Customers (grouped)')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')
print("\nDendrogram saved as dendrogram.png")
plt.show()

# Step 2: Fit Agglomerative Clustering with same k=5 for fair comparison
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['HC_Cluster'] = hc.fit_predict(X_scaled)

print("\nHierarchical Clustering - cluster counts:")
print(df['HC_Cluster'].value_counts())

hc_silhouette = silhouette_score(X_scaled, df['HC_Cluster'])
print(f"\nHierarchical Clustering Silhouette Score (k=5): {hc_silhouette:.3f}")

# Visualize HC clusters
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







from sklearn.cluster import DBSCAN

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN cluster counts (-1 = noise/outliers):")
print(df['DBSCAN_Cluster'].value_counts())

# Only compute silhouette score if more than 1 cluster found (excluding noise)
n_clusters_dbscan = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
print(f"\nNumber of clusters found by DBSCAN: {n_clusters_dbscan}")

if n_clusters_dbscan > 1:
    # Exclude noise points for silhouette calculation
    mask = df['DBSCAN_Cluster'] != -1
    dbscan_silhouette = silhouette_score(X_scaled[mask], df['DBSCAN_Cluster'][mask])
    print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette:.3f}")

# Visualize DBSCAN clusters
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