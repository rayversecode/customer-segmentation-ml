import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("CC GENERAL.csv")


# credit_limit only had 1 missing value, filling with median
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())


# minimum_payments had 313 missing values, also filling with median
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

print("Missing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values remaining")



# selecting features for clustering, dropping cust_id since it's just an identifier
features = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
            'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# reducing dimensions with pca before clustering, since raw features gave weak silhouette scores
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nOriginal number of features: {X_scaled.shape[1]}")
print(f"Number of PCA components (explaining 90% variance): {X_pca.shape[1]}")
print(f"Explained variance ratio per component: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")




# elbow method to find a good number of clusters
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (Credit Card Data)')
plt.savefig('elbow_plot.png')
print("Elbow plot saved as elbow_plot.png")
plt.show()




# checking silhouette scores across a range of k values
print("\nSilhouette scores for different k values:")
for k in range(2, 8):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}: silhouette score = {score:.3f}")




# building the final k-means model, k=2 gave the best silhouette score here
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_pca)

print("\nCluster counts:")
print(df['Cluster'].value_counts())

print("\nAverage feature values per cluster:")
print(df.groupby('Cluster')[features].mean())

final_silhouette = silhouette_score(X_pca, df['Cluster'])
print(f"\nFinal Silhouette Score (k={optimal_k}): {final_silhouette:.3f}")




# visualizing clusters using the first two pca components
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', s=20, alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Credit Card Customer Segments (k={optimal_k}) - PCA Visualization')
plt.colorbar(scatter, label='Cluster')
plt.savefig('cluster_plot.png')
print("\nCluster plot saved as cluster_plot.png")
plt.show()




# dbscan as a third algorithm for comparison
dbscan = DBSCAN(eps=0.8, min_samples=10)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_pca)

print("\nDBSCAN cluster counts (-1 = noise/outliers):")
print(df['DBSCAN_Cluster'].value_counts())

n_clusters_dbscan = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
print(f"\nNumber of clusters found by DBSCAN: {n_clusters_dbscan}")

if n_clusters_dbscan > 1:
    mask = df['DBSCAN_Cluster'] != -1
    dbscan_silhouette = silhouette_score(X_pca[mask], df['DBSCAN_Cluster'][mask])
    print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette:.3f}")
    print(f"Points classified as noise: {(df['DBSCAN_Cluster'] == -1).sum()} out of {len(df)}")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['DBSCAN_Cluster'], cmap='tab10', s=20, alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Credit Card Customer Segments - DBSCAN Clustering')
plt.colorbar(scatter, label='Cluster (-1 = noise)')
plt.savefig('dbscan_cluster_plot.png')
print("\nDBSCAN cluster plot saved as dbscan_cluster_plot.png")
plt.show()