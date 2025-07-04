# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Step 2: Create the DataFrame (simulate loading CSV)
# Use actual file in real code: df = pd.read_csv("Mall_Customers.csv")
df =pd.read_csv(r"C:\Users\ASUS\Documents\Mall_Customers.csv")
# Let's assume you've already read your CSV into a DataFrame called df
# So starting from here:
# Preprocessing
df = df.drop("CustomerID", axis=1)  # Drop ID, not useful for clustering

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])  # Male=1, Female=0

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Step 4: Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.show()

# Step 5: Apply K-Means with optimal K (assume 5 for now)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Assign clusters back to DataFrame
df['Cluster'] = cluster_labels

# Step 6: Visualize clusters using PCA (2D)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(pca_features[df['Cluster'] == cluster, 0],
                pca_features[df['Cluster'] == cluster, 1],
                label=f"Cluster {cluster}")
plt.title("Customer Segments via K-Means (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Evaluate using Silhouette Score
sil_score = silhouette_score(scaled_features, cluster_labels)
print(f"Silhouette Score for K={optimal_k}: {sil_score:.3f}")
