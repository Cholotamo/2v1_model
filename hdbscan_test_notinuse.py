import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle

# --- Load Data (assuming MFCC features here) ---
dataset_path = 'M1_features.npy'  # Update to your dataset path
X = np.load(dataset_path)  # Load the feature data (phase angle, peak amplitude)

# Ensure we are working with the first two features for visualization (Phase Angle and Peak Amplitude)
X_vis = X[:, :2]  # Assuming the first two columns are Phase Angle and Peak Amplitude

# --- Step 1: Visualize Raw Data (Graph 1) ---
plt.figure(figsize=(10, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], s=50, color='blue', alpha=0.5)
plt.title("Raw Data (Phase Angle vs Peak Amplitude)")
plt.xlabel("Phase Angle")
plt.ylabel("Peak Amplitude")
plt.show()

# --- Step 2: Apply KMeans for Baseline Calculation (Graph 2) ---
# Perform KMeans to separate data into two clusters (baseline and the rest)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_vis)

# Visualize the baseline removal (KMeans clusters)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=labels, palette="Set2", s=50, alpha=0.5, legend=None)
plt.title("KMeans Clustering of Data (Baseline vs Data)")
plt.xlabel("Phase Angle")
plt.ylabel("Peak Amplitude")
plt.show()

# --- Step 3: De-noise by Removing Baseline Cluster ---
# Assuming KMeans labeled the baseline as cluster 0, remove it
X_denoised = X_vis[labels == 1]  # Keep the data points not classified as baseline

# --- Step 4: Apply HDBSCAN on De-noised Data (Graph 3 and Graph 4) ---
# Normalize the data before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_denoised)

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)  # Adjust min_cluster_size if needed
hdbscan_labels = clusterer.fit_predict(X_scaled)

# --- Step 5: Visualize Final HDBSCAN Clustering with Boundaries (Graph 4) ---
# Visualizing the clustering results
plt.figure(figsize=(10, 6))

# Get unique clusters (excluding -1 which represents noise)
unique_labels = np.unique(hdbscan_labels)

# Loop through each cluster and plot
for cluster in unique_labels:
    if cluster != -1:  # Ignore noise
        # Get the points belonging to the current cluster
        cluster_data = X_denoised[hdbscan_labels == cluster]
        
        # Plot the points of this cluster
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster}", s=50, alpha=0.7)
        
        # Calculate the bounding box for the cluster
        min_x, min_y = cluster_data.min(axis=0)
        max_x, max_y = cluster_data.max(axis=0)
        
        # Add a rectangle around the cluster (bounding box)
        plt.gca().add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                      linewidth=2, edgecolor='black', facecolor='none'))
        
# Title and labels
plt.title("Final HDBSCAN Clustering with Cluster Boundaries")
plt.xlabel("Phase Angle")
plt.ylabel("Peak Amplitude")
plt.legend()
plt.show()

# --- Step 6: Ablation Study (Compare Clustering Methods) ---
# Perform clustering with DBSCAN, KMeans, and HDBSCAN
dbscan_clusterer = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan_clusterer.fit_predict(X_vis)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_vis)

# Perform HDBSCAN
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
hdbscan_labels = hdbscan_clusterer.fit_predict(X_vis)

# --- Compare Cluster Features for DBSCAN, KMeans, and HDBSCAN ---
def calculate_cluster_features(labels, X_data):
    cluster_features = []
    unique_labels = np.unique(labels)
    
    for cluster in unique_labels:
        # Get the points belonging to the current cluster
        cluster_data = X_data[labels == cluster]
        
        # Calculate cluster centroid (mean of the points in the cluster)
        centroid = np.mean(cluster_data, axis=0)
        
        # Calculate cluster size (number of points)
        size = len(cluster_data)
        
        # Calculate cluster density (standard deviation of points in the cluster)
        density = np.std(cluster_data, axis=0).mean()
        
        # Append features to the list
        cluster_features.append({
            'Cluster': cluster,
            'Centroid': np.linalg.norm(centroid),  # Use the norm (magnitude) of the centroid
            'Size': size,
            'Density': density
        })
    
    return cluster_features

# Get cluster features for each method
dbscan_features = calculate_cluster_features(dbscan_labels, X_vis)
kmeans_features = calculate_cluster_features(kmeans_labels, X_vis)
hdbscan_features = calculate_cluster_features(hdbscan_labels, X_vis)

# Convert the list of cluster features into DataFrames for easy inspection
dbscan_df = pd.DataFrame(dbscan_features)
kmeans_df = pd.DataFrame(kmeans_features)
hdbscan_df = pd.DataFrame(hdbscan_features)

# --- Step 7: Plot Cluster Features (Centroid, Size, and Density) ---
# Centroid Comparison Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='Centroid', data=dbscan_df, color='b', label="DBSCAN", alpha=0.6)
sns.barplot(x='Cluster', y='Centroid', data=kmeans_df, color='g', label="KMeans", alpha=0.6)
sns.barplot(x='Cluster', y='Centroid', data=hdbscan_df, color='r', label="HDBSCAN", alpha=0.6)
plt.title("Centroid Comparison between DBSCAN, KMeans, and HDBSCAN")
plt.xlabel("Cluster")
plt.ylabel("Centroid (Magnitude)")
plt.legend()
plt.show()

# Size Comparison Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='Size', data=dbscan_df, color='b', label="DBSCAN", alpha=0.6)
sns.barplot(x='Cluster', y='Size', data=kmeans_df, color='g', label="KMeans", alpha=0.6)
sns.barplot(x='Cluster', y='Size', data=hdbscan_df, color='r', label="HDBSCAN", alpha=0.6)
plt.title("Cluster Size Comparison between DBSCAN, KMeans, and HDBSCAN")
plt.xlabel("Cluster")
plt.ylabel("Size")
plt.legend()
plt.show()

# Density Comparison Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='Density', data=dbscan_df, color='b', label="DBSCAN", alpha=0.6)
sns.barplot(x='Cluster', y='Density', data=kmeans_df, color='g', label="KMeans", alpha=0.6)
sns.barplot(x='Cluster', y='Density', data=hdbscan_df, color='r', label="HDBSCAN", alpha=0.6)
plt.title("Cluster Density Comparison between DBSCAN, KMeans, and HDBSCAN")
plt.xlabel("Cluster")
plt.ylabel("Density")
plt.legend()
plt.show()
