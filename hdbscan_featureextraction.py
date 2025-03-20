import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

# --- Feature Extraction ---
# Initialize a list to hold cluster features
cluster_features = []

# Loop through unique clusters (excluding noise (-1))
for cluster in set(hdbscan_labels):
    if cluster != -1:  # Ignore noise
        # Get the points belonging to the cluster
        cluster_data = X_denoised[hdbscan_labels == cluster]
        
        # Calculate cluster centroid (mean of the points in the cluster)
        centroid = np.mean(cluster_data, axis=0)
        
        # Calculate cluster size (number of points)
        size = len(cluster_data)
        
        # Calculate cluster density (standard deviation of points in the cluster)
        density = np.std(cluster_data, axis=0).mean()  # You can change this to calculate specific features
        
        # Append features to the list
        cluster_features.append({
            'Cluster': cluster,
            'Centroid': centroid,
            'Size': size,
            'Density': density
        })

# Convert the list of cluster features into a DataFrame for easy inspection
cluster_features_df = pd.DataFrame(cluster_features)

# --- Display Cluster Features (Centroid, Size, Density) ---
print(cluster_features_df)
