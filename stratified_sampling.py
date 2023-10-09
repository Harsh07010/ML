import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering

# Generate a synthetic dataset with 3 clusters
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Split the dataset into train and test sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

# Perform spectral clustering on the training data
n_clusters_pred = 3  # Number of clusters to predict
spectral_clustering = SpectralClustering(n_clusters=n_clusters_pred, affinity='nearest_neighbors', random_state=random_state)
predicted_labels = spectral_clustering.fit_predict(X_train)

# Evaluate the clustering performance (you can use different metrics)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y_train, predicted_labels)
print(f"Adjusted Rand Index: {ari}")

