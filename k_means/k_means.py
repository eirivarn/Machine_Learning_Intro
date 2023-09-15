import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class KMeans:
    
    def __init__(self, k=2, max_iterations=100, tol=0.0001, restart_threshold=1, max_restarts=100):
        """
        Initializes the KMeans classifier.
        
        Parameters:
        - k (int): Number of clusters.
        - preprocess (bool): Whether to preprocess the data (normalize).
        - max_iterations (int): Maximum number of iterations for KMeans clustering.
        - tol (float): Tolerance level for checking convergence.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = None  
        self.clusters = None
        self.restart_threshold = restart_threshold
        self.max_restarts = max_restarts

    def preprocess(self, X):
        """
        Standardizes a pandas DataFrame.
        
        Parameters:
        - df: Input pandas DataFrame
        
        Returns:
        - Standardized pandas DataFrame
        """
        return (X - X.mean()) / X.std()

    def fit(self, X):
        """
        Estimates parameters for the classifier using KMeans algorithm.

        Args:
        - X (DataFrame): Input data to cluster.
        """

        # Store dimensions of data
        self.m, self.n = X.shape
        # Store data values for easier referencing
        self.X_values = X.values

        best_silhouette = -np.inf  # Initialize with a very low value
        restarts = 0

        while restarts < self.max_restarts:
            # Initialize centroids using the KMeans++ method
            self.centroids = self._initialize_centroids_PP(X)
            # Assign data points to the initialized clusters
            self.clusters = self._assign_to_clusters()

            # Iterate and refine cluster assignments and centroids
            for _ in range(self.max_iterations):
                prev_centroids = np.copy(self.centroids)
                # Compute new centroids based on current cluster assignments
                self.centroids = self._compute_centroids()
                # Reassign data points to clusters based on new centroids
                self.clusters = self._assign_to_clusters()

                # Calculate silhouette score after reassignment
                current_silhouette = euclidean_silhouette(X, self.predict(X))

                # If the silhouette score does not improve, break out of the inner loop
                if current_silhouette <= best_silhouette:
                    break

                best_silhouette = current_silhouette

                # Check for convergence using the tolerance level
                if np.allclose(self.centroids, prev_centroids, atol=self.tol):
                    break

            # If the silhouette score is above the restart threshold, break out of the outer loop
            if best_silhouette > self.restart_threshold:
                break

            restarts += 1

        # Store the best silhouette score
        self.best_silhouette = best_silhouette

    def predict(self, X):
        """
        Generates cluster predictions for each data point.

        Args:
        - X (DataFrame): Data for which to predict clusters.
        
        Returns:
        - array: Cluster assignments for each data point.
        """
        y_pred = np.zeros(self.m, dtype=int)
        clusters = self._assign_to_clusters()
        # Assign cluster labels to each data point
        for cluster_idx, cluster in enumerate(clusters.values()):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-means algorithm.
        
        Returns:
        - array: Centroid values.
        """
        return self.centroids
        
    def _initialize_centroids_simple(self, X):
        """
        Simple method to initialize centroids by randomly selecting k data points.
        
        Args:
        - X (DataFrame): Data from which to select initial centroids.
        
        Returns:
        - array: Initial centroid values.
        """
        # Randomly select k data points as initial centroids
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        centroids = X.iloc[random_indices].values
        return centroids

    def _initialize_centroids_PP(self, X):
        """
        Initialize centroids using KMeans++.
        
        Args:
        - X (DataFrame): Data from which to select initial centroids.
        
        Returns:
        - array: Initial centroid values using KMeans++ method.
        """
        # Start by randomly selecting the first centroid
        centroids = [X.iloc[np.random.choice(range(X.shape[0]))].values]
        
        # Select remaining centroids based on distance probabilities
        for _ in range(1, self.k):
            # Calculate the distance of each data point to the nearest centroid
            distances = np.array([min(euclidean_distance(p, centroid) for centroid in centroids) for p in X.values])
            # Convert distances to probabilities
            probs = distances / distances.sum()
            # Select the next centroid based on computed probabilities
            next_centroid = X.iloc[np.random.choice(range(X.shape[0]), p=probs)].values
            centroids.append(next_centroid)
            
        return np.array(centroids)
    
    def _assign_to_clusters(self):
        """
        Assign each datapoint to a cluster based on distance to centroids.
        
        Returns:
        - dict: Dictionary with cluster assignments.
        """
        clusters = {i: [] for i in range(self.k)}
        # Iterate over each data point and assign it to the closest centroid's cluster
        for idx, data_point in enumerate(self.X_values): 
            distances = euclidean_distance(data_point, self.centroids)
            cluster_num = np.argmin(distances)
            clusters[cluster_num].append(idx)
        return clusters

    def _compute_centroids(self):
        """
        Compute new centroid values based on current cluster assignments.
        
        Returns:
        - array: New centroid values.
        """
        new_centroids = []

        # For each cluster, compute its centroid as the mean of its data points
        for cluster_indices in self.clusters.values():
            if len(cluster_indices) == 0:
                new_centroids.append(np.zeros(self.n))
            else:
                cluster_data_points = self.X_values[cluster_indices]
                new_centroids.append(np.mean(cluster_data_points, axis=0))
                
        return np.array(new_centroids)

# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        # Fix: ensure broadcasting by keeping dimensions consistent
        distortion += ((Xc - mu.reshape(1, -1)) ** 2).sum(axis=1).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
