import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k=2, preprocess=False, max_iterations=1000, tol=0):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.preprocess = preprocess
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        if self.preprocess:
            X = preprocess_data(X)
        
        self.m, self.n = X.shape
        self.X_values = X.values  # Keep a copy of the data values for referencing
        self.centroids = self.init_centroids_kmeanspp(X)
        self.clusters = self.assign_points_to_clusters(X)
        
        prev_centroids = np.zeros_like(self.centroids)

        for _ in range(self.max_iterations):
            prev_centroids = np.copy(self.centroids)
            self.centroids = self.update_centroid_value(X)
            self.clusters = self.assign_points_to_clusters(X)

            if np.allclose(self.centroids, prev_centroids, atol=self.tol):
                break

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        y_pred = np.zeros(self.m, dtype=int)
        clusters = self.assign_points_to_clusters(X)
        for cluster_idx, cluster in enumerate(clusters.values()):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    def init_centroids_kmeanspp(self, X, k=None):
        """
        Initialize centroids using KMeans++.
        
        Args:
            X (DataFrame): Data.
            k (int, optional): Number of clusters. Defaults to self.k.
            
        Returns:
            numpy.array: Initialized centroids.
        """
        if k is None:
            k = self.k
        
        # Step 1: Choose one center uniformly at random from the data points
        centroids = [X.iloc[np.random.choice(range(X.shape[0]))].values]
        
        # Steps 2-4: Choose the rest of the centroids
        for _ in range(1, k):
            distances = np.array([np.min([euclidean_distance(p, centroid) for centroid in centroids]) for p in X.values])
            probs = distances / distances.sum()
            next_centroid = X.iloc[np.random.choice(range(X.shape[0]), p=probs)].values
            centroids.append(next_centroid)
            
        return np.array(centroids)
    
    def assign_points_to_clusters(self, X):
        """
        Assign each datapoint to a cluster
        """
        clusters = {i: [] for i in range(self.k)}
        for idx, data_point in enumerate(self.X_values): 
            distances = euclidean_distance(data_point, self.centroids)
            cluster_num = np.argmin(distances)
            clusters[cluster_num].append(idx)
        return clusters

    def update_centroid_value(self, X):
        new_centroids = []

        for cluster_indices in self.clusters.values():
            if len(cluster_indices) == 0:
                new_centroids.append(np.zeros(X.shape[1]))
            else:
                cluster_data_points = self.X_values[cluster_indices]
                new_centroids.append(np.mean(cluster_data_points, axis=0))
                
        return np.array(new_centroids)


    
# --- Some utility functions 

def preprocess_data(X):
        X['x1'] = X['x1'] * 9
        return X

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


def determine_optimal_k_differential_elbow(X, k_range=(2, 11)):
    """
    Determine the optimal k value using the Differential Elbow Method.
    
    Args:
        X (DataFrame): The data to cluster.
        k_range (tuple, optional): A tuple indicating the range of k values to check.
                                   Defaults to (2, 15).

    Returns:
        int: Optimal value of k based on the Differential Elbow Method.
    """
    distortions = []

    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(k=k)
        model.fit(X)
        z = model.predict(X)
        
        distortions.append(euclidean_distortion(X, z))
    
    # Compute the second order difference of the distortions (differential elbow)
    second_order_diff = np.diff(distortions, n=2)
    
    # The optimal k is where the second order difference is minimized
    elbow_point = np.argmin(second_order_diff) + 2

    return elbow_point

def smooth_curve(points, factor=0.6):
    """Smooths a curve using exponential moving average."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def determine_optimal_k_elbow_smoothed(X, k_range=(2, 11)):
    """
    Determine the optimal k value using the Elbow Method with a smoothed distortion curve.
    
    Args:
        X (DataFrame): The data to cluster.
        k_range (tuple, optional): A tuple indicating the range of k values to check.
                                   Defaults to (2, 15).

    Returns:
        int: Optimal value of k based on the smoothed Elbow Method.
    """
    distortions = []

    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(k=k)
        model.fit(X)
        z = model.predict(X)
        
        distortions.append(euclidean_distortion(X, z))
    
    # Smooth the distortions curve
    smoothed_distortions = smooth_curve(distortions)
    
    # Use Elbow Method on the smoothed curve
    elbow_point = np.argmin(np.diff(np.diff(smoothed_distortions))) + 2

    return elbow_point

def determine_optimal_k_combo(X, k_range=(2, 11)):
    """
    Determine the optimal k value using the original KMeans initialization and 
    a combination of Elbow Method and Silhouette Score.
    
    Args:
        X (DataFrame): The data to cluster.
        k_range (tuple, optional): A tuple indicating the range of k values to check.
                                   Defaults to (2, 15).

    Returns:
        int: Optimal value of k based on combined metrics.
    """
    distortions = []
    silhouette_scores = []
    
    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(k=k)
        model.fit(X)
        z = model.predict(X)
        
        distortions.append(euclidean_distortion(X, z))
        silhouette_scores.append(euclidean_silhouette(X, z))
    
    # Use Elbow Method
    elbow_point = np.argmin(np.diff(np.diff(distortions))) + 2
    
    # Use Silhouette Score
    max_silhouette_k = np.argmax(silhouette_scores) + 2
    return elbow_point
   