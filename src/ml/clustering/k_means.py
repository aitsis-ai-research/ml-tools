import numpy as np
from typing import Union, List
from .cluster import Cluster

class KMeans:
    def __init__(self, X: Union[np.ndarray, list], centroids: Union[np.ndarray, list] = None, n_clusters=8, max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

        if isinstance(X, list):
            self.X = np.array(X)
        else:
            self.X = X

        if centroids is not None:
            if isinstance(centroids, list):
                self.centroids = np.array(centroids)
            else:
                self.centroids = centroids
            if self.centroids.shape[0] != self.n_clusters:
                raise ValueError(
                    "The number of centroids should be equal to the number of clusters")
            if self.centroids.shape[1] != self.X.shape[1]:
                raise ValueError(
                    "The number of features of the centroids should be equal to the number of features of the data")
        else:
            self._init_centroids()

    def _init_centroids(self):
        indices = np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
        self.centroids = self.X[indices]

    def fit(self, distance_method):
        iter = 0
        
        while(self.max_iters > iter):
            clusters = [[] for _ in range(self.n_clusters)]
            for i in range(self.X.shape[0]):
                distances = [distance_method(self.X[i], c) for c in self.centroids]                
                cluster = np.argmin(distances)
                clusters[cluster].append(self.X[i])
            # if any cluster is empty, reassign the centroid
            print("one cluster get empty. so reassigning the centroid.")
            if any([len(cluster) == 0 for cluster in clusters]):
                self._init_centroids()
                iter = 0
                continue
                
            new_centroids = np.array(
                [np.mean(cluster, axis=0) for cluster in clusters])
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
            iter += 1
        return self.centroids
    
    def get_clusters(self):
        Cluster_list = []
        clusters = [[] for _ in range(self.n_clusters)]
        for i in range(self.X.shape[0]):
            distances = [np.linalg.norm(self.X[i] - c) for c in self.centroids]
            cluster = np.argmin(distances)
            clusters[cluster].append(i)
        
        for i in range(self.n_clusters):
            Cluster_list.append(Cluster(i, clusters[i], self.X[clusters[i]]))
            
        return Cluster_list
        
        

__all__ = ["KMeans"]
