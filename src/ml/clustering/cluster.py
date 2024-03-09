import numpy as np
from typing import Union, List

class Cluster:
    def __init__(self, label_id: int, indexes: List[int], features: np.ndarray):
        self.label_id = label_id
        self.indexes = indexes
        self.features = features

    def __iter__(self):
        yield from self.indexes

    def __len__(self):
        return len(self.indexes)

def cluster_match_score(cluster_actual: Cluster, cluster_predicted: Cluster):
    intersect_index_count = 0
    for index in cluster_actual.indexes:
        if index in cluster_predicted.indexes:
            intersect_index_count += 1
    total_index_count = len(cluster_actual) + \
        len(cluster_predicted) - intersect_index_count
    return intersect_index_count / total_index_count

__all__ = ["Cluster", "cluster_match_score"]