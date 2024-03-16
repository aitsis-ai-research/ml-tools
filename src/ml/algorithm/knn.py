import numpy as np
from collections import Counter
from typing import Iterable, Union, Callable

class KNN:
    def __init__(self, data: Iterable, label: Iterable[int]):
        self.data = data
        self.label = label
        self.classes = list(set(label))
        
    def check_valid_query(self, query: Union[list, np.array, tuple]) -> np.array:
        if not isinstance(query, (list, np.array, tuple)):
            raise ValueError("Query must be a list, np.array, or tuple")
        if isinstance(query, list) and len(query) != len(self.data[0]):
            raise ValueError("Query must have the same length as the data")
        if isinstance(query, np.ndarray) and query.shape[0] != len(self.data[0]):
            raise ValueError("Query must have the same length as the data")
        if isinstance(query, tuple) and len(query) != len(self.data[0]):
            raise ValueError("Query must have the same length as the data")
        return np.array(query)
        
    def predict(self, k: int, query: Union[list, np.array, tuple], method: Callable[..., float]) -> int:
        if k > len(self.classes):
            raise ValueError("k cannot be greater than the number of classes")
        
        query = self.check_valid_query(query)
        
        distance_key_pairs = []
        for i in range(len(self.data)):
            distance = method(self.data[i], query)
            distance_key_pairs.append({"distance": distance, "key": self.label[i]})
            
        distance_key_pairs = sorted(distance_key_pairs, key=lambda x: x["distance"])
        k_nearest = distance_key_pairs[:k]
        k_nearest_labels = [x["key"] for x in k_nearest]
        
        counter = Counter(k_nearest_labels)
        most_common = counter.most_common(1)
        
        return most_common[0][0]
