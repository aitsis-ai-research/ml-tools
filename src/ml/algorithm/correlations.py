import numpy as np
from .distances import euclidean_distance, cosine_distance

def inter_class_correlation(np_class1: np.ndarray, np_class2: np.ndarray, method:str = "euclidean"):
    
    if np_class1.shape[1] != np_class2.shape[1]:
        raise ValueError("Class dimensions must be equal")
    distance = 0
    
    if method == "euclidean":
        for i in range(np_class1.shape[0]):
            for j in range(np_class2.shape[0]):
                distance +=  euclidean_distance(np_class1[i], np_class2[j])
               
    elif method == "cosine":
        for i in range(np_class1.shape[0]):
            for j in range(np_class2.shape[0]):
                distance += cosine_distance(np_class1[i], np_class2[j])
    else:
        raise ValueError("Invalid method. Method must be either 'euclidean' or 'cosine'")
    
    distance = distance / (np_class1.shape[0] * np_class2.shape[0])
    return distance

def intra_class_correlation(np_class:np.ndarray, method:str = "euclidean"):
   
    distance = 0
    
    if method == "euclidean":
        for i in range(np_class.shape[0]):
            for j in range(np_class.shape[0]):
                if i == j:
                    continue
                distance += euclidean_distance(np_class[i] - np_class[j])
    elif method == "cosine":
        for i in range(np_class.shape[0]):
            for j in range(np_class.shape[0]):
                if i == j:
                    continue
                distance += cosine_distance(np_class[i], np_class[j])
    else:
        raise ValueError("Invalid method. Method must be either 'euclidean' or 'cosine'")
    distance = distance / (np_class.shape[0] * (np_class.shape[0] - 1))
    return distance



__all__ = ["inter_class_distance", "intra_class_distance"]


if __name__ == "__main__":
    # Test inter_class_distance
    expected_output = 8.4852
    class1 = np.array([[1, 2], [3, 4], [5, 6]])
    class2 = np.array([[7, 8], [9, 10], [11, 12]])
    
    print("Inter class distance ", inter_class_correlation(class1, class2, method="euclidean"))
    print("Expected output: ", expected_output)
    
    # Test intra_class_distance
    expected_output = 3.77
    class3 = np.array([[1, 2], [3, 4], [5, 6]])
    print("Intra class distance ", intra_class_correlation(class3, method="euclidean"))
    print("Expected output: ", expected_output)
    
