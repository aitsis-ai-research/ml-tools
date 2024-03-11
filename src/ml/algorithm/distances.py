import numpy as np
def euclidean_distance(x:np.ndarray, y:np.ndarray):
    return np.linalg.norm(x - y)

def cosine_distance(x:np.ndarray, y:np.ndarray):
    dot_product = np.dot(x, y.T)
    
    norm_mult = np.linalg.norm(x) * np.linalg.norm(y)
    if norm_mult == 0:
        return 0
    cosine_similarity = dot_product / norm_mult
    cosine_similarity = cosine_similarity.item()
    if cosine_similarity > 1:
        cosine_similarity = 1
    if cosine_similarity < -1:
        cosine_similarity = -1
    return 1 - cosine_similarity



__all__ = ["euclidean_distance", "cosine_distance"]

if __name__ == "__main__":
    # Test cosine_similarity
    x = np.array([1, 1, 1])
    y = np.array([-1, -1, -1])
    z = np.array([0, 0, 0])
    
    print( "Cosine similarity distance for x[1,1,1] and y[-1,-1,-1] : ", cosine_distance(x, y))
    print( "Cosine similarity distance for x[1,1,1] and z[0,0,0] : ", cosine_distance(x, z))
    print( "Cosine similarity distance for x[1,1,1] and x[1,1,1] : ", cosine_distance(x, x))
