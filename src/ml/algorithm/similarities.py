import numpy as np

def cosine_similarity(x:np.ndarray, y:np.ndarray):
    x = x.flatten()
    y = y.flatten()
    dot_product = np.dot(x, y)
    
    norm_mult = np.linalg.norm(x) * np.linalg.norm(y)
    if norm_mult == 0:
        return float('inf')  # Return positive infinity for cosine distance when one of the vectors is a zero vector
    cosine_similarity = dot_product / norm_mult
    cosine_similarity = cosine_similarity.item()
    if cosine_similarity > 1:
        cosine_similarity = 1
    if cosine_similarity < -1:
        cosine_similarity = -1
    return cosine_similarity
   
__all__ = ["cosine_similarity"]

if __name__ == "__main__":
    # Test cosine_similarity
    x = np.array([1, 1, 1])
    y = np.array([-1, -1, -1])
    z = np.array([0, 0, 0])
    
    print( "Cosine similarity for x[1,1,1] and y[-1,-1,-1] : ", cosine_similarity(x, y))
    print( "Cosine similarity for x[1,1,1] and z[0,0,0] : ", cosine_similarity(x, z))
    print( "Cosine similarity for x[1,1,1] and x[1,1,1] : ", cosine_similarity(x, x))
    

