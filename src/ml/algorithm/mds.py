import numpy as np

class MDS:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, distances_matrix):
        n = distances_matrix.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * centering.dot(distances_matrix ** 2).dot(centering)

        eig_values, eig_vectors = np.linalg.eigh(B)
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]

        self.components_ = eig_vectors[:, :self.n_components] @ np.diag(np.sqrt(eig_values[:self.n_components]))

        return self