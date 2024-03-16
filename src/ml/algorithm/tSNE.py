import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class tSNE:
    def __init__(self, data, n_components=2, perplexity=30, n_iter=1000, random_state=0):
        self.data = data
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_components = n_components

    def fit_transform(self):
        tsne = TSNE(perplexity=self.perplexity, n_components=self.n_components, n_iter=self.n_iter, random_state=self.random_state)
        self.data = tsne.fit_transform(self.data)
        return self.data