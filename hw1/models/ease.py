import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, l2=200.0):
        self.l2 = l2

    def fit(self, matrix):
        self.matrix = matrix.tocsr().astype(np.float32)
        gram = (self.matrix.T @ self.matrix).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.l2
        inverse = cho_solve(cho_factor(gram), np.eye(gram.shape[0]))
        weights = inverse / -np.diag(inverse)
        np.fill_diagonal(weights, 0.0)
        self.weights = weights.astype(np.float32)
        return self

    def predict(self, users):
        return self.matrix[users] @ self.weights
