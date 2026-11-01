import numpy as np
import scipy.sparse as sp


class UserKNN:
    def __init__(self, neighbours=50, shrink=0.0, batch=512):
        self.neighbours = neighbours
        self.shrink = shrink
        self.batch = batch

    def fit(self, matrix):
        self.matrix = matrix.tocsr().astype(np.float32)
        self.norms = np.sqrt(np.asarray(self.matrix.sum(axis=1)).ravel())
        return self

    def predict(self, users):
        blocks = []
        for start in range(0, len(users), self.batch):
            chunk = users[start:start + self.batch]
            overlap = (self.matrix @ self.matrix[chunk].T.toarray()).T
            sim = overlap / (np.outer(self.norms[chunk], self.norms) + self.shrink + 1e-9)
            sim[np.arange(len(chunk)), chunk] = 0.0
            top = np.argpartition(-sim, self.neighbours, axis=1)[:, :self.neighbours]
            graph = sp.csr_matrix(
                (np.take_along_axis(sim, top, axis=1).ravel(), top.ravel(),
                 np.arange(0, top.size + 1, self.neighbours)),
                shape=sim.shape,
            )
            blocks.append((graph @ self.matrix).toarray())
        return np.vstack(blocks)
