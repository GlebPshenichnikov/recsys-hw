import numpy as np
import scipy.sparse as sp


class ImplicitALS:
    def __init__(self, factors=64, regularization=1.0, alpha=10.0, iterations=10, cg_steps=3, seed=2021):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.seed = seed

    def fit(self, users, items, shape, preference=None):
        if preference is None:
            preference = np.ones(len(users))
        weight = sp.csr_matrix((np.full(len(users), self.alpha, dtype=np.float32), (users, items)), shape=shape)
        target = sp.csr_matrix(((1.0 + self.alpha) * np.asarray(preference, dtype=np.float32), (users, items)),
                               shape=shape)
        weight_t, target_t = weight.T.tocsr(), target.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0.0, 0.01, (shape[0], self.factors)).astype(np.float32)
        self.item_factors = rng.normal(0.0, 0.01, (shape[1], self.factors)).astype(np.float32)
        for _ in range(self.iterations):
            self.user_factors = self._conjugate_gradient(weight, target, self.item_factors, self.user_factors)
            self.item_factors = self._conjugate_gradient(weight_t, target_t, self.user_factors, self.item_factors)
        return self

    def _conjugate_gradient(self, weight, target, fixed, start):
        gram = fixed.T @ fixed + self.regularization * np.eye(self.factors, dtype=np.float32)
        owners = np.repeat(np.arange(weight.shape[0]), np.diff(weight.indptr))
        neighbours = fixed[weight.indices]

        def apply(v):
            local = weight.data * np.einsum("ij,ij->i", v[owners], neighbours)
            return v @ gram + sp.csr_matrix((local, weight.indices, weight.indptr), shape=weight.shape) @ fixed

        x = start.copy()
        residual = target @ fixed - apply(x)
        direction = residual.copy()
        norm = np.einsum("ij,ij->i", residual, residual)
        for _ in range(self.cg_steps):
            image = apply(direction)
            step = norm / np.maximum(np.einsum("ij,ij->i", direction, image), 1e-10)
            x += step[:, None] * direction
            residual -= step[:, None] * image
            new_norm = np.einsum("ij,ij->i", residual, residual)
            direction = residual + (new_norm / np.maximum(norm, 1e-10))[:, None] * direction
            norm = new_norm
        return x

    def predict(self, users):
        return self.user_factors[users] @ self.item_factors.T
