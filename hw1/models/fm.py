import numpy as np
import torch
import torch.nn.functional as F


def to_tensor(matrix):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(coo.data.astype(np.float32)), coo.shape,
                                   check_invariants=False)


class FactorizationMachine(torch.nn.Module):
    def __init__(self, user_x, item_x, dim=32, lr=0.01, l2=1e-4, epochs=5, negatives=4, batch=8192, seed=2021):
        super().__init__()
        torch.manual_seed(seed)
        self.user_x, self.item_x = user_x.tocsr(), item_x.tocsr()
        self.lr, self.l2, self.epochs = lr, l2, epochs
        self.negatives, self.batch, self.seed = negatives, batch, seed
        self.user_v = torch.nn.Parameter(0.01 * torch.randn(user_x.shape[1], dim))
        self.item_v = torch.nn.Parameter(0.01 * torch.randn(item_x.shape[1], dim))
        self.user_w = torch.nn.Parameter(torch.zeros(user_x.shape[1], 1))
        self.item_w = torch.nn.Parameter(torch.zeros(item_x.shape[1], 1))
        self.w0 = torch.nn.Parameter(torch.zeros(1))

    def _side(self, x, v, w):
        total = torch.sparse.mm(to_tensor(x), v)
        squares = torch.sparse.mm(to_tensor(x.multiply(x)), v.pow(2))
        bias = torch.sparse.mm(to_tensor(x), w).squeeze(1) + 0.5 * (total.pow(2) - squares).sum(1)
        return total, bias

    def forward(self, users, items):
        user_vec, user_bias = self._side(self.user_x[users], self.user_v, self.user_w)
        item_vec, item_bias = self._side(self.item_x[items], self.item_v, self.item_w)
        return self.w0 + user_bias + item_bias + (user_vec * item_vec).sum(1)

    def fit(self, users, items, candidates):
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        pool = np.flatnonzero(candidates)
        for _ in range(self.epochs):
            order = rng.permutation(len(users))
            for start in range(0, len(order), self.batch):
                idx = order[start:start + self.batch]
                u = np.tile(users[idx], self.negatives + 1)
                i = np.concatenate([items[idx], rng.choice(pool, len(idx) * self.negatives)])
                y = torch.cat([torch.ones(len(idx)), torch.zeros(len(idx) * self.negatives)])
                loss = F.binary_cross_entropy_with_logits(self(u, i), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            self.user_emb = self._side(self.user_x, self.user_v, self.user_w)[0].numpy()
            item_emb, item_bias = self._side(self.item_x, self.item_v, self.item_w)
            self.item_emb, self.item_bias = item_emb.numpy(), item_bias.numpy()
        return self

    def predict(self, users):
        return self.user_emb[users] @ self.item_emb.T + self.item_bias
