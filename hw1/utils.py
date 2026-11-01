from itertools import product

import numpy as np
import polars as pl
import scipy.sparse as sp


def interaction_matrix(frame, n_users, n_items):
    return sp.csr_matrix(
        (np.ones(frame.height, dtype=np.float32), (frame["user_idx"].to_numpy(), frame["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    )


class Fold:
    def __init__(self, data, start, end, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        self.start = start
        self.history = data.filter(pl.col("last_watch_dt") < start)
        self.window = data.filter((pl.col("last_watch_dt") >= start) & (pl.col("last_watch_dt") < end))
        self.matrix = interaction_matrix(self.history, n_users, n_items)
        self.candidates = np.zeros(n_items, dtype=bool)
        self.candidates[self.history["item_idx"].unique().to_numpy()] = True
        known = np.zeros(n_users, dtype=bool)
        known[self.history["user_idx"].unique().to_numpy()] = True
        target = self.window.filter(pl.col("relevant"))
        u, i = target["user_idx"].to_numpy(), target["item_idx"].to_numpy()
        keep = known[u] & self.candidates[i]
        self.users = np.unique(u[keep])
        self.truth = sp.csr_matrix(
            (np.ones(keep.sum(), dtype=np.float32), (np.searchsorted(self.users, u[keep]), i[keep])),
            shape=(len(self.users), n_items),
        )

    def arrays(self, *columns):
        return [self.history[c].to_numpy() for c in columns]

    def age_days(self):
        return self.history.select((pl.lit(self.start) - pl.col("last_watch_dt")).dt.total_days()).to_series().to_numpy()


def top_k(scores, seen, candidates, k=10):
    scores = np.array(scores, dtype=np.float32)
    scores[:, ~candidates] = -np.inf
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def evaluate(model, fold, k=10, batch=1024):
    ap, recall, hit = [], [], []
    ranks = np.arange(1, k + 1)
    for start in range(0, len(fold.users), batch):
        users = fold.users[start:start + batch]
        recs = top_k(model.predict(users), fold.matrix[users], fold.candidates, k)
        truth = fold.truth[start:start + batch].toarray() > 0
        hits = np.take_along_axis(truth, recs, axis=1)
        precision = np.cumsum(hits, axis=1) / ranks
        ap.append((precision * hits).sum(axis=1) / np.minimum(truth.sum(axis=1), k))
        recall.append(hits.sum(axis=1) / truth.sum(axis=1))
        hit.append(hits.any(axis=1))
    return {"MAP@10": float(np.concatenate(ap).mean()), "Recall@10": float(np.concatenate(recall).mean()),
            "HitRate@10": float(np.concatenate(hit).mean())}


def grid_search(fit, grid, fold):
    rows = []
    for values in product(*grid.values()):
        params = dict(zip(grid, values))
        rows.append({**params, **evaluate(fit(params, fold), fold)})
    return pl.DataFrame(rows).sort("MAP@10", descending=True)
