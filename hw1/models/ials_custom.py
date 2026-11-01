import numpy as np

from .ials import ImplicitALS


class EngagementALS(ImplicitALS):
    def __init__(self, floor=0.5, tau=60.0, full_pct=50.0, full_seconds=1200.0, **kwargs):
        super().__init__(**kwargs)
        self.floor = floor
        self.tau = tau
        self.full_pct = full_pct
        self.full_seconds = full_seconds

    def preference(self, watched_pct, total_dur, age_days):
        depth = np.minimum(1.0, np.maximum(watched_pct / self.full_pct, total_dur / self.full_seconds))
        return (self.floor + (1.0 - self.floor) * depth) * np.exp(-age_days / self.tau)

    def fit(self, users, items, shape, watched_pct, total_dur, age_days):
        return super().fit(users, items, shape, self.preference(watched_pct, total_dur, age_days))
