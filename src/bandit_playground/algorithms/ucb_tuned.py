from __future__ import annotations
import math
import numpy as np
from .base import BanditAlgo


class UCBTuned(BanditAlgo):
    """UCB-Tuned algorithm (Auer et al., 2002)."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms, seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.square_sums = np.zeros(n_arms, dtype=float)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        log_t = math.log(self.t)
        indices = np.zeros(self.n_arms, dtype=float)
        for a in range(self.n_arms):
            count = self.counts[a]
            mean = self.values[a]
            var_est = self.square_sums[a] / count - mean ** 2
            var_est = max(var_est, 0.0)
            var_term = var_est + math.sqrt(2.0 * log_t / count)
            bonus = math.sqrt(max(log_t / count, 0.0) * min(0.25, var_term))
            indices[a] = mean + bonus
        return int(np.argmax(indices))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.square_sums[arm] += reward ** 2
