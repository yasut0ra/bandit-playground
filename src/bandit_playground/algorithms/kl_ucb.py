from __future__ import annotations
import math
import numpy as np
from .base import BanditAlgo

class KLUCB(BanditAlgo):
    """KL-UCB for Bernoulli rewards (Cappé et al., 2013)."""

    def __init__(
        self,
        n_arms: int,
        c: float = 3.0,
        tol: float = 1e-6,
        max_iter: int = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms, seed)
        if c <= 0.0:
            raise ValueError("c must be positive")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.c = float(c)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.t = 0

    @staticmethod
    def _kl(p: float, q: float) -> float:
        eps = 1e-12
        p = min(max(p, eps), 1.0 - eps)
        q = min(max(q, eps), 1.0 - eps)
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def _upper_conf(self, mean: float, count: int) -> float:
        if count == 0:
            return 1.0
        total_count = max(self.t, 1)
        log_term = math.log(total_count) + self.c * math.log(max(math.log(total_count), 1.0))
        bound = log_term / count
        low = mean
        high = 1.0
        for _ in range(self.max_iter):
            mid = (low + high) / 2.0
            if self._kl(mean, mid) > bound:
                high = mid
            else:
                low = mid
            if high - low < self.tol:
                break
        return low

    def select_arm(self) -> int:
        self.t += 1
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        indices = np.zeros(self.n_arms, dtype=float)
        for a in range(self.n_arms):
            indices[a] = self._upper_conf(self.values[a], int(self.counts[a]))
        return int(np.argmax(indices))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
