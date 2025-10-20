from __future__ import annotations
import math
import numpy as np
from .base import BanditAlgo


class MOSS(BanditAlgo):
    """Minimax Optimal Strategy for Stochastic bandits (Audibert & Bubeck, 2009)."""

    def __init__(self, n_arms: int, horizon: int, seed: int | None = None) -> None:
        super().__init__(n_arms, seed)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.t = 0

    def _bonus(self, count: int) -> float:
        log_arg = max(self.horizon / (self.n_arms * max(count, 1)), 1.0)
        log_term = math.log(log_arg)
        return math.sqrt(max(log_term, 0.0) / max(count, 1))

    def select_arm(self) -> int:
        self.t += 1
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        indices = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            bonus = self._bonus(int(self.counts[arm]))
            indices[arm] = self.values[arm] + bonus
        return int(np.argmax(indices))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
