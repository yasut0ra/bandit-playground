from __future__ import annotations
import numpy as np
from .base import BanditAlgo

class EpsilonGreedy(BanditAlgo):
    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int | None = None) -> None:
        super().__init__(n_arms, seed)
        self.epsilon = float(epsilon)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean
        self.values[arm] += (reward - self.values[arm]) / n
