from __future__ import annotations
import numpy as np
from .base import BanditAlgo

class UCB1(BanditAlgo):
    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms, seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        # pull each arm once initially
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        ucb = self.values + np.sqrt(2.0 * np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
