from __future__ import annotations
import numpy as np
from .base import BanditAlgo

class ThompsonBernoulli(BanditAlgo):
    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms, seed)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if reward >= 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
