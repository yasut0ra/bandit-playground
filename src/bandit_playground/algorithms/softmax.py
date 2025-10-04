from __future__ import annotations
import numpy as np
from .base import BanditAlgo

class Softmax(BanditAlgo):
    """Boltzmann/Softmax exploration with fixed temperature."""

    def __init__(self, n_arms: int, tau: float = 0.1, seed: int | None = None) -> None:
        super().__init__(n_arms, seed)
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self) -> int:
        scaled = self.values / self.tau
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs_sum = probs.sum()
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            probs = np.full(self.n_arms, 1.0 / self.n_arms)
        else:
            probs /= probs_sum
        arm = int(self.rng.choice(self.n_arms, p=probs))
        return arm

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
