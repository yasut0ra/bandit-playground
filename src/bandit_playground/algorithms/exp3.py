from __future__ import annotations
import numpy as np
from .base import BanditAlgo

class Exp3(BanditAlgo):
    """EXP3 algorithm for adversarial bandits with bounded rewards."""

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.07,
        reward_min: float = 0.0,
        reward_max: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms, seed)
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        self._range = self.reward_max - self.reward_min
        if self._range <= 0.0:
            raise ValueError("reward_max must be greater than reward_min")
        self.weights = np.ones(n_arms, dtype=float)
        self._last_probs = np.full(n_arms, 1.0 / n_arms, dtype=float)
        self._last_arm: int | None = None

    def select_arm(self) -> int:
        weight_sum = float(self.weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            self.weights.fill(1.0)
            weight_sum = float(self.weights.sum())
        probs = (1.0 - self.gamma) * (self.weights / weight_sum)
        probs += self.gamma / self.n_arms
        probs /= probs.sum()
        arm = int(self.rng.choice(self.n_arms, p=probs))
        self._last_probs = probs
        self._last_arm = arm
        return arm

    def update(self, arm: int, reward: float) -> None:
        if self._last_arm is None:
            raise RuntimeError("select_arm must be called before update")
        if arm != self._last_arm:
            raise ValueError("update arm must match the last selected arm")
        payoff = (reward - self.reward_min) / self._range
        payoff = float(np.clip(payoff, 0.0, 1.0))
        prob = float(self._last_probs[arm])
        if prob <= 0.0:
            raise RuntimeError("selection probability is zero; cannot update")
        est_reward = payoff / prob
        growth = np.exp(self.gamma * est_reward / self.n_arms)
        self.weights[arm] *= growth
        self._last_arm = None
