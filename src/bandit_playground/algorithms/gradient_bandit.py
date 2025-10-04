from __future__ import annotations
import numpy as np
from .base import BanditAlgo

class GradientBandit(BanditAlgo):
    """Preference-based gradient bandit with optional baseline."""

    def __init__(
        self,
        n_arms: int,
        alpha: float = 0.1,
        use_baseline: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms, seed)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.use_baseline = bool(use_baseline)
        self.preferences = np.zeros(n_arms, dtype=float)
        self._last_probs = np.full(n_arms, 1.0 / n_arms, dtype=float)
        self._last_arm: int | None = None
        self._avg_reward = 0.0
        self._t = 0

    def _softmax(self) -> np.ndarray:
        prefs = self.preferences - self.preferences.max()
        probs = np.exp(prefs)
        total = probs.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.full(self.n_arms, 1.0 / self.n_arms)
        return probs / total

    def select_arm(self) -> int:
        probs = self._softmax()
        arm = int(self.rng.choice(self.n_arms, p=probs))
        self._last_probs = probs
        self._last_arm = arm
        return arm

    def update(self, arm: int, reward: float) -> None:
        if self._last_arm is None:
            raise RuntimeError("select_arm must be called before update")
        if arm != self._last_arm:
            raise ValueError("update arm must match the last selected arm")
        self._t += 1
        if self.use_baseline:
            self._avg_reward += (reward - self._avg_reward) / self._t
            baseline = self._avg_reward
        else:
            baseline = 0.0
        diff = reward - baseline
        probs = self._last_probs
        grad = -self.alpha * diff * probs
        grad[arm] += self.alpha * diff
        self.preferences += grad
        self._last_arm = None
