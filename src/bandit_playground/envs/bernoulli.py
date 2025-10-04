from __future__ import annotations
import numpy as np

class BernoulliKArms:
    """K-armed bandit with Bernoulli rewards."""
    def __init__(self, probs: np.ndarray, seed: int | None = None) -> None:
        self.probs = np.asarray(probs, dtype=float)
        self.n_arms = self.probs.shape[0]
        self.rng = np.random.default_rng(seed)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.probs[arm])

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def optimal_p(self) -> float:
        return float(np.max(self.probs))
