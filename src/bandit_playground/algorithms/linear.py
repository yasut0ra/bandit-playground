from __future__ import annotations
import numpy as np
from .contextual_base import ContextualBanditAlgo


class LinUCB(ContextualBanditAlgo):
    """Disjoint LinUCB (per-arm models with fixed alpha)."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, seed: int | None = None) -> None:
        super().__init__(n_arms, dim, seed)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.A = np.array([np.eye(dim, dtype=float) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim), dtype=float)

    def select_arm(self, contexts: np.ndarray) -> int:
        scores = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            x = contexts[arm]
            A = self.A[arm]
            theta_hat = np.linalg.solve(A, self.b[arm])
            invAx = np.linalg.solve(A, x)
            bonus = self.alpha * np.sqrt(float(x @ invAx))
            scores[arm] = float(theta_hat @ x) + bonus
        return int(np.argmax(scores))

    def update(self, contexts: np.ndarray, arm: int, reward: float) -> None:
        x = contexts[arm]
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


class LinThompson(ContextualBanditAlgo):
    """Thompson Sampling with Gaussian posterior for disjoint linear bandits."""

    def __init__(
        self,
        n_arms: int,
        dim: int,
        v: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms, dim, seed)
        if v <= 0.0:
            raise ValueError("v must be positive")
        self.v = float(v)
        self.A = np.array([np.eye(dim, dtype=float) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim), dtype=float)

    def select_arm(self, contexts: np.ndarray) -> int:
        scores = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            A = self.A[arm]
            mu = np.linalg.solve(A, self.b[arm])
            cov = self.v ** 2 * np.linalg.inv(A)
            theta_sample = self.rng.multivariate_normal(mu, cov)
            scores[arm] = float(theta_sample @ contexts[arm])
        return int(np.argmax(scores))

    def update(self, contexts: np.ndarray, arm: int, reward: float) -> None:
        x = contexts[arm]
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
