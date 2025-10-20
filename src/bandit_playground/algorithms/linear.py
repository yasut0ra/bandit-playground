from __future__ import annotations
import math
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


class OFUL(ContextualBanditAlgo):
    """Optimism in the Face of Uncertainty for linear bandits (global parameter)."""

    def __init__(
        self,
        n_arms: int,
        dim: int,
        lambda_: float = 1.0,
        delta: float = 0.1,
        context_bound: float = 1.0,
        theta_bound: float = 1.0,
        noise_scale: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms, dim, seed)
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if context_bound <= 0.0:
            raise ValueError("context_bound must be positive")
        if theta_bound <= 0.0:
            raise ValueError("theta_bound must be positive")
        if noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive")
        self.lambda_ = float(lambda_)
        self.delta = float(delta)
        self.context_bound = float(context_bound)
        self.theta_bound = float(theta_bound)
        self.noise_scale = float(noise_scale)
        self.V = self.lambda_ * np.eye(dim, dtype=float)
        self.b = np.zeros(dim, dtype=float)
        self.t = 0

    def _beta(self) -> float:
        t = max(self.t, 1)
        log_arg = 1.0 + (t * (self.context_bound ** 2)) / self.lambda_
        confidence = self.dim * math.log(log_arg) + 2.0 * math.log(1.0 / self.delta)
        confidence = max(confidence, 0.0)
        return self.noise_scale * math.sqrt(confidence) + math.sqrt(self.lambda_) * self.theta_bound

    def select_arm(self, contexts: np.ndarray) -> int:
        theta_hat = np.linalg.solve(self.V, self.b)
        beta = self._beta()
        scores = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            x = contexts[arm]
            v_inv_x = np.linalg.solve(self.V, x)
            bonus = beta * math.sqrt(max(float(x @ v_inv_x), 0.0))
            scores[arm] = float(theta_hat @ x) + bonus
        return int(np.argmax(scores))

    def update(self, contexts: np.ndarray, arm: int, reward: float) -> None:
        x = contexts[arm]
        self.V += np.outer(x, x)
        self.b += reward * x
        self.t += 1
