from __future__ import annotations
import numpy as np


class LinearBanditEnv:
    """Contextual linear bandit with per-arm parameters."""

    def __init__(
        self,
        weights: np.ndarray,
        context_cov: np.ndarray | None = None,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 2:
            raise ValueError("weights must be 2D (n_arms, dim)")
        self.weights = weights
        self.n_arms, self.dim = weights.shape
        if context_cov is None:
            self.context_cov = np.eye(self.dim, dtype=float)
        else:
            cov = np.asarray(context_cov, dtype=float)
            if cov.shape != (self.dim, self.dim):
                raise ValueError("context_cov must be (dim, dim)")
            self.context_cov = cov
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self._current_contexts: np.ndarray | None = None

    def sample_contexts(self) -> np.ndarray:
        contexts = self.rng.multivariate_normal(
            np.zeros(self.dim), self.context_cov, size=self.n_arms
        )
        self._current_contexts = contexts
        return contexts

    def expected_rewards(self) -> np.ndarray:
        if self._current_contexts is None:
            raise RuntimeError("sample_contexts must be called before expected_rewards")
        return np.einsum("ad,ad->a", self._current_contexts, self.weights)

    def expected_reward(self, arm: int) -> float:
        return float(self.expected_rewards()[arm])

    def pull(self, arm: int) -> float:
        mean = self.expected_reward(arm)
        reward = mean
        if self.noise_std > 0.0:
            reward += self.rng.normal(0.0, self.noise_std)
        return float(reward)

    @property
    def optimal_arm(self) -> int:
        rewards = self.expected_rewards()
        return int(np.argmax(rewards))

    @property
    def optimal_reward(self) -> float:
        return float(self.expected_rewards().max())
