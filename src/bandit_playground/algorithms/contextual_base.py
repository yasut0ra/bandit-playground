from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ContextualBanditAlgo(ABC):
    """Abstract base for contextual bandit algorithms."""

    def __init__(self, n_arms: int, dim: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_arm(self, contexts: np.ndarray) -> int:
        ...

    @abstractmethod
    def update(self, contexts: np.ndarray, arm: int, reward: float) -> None:
        ...
