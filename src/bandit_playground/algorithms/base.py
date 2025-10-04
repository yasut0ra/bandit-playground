from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BanditAlgo(ABC):
    """Abstract base for bandit algorithms."""
    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_arm(self) -> int:
        ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        ...
