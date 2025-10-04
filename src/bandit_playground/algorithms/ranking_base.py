from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class RankingBanditAlgo(ABC):
    """Base class for ranking bandit algorithms that output ordered lists of items."""

    def __init__(self, n_items: int, list_size: int, seed: int | None = None) -> None:
        if list_size <= 0 or list_size > n_items:
            raise ValueError("list_size must be between 1 and n_items")
        self.n_items = int(n_items)
        self.list_size = int(list_size)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_ranking(self) -> list[int]:
        """Return an ordered list of item indices to display."""

    @abstractmethod
    def update(self, ranking: list[int], feedback) -> None:
        """Update internal state given the served ranking and observed feedback."""
