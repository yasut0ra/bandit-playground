from __future__ import annotations
import numpy as np


class CascadeClickEnv:
    """Cascade click model environment for ranking bandits."""

    def __init__(
        self,
        probs: np.ndarray,
        list_size: int,
        seed: int | None = None,
    ) -> None:
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 1:
            raise ValueError("probs must be 1D")
        if np.any(probs < 0.0) or np.any(probs > 1.0):
            raise ValueError("probs must be in [0, 1]")
        if list_size <= 0 or list_size > probs.size:
            raise ValueError("list_size must be between 1 and number of items")
        self.probs = probs
        self.n_items = probs.size
        self.list_size = int(list_size)
        self.rng = np.random.default_rng(seed)

    def _validate_ranking(self, ranking: list[int]) -> np.ndarray:
        ranking_arr = np.asarray(ranking, dtype=int)
        if ranking_arr.ndim != 1 or ranking_arr.size != self.list_size:
            raise ValueError("ranking must be a 1D sequence with length equal to list_size")
        if len(set(ranking_arr.tolist())) != ranking_arr.size:
            raise ValueError("ranking must contain unique items")
        if np.any(ranking_arr < 0) or np.any(ranking_arr >= self.n_items):
            raise ValueError("ranking contains invalid item indices")
        return ranking_arr

    def sample_click(self, ranking: list[int]) -> tuple[int | None, float]:
        ranking_arr = self._validate_ranking(ranking)
        for pos, item in enumerate(ranking_arr):
            if self.rng.random() < self.probs[item]:
                return pos, 1.0
        return None, 0.0

    def expected_click_prob(self, ranking: list[int]) -> float:
        ranking_arr = self._validate_ranking(ranking)
        probs = self.probs[ranking_arr]
        return float(1.0 - np.prod(1.0 - probs))

    def step(self, ranking: list[int]) -> tuple[int | None, float]:
        return self.sample_click(ranking)

    @property
    def optimal_ranking(self) -> list[int]:
        return np.argsort(-self.probs)[: self.list_size].tolist()

    @property
    def optimal_click_prob(self) -> float:
        return self.expected_click_prob(self.optimal_ranking)
