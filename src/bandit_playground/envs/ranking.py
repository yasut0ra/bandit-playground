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

    def expected_reward(self, ranking: list[int]) -> float:
        return self.expected_click_prob(ranking)

    def step(self, ranking: list[int]) -> tuple[int | None, float]:
        return self.sample_click(ranking)

    @property
    def optimal_ranking(self) -> list[int]:
        return np.argsort(-self.probs)[: self.list_size].tolist()

    @property
    def optimal_click_prob(self) -> float:
        return self.expected_click_prob(self.optimal_ranking)

    @property
    def optimal_reward(self) -> float:
        return self.expected_reward(self.optimal_ranking)


class PositionBasedClickEnv:
    """Position-Based Model (PBM) environment with multiple possible clicks."""

    def __init__(
        self,
        attractiveness: np.ndarray,
        exam_prob: np.ndarray,
        seed: int | None = None,
    ) -> None:
        attractiveness = np.asarray(attractiveness, dtype=float)
        exam_prob = np.asarray(exam_prob, dtype=float)
        if attractiveness.ndim != 1:
            raise ValueError("attractiveness must be 1D")
        if exam_prob.ndim != 1:
            raise ValueError("exam_prob must be 1D")
        if np.any(attractiveness < 0.0) or np.any(attractiveness > 1.0):
            raise ValueError("attractiveness must be in [0, 1]")
        if np.any(exam_prob < 0.0) or np.any(exam_prob > 1.0):
            raise ValueError("exam_prob must be in [0, 1]")
        if attractiveness.size == 0:
            raise ValueError("attractiveness cannot be empty")
        if exam_prob.size == 0:
            raise ValueError("exam_prob cannot be empty")
        self.attractiveness = attractiveness
        self.exam_prob = exam_prob
        self.n_items = attractiveness.size
        self.list_size = exam_prob.size
        if self.list_size > self.n_items:
            raise ValueError("list_size cannot exceed number of items")
        if np.any(exam_prob <= 0.0):
            raise ValueError("exam_prob entries must be > 0 for PBM")
        self.rng = np.random.default_rng(seed)

    def _validate_ranking(self, ranking: list[int]) -> np.ndarray:
        ranking_arr = np.asarray(ranking, dtype=int)
        if ranking_arr.ndim != 1 or ranking_arr.size != self.list_size:
            raise ValueError("ranking must be length equal to list_size")
        if len(set(ranking_arr.tolist())) != ranking_arr.size:
            raise ValueError("ranking must contain unique items")
        if np.any(ranking_arr < 0) or np.any(ranking_arr >= self.n_items):
            raise ValueError("ranking contains invalid item indices")
        return ranking_arr

    def step(self, ranking: list[int]) -> tuple[np.ndarray, float]:
        ranking_arr = self._validate_ranking(ranking)
        clicks = np.zeros(self.list_size, dtype=float)
        for pos, item in enumerate(ranking_arr):
            prob = self.exam_prob[pos] * self.attractiveness[item]
            clicks[pos] = float(self.rng.random() < prob)
        reward = float(clicks.sum())
        return clicks, reward

    def expected_clicks(self, ranking: list[int]) -> np.ndarray:
        ranking_arr = self._validate_ranking(ranking)
        probs = self.exam_prob * self.attractiveness[ranking_arr]
        return probs

    def expected_reward(self, ranking: list[int]) -> float:
        return float(self.expected_clicks(ranking).sum())

    @property
    def optimal_ranking(self) -> list[int]:
        # order by attractiveness since exam probabilities are fixed by position
        order = np.argsort(-self.attractiveness)[: self.list_size]
        return order.tolist()

    @property
    def optimal_reward(self) -> float:
        return self.expected_reward(self.optimal_ranking)
