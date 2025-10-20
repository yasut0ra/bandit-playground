from __future__ import annotations
import math
import numpy as np
from .ranking_base import RankingBanditAlgo


def _extract_observations(list_size: int, feedback) -> tuple[np.ndarray, np.ndarray]:
    """Return clicks and exposure indicators for the served list."""
    clicks = np.zeros(list_size, dtype=float)
    observed = np.ones(list_size, dtype=float)
    if isinstance(feedback, np.ndarray):
        clicks = feedback.astype(float)
        observed.fill(1.0)
    elif isinstance(feedback, (list, tuple)):
        arr = np.asarray(feedback, dtype=float)
        if arr.shape != (list_size,):
            raise ValueError("feedback length must match list_size")
        clicks = arr
        observed.fill(1.0)
    elif feedback is None:
        observed.fill(1.0)
    else:
        if isinstance(feedback, (int, np.integer)):
            idx = int(feedback)
            if idx < 0 or idx >= list_size:
                raise ValueError("click index out of range")
            observed[: idx + 1] = 1.0
            observed[idx + 1 :] = 0.0
            clicks[idx] = 1.0
        else:
            raise ValueError(f"Unsupported feedback type: {type(feedback).__name__}")
    return clicks, observed


class TopRank(RankingBanditAlgo):
    """TopRank (Lagrée et al., 2016) for general click models via pairwise confidence bounds."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        confidence: float = 1.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        if confidence <= 0.0:
            raise ValueError("confidence must be positive")
        self.confidence = float(confidence)
        self.wins = np.zeros((n_items, n_items), dtype=float)
        self.counts = np.zeros((n_items, n_items), dtype=float)
        self.t = 0

    def _confidence_radius(self, n: float) -> float:
        if n <= 0.0:
            return float("inf")
        log_term = math.log(max(self.t, 2.0))
        return math.sqrt((self.confidence * log_term) / n)

    def _build_levels(self) -> list[list[int]]:
        edges = np.zeros((self.n_items, self.n_items), dtype=bool)
        for i in range(self.n_items):
            for j in range(self.n_items):
                if i == j:
                    continue
                n_ij = self.counts[i, j]
                if n_ij <= 0.0:
                    continue
                mean = self.wins[i, j] / n_ij
                rad = self._confidence_radius(n_ij)
                if mean - rad > 0.5:
                    edges[i, j] = True
                elif mean + rad < 0.5:
                    edges[j, i] = True
        remaining = set(range(self.n_items))
        levels: list[list[int]] = []
        while remaining:
            level = [i for i in remaining if not any(edges[j, i] for j in remaining)]
            if not level:
                level = list(remaining)
            levels.append(level)
            remaining.difference_update(level)
        return levels

    def select_ranking(self) -> list[int]:
        levels = self._build_levels()
        ranking: list[int] = []
        for level in levels:
            pool = level[:]
            self.rng.shuffle(pool)
            for item in pool:
                ranking.append(item)
                if len(ranking) == self.list_size:
                    return ranking
        if len(ranking) < self.list_size:
            remaining = [i for i in range(self.n_items) if i not in ranking]
            self.rng.shuffle(remaining)
            ranking.extend(remaining[: self.list_size - len(ranking)])
        return ranking[: self.list_size]

    def update(self, ranking: list[int], feedback) -> None:
        self.t += 1
        clicks, observed = _extract_observations(len(ranking), feedback)
        for idx_i in range(len(ranking)):
            if observed[idx_i] <= 0.0:
                continue
            item_i = ranking[idx_i]
            click_i = clicks[idx_i]
            for idx_j in range(idx_i + 1, len(ranking)):
                if observed[idx_j] <= 0.0:
                    continue
                item_j = ranking[idx_j]
                click_j = clicks[idx_j]
                if click_i == click_j:
                    continue
                self.counts[item_i, item_j] += 1.0
                self.counts[item_j, item_i] += 1.0
                if click_i > click_j:
                    self.wins[item_i, item_j] += 1.0
                else:
                    self.wins[item_j, item_i] += 1.0


class UniRank(RankingBanditAlgo):
    """Simple UCB-style general ranking learner aggregating per-item click statistics."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.exposures = np.zeros(n_items, dtype=float)
        self.clicks = np.zeros(n_items, dtype=float)
        self.t = 0

    def _scores(self) -> np.ndarray:
        scores = np.zeros(self.n_items, dtype=float)
        total = max(self.t, 1)
        log_term = math.log(total + 1.0)
        for item in range(self.n_items):
            count = self.exposures[item]
            if count <= 0.0:
                scores[item] = float("inf")
                continue
            mean = self.clicks[item] / count
            bonus = math.sqrt((self.alpha * log_term) / count)
            scores[item] = mean + bonus
        return scores

    def select_ranking(self) -> list[int]:
        scores = self._scores()
        noise = self.rng.normal(0.0, 1e-9, size=self.n_items)
        ranking = np.argsort(-(scores + noise))[: self.list_size]
        return ranking.tolist()

    def update(self, ranking: list[int], feedback) -> None:
        self.t += 1
        clicks, observed = _extract_observations(len(ranking), feedback)
        for pos, item in enumerate(ranking):
            if observed[pos] <= 0.0:
                continue
            self.exposures[item] += 1.0
            self.clicks[item] += clicks[pos]
