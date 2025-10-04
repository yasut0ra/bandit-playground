from __future__ import annotations
import math
import numpy as np
from .ranking_base import RankingBanditAlgo


class CascadeUCB1(RankingBanditAlgo):
    """Cascade-style UCB1 (Kveton et al., 2015) for click feedback."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        alpha: float = 1.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.counts = np.zeros(n_items, dtype=int)
        self.successes = np.zeros(n_items, dtype=float)
        self.rounds = 0

    def select_ranking(self) -> list[int]:
        self.rounds += 1
        total = max(self.rounds, 1)
        ucb = np.zeros(self.n_items, dtype=float)
        for item in range(self.n_items):
            c = self.counts[item]
            if c == 0:
                ucb[item] = math.inf
            else:
                mean = self.successes[item] / c
                bonus = math.sqrt((self.alpha * math.log(total)) / c)
                ucb[item] = mean + bonus
        ranking = np.argsort(-ucb)[: self.list_size]
        return ranking.tolist()

    def update(self, ranking: list[int], click_index: int | None) -> None:
        for pos, item in enumerate(ranking):
            self.counts[item] += 1
            if click_index is not None and pos == click_index:
                self.successes[item] += 1.0
                break
            if click_index is not None and pos > click_index:
                break


class CascadeThompson(RankingBanditAlgo):
    """Thompson Sampling under the cascade click model."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("priors must be positive")
        self.alpha = np.full(n_items, alpha_prior, dtype=float)
        self.beta = np.full(n_items, beta_prior, dtype=float)

    def select_ranking(self) -> list[int]:
        samples = self.rng.beta(self.alpha, self.beta)
        ranking = np.argsort(-samples)[: self.list_size]
        return ranking.tolist()

    def update(self, ranking: list[int], click_index: int | None) -> None:
        for pos, item in enumerate(ranking):
            if click_index is not None and pos == click_index:
                self.alpha[item] += 1.0
                break
            else:
                self.beta[item] += 1.0
            if click_index is not None and pos > click_index:
                break
