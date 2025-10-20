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

    def update(self, ranking: list[int], feedback) -> None:
        click_index = feedback
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

    def update(self, ranking: list[int], feedback) -> None:
        click_index = feedback
        for pos, item in enumerate(ranking):
            if click_index is not None and pos == click_index:
                self.alpha[item] += 1.0
                break
            else:
                self.beta[item] += 1.0
            if click_index is not None and pos > click_index:
                break


class CascadeKLUCB(RankingBanditAlgo):
    """Cascade-style KL-UCB (complements CascadeUCB1 with tighter confidence sets)."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        c: float = 3.0,
        tol: float = 1e-6,
        max_iter: int = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        if c <= 0.0:
            raise ValueError("c must be positive")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.c = float(c)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.counts = np.zeros(n_items, dtype=int)
        self.successes = np.zeros(n_items, dtype=float)
        self.rounds = 0

    @staticmethod
    def _kl(p: float, q: float) -> float:
        eps = 1e-12
        p = min(max(p, eps), 1.0 - eps)
        q = min(max(q, eps), 1.0 - eps)
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def _upper_conf(self, mean: float, count: int) -> float:
        if count == 0:
            return 1.0
        total = max(self.rounds, 1)
        log_term = math.log(total) + self.c * math.log(max(math.log(total), 1.0))
        bound = max(log_term / count, 0.0)
        low, high = mean, 1.0
        for _ in range(self.max_iter):
            mid = (low + high) / 2.0
            if self._kl(mean, mid) > bound:
                high = mid
            else:
                low = mid
            if high - low < self.tol:
                break
        return low

    def select_ranking(self) -> list[int]:
        self.rounds += 1
        indices = np.zeros(self.n_items, dtype=float)
        for item in range(self.n_items):
            count = self.counts[item]
            if count == 0:
                indices[item] = math.inf
            else:
                mean = self.successes[item] / count
                indices[item] = self._upper_conf(mean, int(count))
        ranking = np.argsort(-indices)[: self.list_size]
        return ranking.tolist()

    def update(self, ranking: list[int], feedback) -> None:
        click_index = feedback
        for pos, item in enumerate(ranking):
            self.counts[item] += 1
            if click_index is not None and pos == click_index:
                self.successes[item] += 1.0
                break
            if click_index is not None and pos > click_index:
                break
