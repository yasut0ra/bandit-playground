from __future__ import annotations
import math
import numpy as np
from .ranking_base import RankingBanditAlgo


class PBMUCB1(RankingBanditAlgo):
    """UCB-style learner for PBM using importance-weighted item estimates."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        exam_prob: np.ndarray,
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        exam_prob = np.asarray(exam_prob, dtype=float)
        if exam_prob.shape != (list_size,):
            raise ValueError("exam_prob must have shape (list_size,)")
        if np.any(exam_prob <= 0.0) or np.any(exam_prob > 1.0):
            raise ValueError("exam_prob must lie in (0, 1]")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.exam_prob = exam_prob
        self.alpha = float(alpha)
        self.counts = np.zeros(n_items, dtype=float)
        self.estimates = np.zeros(n_items, dtype=float)
        self.rounds = 0

    def select_ranking(self) -> list[int]:
        self.rounds += 1
        ucb = np.zeros(self.n_items, dtype=float)
        for item in range(self.n_items):
            if self.counts[item] == 0:
                ucb[item] = math.inf
            else:
                bonus = math.sqrt((self.alpha * math.log(self.rounds)) / self.counts[item])
                ucb[item] = self.estimates[item] + bonus
        ranking = np.argsort(-ucb)[: self.list_size]
        return ranking.tolist()

    def update(self, ranking: list[int], feedback) -> None:
        clicks = np.asarray(feedback, dtype=float)
        if clicks.shape != (self.list_size,):
            raise ValueError("feedback must be array-like with length equal to list_size")
        for pos, item in enumerate(ranking):
            iw = clicks[pos] / self.exam_prob[pos]
            self.counts[item] += 1.0
            n = self.counts[item]
            self.estimates[item] += (iw - self.estimates[item]) / n


class PBMThompson(RankingBanditAlgo):
    """Gaussian Thompson Sampling for PBM with importance-weighted observations."""

    def __init__(
        self,
        n_items: int,
        list_size: int,
        exam_prob: np.ndarray,
        prior_mean: float = 0.5,
        prior_var: float = 1.0,
        noise_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_items, list_size, seed)
        exam_prob = np.asarray(exam_prob, dtype=float)
        if exam_prob.shape != (list_size,):
            raise ValueError("exam_prob must have shape (list_size,)")
        if np.any(exam_prob <= 0.0) or np.any(exam_prob > 1.0):
            raise ValueError("exam_prob must lie in (0, 1]")
        if prior_var <= 0.0 or noise_scale <= 0.0:
            raise ValueError("prior_var and noise_scale must be positive")
        self.exam_prob = exam_prob
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.noise_scale = float(noise_scale)
        self.counts = np.zeros(n_items, dtype=float)
        self.mean = np.full(n_items, self.prior_mean, dtype=float)
        self.m2 = np.zeros(n_items, dtype=float)

    def select_ranking(self) -> list[int]:
        samples = np.zeros(self.n_items, dtype=float)
        for item in range(self.n_items):
            if self.counts[item] == 0:
                samples[item] = self.rng.normal(self.prior_mean, math.sqrt(self.prior_var))
            else:
                var = self.m2[item] / self.counts[item] if self.counts[item] > 0 else self.prior_var
                var = max(var, 1e-6)
                scale = math.sqrt(var) / math.sqrt(self.counts[item])
                samples[item] = self.rng.normal(self.mean[item], self.noise_scale * scale)
        ranking = np.argsort(-samples)[: self.list_size]
        return ranking.tolist()

    def update(self, ranking: list[int], feedback) -> None:
        clicks = np.asarray(feedback, dtype=float)
        if clicks.shape != (self.list_size,):
            raise ValueError("feedback must be array-like with length equal to list_size")
        for pos, item in enumerate(ranking):
            iw = clicks[pos] / self.exam_prob[pos]
            self.counts[item] += 1.0
            delta = iw - self.mean[item]
            self.mean[item] += delta / self.counts[item]
            delta2 = iw - self.mean[item]
            self.m2[item] += delta * delta2
