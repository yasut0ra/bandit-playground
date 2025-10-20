import numpy as np
from bandit_playground.envs.ranking import CascadeClickEnv, PositionBasedClickEnv
from bandit_playground.algorithms.ranking_cascade import (
    CascadeUCB1,
    CascadeThompson,
    CascadeEpsilonGreedy,
    CascadeKLUCB,
)
from bandit_playground.algorithms.ranking_pbm import PBMUCB1, PBMThompson, PBMSoftmax
from bandit_playground.algorithms.ranking_general import TopRank, UniRank
from bandit_playground.experiment import run_ranking_bandit


def test_cascade_algorithms_basic():
    probs = np.array([0.55, 0.5, 0.3, 0.2, 0.1])
    env = CascadeClickEnv(probs, list_size=3, seed=0)
    steps = 4000
    algos = (
        (CascadeUCB1, {"alpha": 1.5, "seed": 0}, 900.0),
        (CascadeThompson, {"alpha_prior": 1.0, "beta_prior": 1.0, "seed": 0}, 1100.0),
        (CascadeKLUCB, {"c": 2.0, "seed": 0}, 950.0),
        (CascadeEpsilonGreedy, {"epsilon": 0.1, "seed": 0}, 1350.0),
        (TopRank, {"confidence": 1.2, "seed": 0}, 1100.0),
        (UniRank, {"alpha": 1.0, "seed": 0}, 1250.0),
    )
    for Algo, kwargs, limit in algos:
        algo = Algo(env.n_items, env.list_size, **kwargs)
        result = run_ranking_bandit(env, algo, steps=steps, seed=1)
        assert result.rewards.shape == (steps,)
        assert result.pseudo_regret.shape == (steps,)
        assert result.instant_regret.shape == (steps,)
        assert result.pseudo_regret.sum() < limit


def test_pbm_algorithms_basic():
    attractiveness = np.array([0.6, 0.55, 0.4, 0.3, 0.2, 0.1])
    exam_prob = np.array([1.0, 0.7, 0.5])
    env = PositionBasedClickEnv(attractiveness, exam_prob=exam_prob, seed=3)
    steps = 6000
    algos = (
        (PBMUCB1, {"exam_prob": exam_prob, "alpha": 1.0, "seed": 0}, 900.0),
        (PBMThompson, {"exam_prob": exam_prob, "prior_mean": 0.5, "seed": 1}, 1100.0),
        (PBMSoftmax, {"exam_prob": exam_prob, "tau": 0.25, "seed": 0}, 1400.0),
        (TopRank, {"confidence": 1.5, "seed": 0}, 1600.0),
        (UniRank, {"alpha": 1.2, "seed": 0}, 1500.0),
    )
    for Algo, kwargs, limit in algos:
        algo = Algo(env.n_items, env.list_size, **kwargs)
        result = run_ranking_bandit(env, algo, steps=steps, seed=2)
        assert result.rewards.shape == (steps,)
        assert result.pseudo_regret.shape == (steps,)
        assert result.instant_regret.shape == (steps,)
        assert result.pseudo_regret.sum() < limit
