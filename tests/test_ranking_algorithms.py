import numpy as np
from bandit_playground.envs.ranking import CascadeClickEnv, PositionBasedClickEnv
from bandit_playground.algorithms.ranking_cascade import CascadeUCB1, CascadeThompson
from bandit_playground.algorithms.ranking_pbm import PBMUCB1, PBMThompson
from bandit_playground.experiment import run_ranking_bandit


def test_cascade_algorithms_basic():
    probs = np.array([0.55, 0.5, 0.3, 0.2, 0.1])
    env = CascadeClickEnv(probs, list_size=3, seed=0)
    steps = 4000
    algos = (
        (CascadeUCB1, {"alpha": 1.5, "seed": 0}, 900.0),
        (CascadeThompson, {"alpha_prior": 1.0, "beta_prior": 1.0, "seed": 0}, 1100.0),
    )
    for Algo, kwargs, limit in algos:
        algo = Algo(env.n_items, env.list_size, **kwargs)
        rewards, regrets, _ = run_ranking_bandit(env, algo, steps=steps, seed=1)
        assert rewards.shape == (steps,)
        assert regrets.shape == (steps,)
        assert regrets.sum() < limit


def test_pbm_algorithms_basic():
    attractiveness = np.array([0.6, 0.55, 0.4, 0.3, 0.2, 0.1])
    exam_prob = np.array([1.0, 0.7, 0.5])
    env = PositionBasedClickEnv(attractiveness, exam_prob=exam_prob, seed=3)
    steps = 6000
    algos = (
        (PBMUCB1, {"exam_prob": exam_prob, "alpha": 1.0, "seed": 0}, 900.0),
        (PBMThompson, {"exam_prob": exam_prob, "prior_mean": 0.5, "seed": 1}, 1100.0),
    )
    for Algo, kwargs, limit in algos:
        algo = Algo(env.n_items, env.list_size, **kwargs)
        rewards, regrets, _ = run_ranking_bandit(env, algo, steps=steps, seed=2)
        assert rewards.shape == (steps,)
        assert regrets.shape == (steps,)
        assert regrets.sum() < limit
