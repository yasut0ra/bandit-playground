import numpy as np
from bandit_playground.envs.ranking import CascadeClickEnv
from bandit_playground.algorithms.ranking_cascade import CascadeUCB1, CascadeThompson
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
