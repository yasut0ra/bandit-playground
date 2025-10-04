import numpy as np
from bandit_playground.envs.linear import LinearBanditEnv
from bandit_playground.algorithms.linear import LinUCB, LinThompson
from bandit_playground.experiment import run_contextual_bandit


def test_linear_algorithms_basic():
    weights = np.array(
        [
            [1.0, 0.2, -0.3],
            [0.5, 0.4, 0.1],
            [0.8, -0.1, 0.6],
        ]
    )
    steps = 1500
    algos = (
        (LinUCB, {"alpha": 0.6, "seed": 0}, 220.0),
        (LinThompson, {"v": 0.2, "seed": 0}, 260.0),
    )
    for Algo, kwargs, limit in algos:
        env = LinearBanditEnv(weights, noise_std=0.05, seed=0)
        algo = Algo(weights.shape[0], weights.shape[1], **kwargs)
        rewards, regrets, _ = run_contextual_bandit(env, algo, steps=steps, seed=1)
        assert rewards.shape == (steps,)
        assert regrets.shape == (steps,)
        assert regrets.sum() < limit
