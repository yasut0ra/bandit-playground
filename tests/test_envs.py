import numpy as np
from bandit_playground.envs.bernoulli import BernoulliKArms
from bandit_playground.envs.linear import LinearBanditEnv

def test_bernoulli_shapes():
    probs = np.array([0.1, 0.5, 0.9])
    env = BernoulliKArms(probs, seed=0)
    assert env.n_arms == 3
    for _ in range(10):
        r = env.pull(1)
        assert r in (0.0, 1.0)


def test_linear_env_shapes():
    weights = np.array([[1.0, 0.0], [0.5, -0.2]])
    env = LinearBanditEnv(weights, noise_std=0.0, seed=0)
    contexts = env.sample_contexts()
    assert contexts.shape == (2, 2)
    reward = env.pull(0)
    assert isinstance(reward, float)
    assert env.optimal_arm in (0, 1)
