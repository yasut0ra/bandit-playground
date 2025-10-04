import numpy as np
from bandit_playground.envs.bernoulli import BernoulliKArms

def test_bernoulli_shapes():
    probs = np.array([0.1, 0.5, 0.9])
    env = BernoulliKArms(probs, seed=0)
    assert env.n_arms == 3
    for _ in range(10):
        r = env.pull(1)
        assert r in (0.0, 1.0)
