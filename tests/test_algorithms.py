import numpy as np
from bandit_playground.envs.bernoulli import BernoulliKArms
from bandit_playground.algorithms.ucb1 import UCB1
from bandit_playground.algorithms.epsilon_greedy import EpsilonGreedy
from bandit_playground.algorithms.thompson_bernoulli import ThompsonBernoulli
from bandit_playground.algorithms.exp3 import Exp3
from bandit_playground.algorithms.softmax import Softmax
from bandit_playground.algorithms.gradient_bandit import GradientBandit
from bandit_playground.experiment import run_bandit

def test_algorithms_basic():
    probs = np.array([0.1, 0.3, 0.8])
    env = BernoulliKArms(probs, seed=0)
    algos = (
        (UCB1, {"seed": 0}, 600),
        (EpsilonGreedy, {"epsilon": 0.1, "seed": 0}, 600),
        (ThompsonBernoulli, {"seed": 0}, 600),
        (Exp3, {"gamma": 0.05, "seed": 0}, 900),
        (Softmax, {"tau": 0.2, "seed": 0}, 800),
        (GradientBandit, {"alpha": 0.1, "seed": 0}, 950),
    )
    for Algo, kwargs, limit in algos:
        algo = Algo(env.n_arms, **kwargs)
        rewards, regrets, _ = run_bandit(env, algo, steps=2000, seed=0)
        assert rewards.shape == (2000,)
        assert regrets.shape == (2000,)
        assert regrets.sum() < limit
