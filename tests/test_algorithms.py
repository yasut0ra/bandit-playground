import numpy as np
from bandit_playground.envs.bernoulli import BernoulliKArms
from bandit_playground.algorithms.ucb1 import UCB1
from bandit_playground.algorithms.epsilon_greedy import EpsilonGreedy
from bandit_playground.algorithms.thompson_bernoulli import ThompsonBernoulli
from bandit_playground.experiment import run_bandit

def test_algorithms_basic():
    probs = np.array([0.1, 0.3, 0.8])
    env = BernoulliKArms(probs, seed=0)
    for Algo in (UCB1, EpsilonGreedy, ThompsonBernoulli):
        algo = Algo(env.n_arms, seed=0) if Algo is not EpsilonGreedy else Algo(env.n_arms, epsilon=0.1, seed=0)
        rewards, regrets, _ = run_bandit(env, algo, steps=2000, seed=0)
        assert rewards.shape == (2000,)
        assert regrets.shape == (2000,)
        # Expect decent performance: cum regret < 600 on this simple env
        assert regrets.sum() < 600
