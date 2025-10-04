import numpy as np
from bandit_playground.envs.bernoulli import BernoulliKArms
from bandit_playground.envs.linear import LinearBanditEnv
from bandit_playground.envs.ranking import CascadeClickEnv, PositionBasedClickEnv

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


def test_ranking_env_clicks():
    probs = np.array([0.3, 0.6, 0.1])
    env = CascadeClickEnv(probs, list_size=2, seed=1)
    ranking = [1, 0]
    click_index, reward = env.step(ranking)
    assert click_index in (0, 1, None)
    assert reward in (0.0, 1.0)
    expected = env.expected_click_prob(ranking)
    assert 0.0 <= expected <= 1.0
    assert env.optimal_ranking[0] == 1


def test_pbm_env_clicks():
    attractiveness = np.array([0.8, 0.5, 0.3])
    exam_prob = np.array([1.0, 0.6])
    env = PositionBasedClickEnv(attractiveness, exam_prob=exam_prob, seed=2)
    ranking = [0, 1]
    clicks, reward = env.step(ranking)
    assert clicks.shape == (2,)
    assert reward == clicks.sum()
    expected = env.expected_reward(ranking)
    assert 0.0 <= expected <= 2.0
    assert env.optimal_ranking[0] == 0
