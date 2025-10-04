from __future__ import annotations
import numpy as np
import csv, pathlib
from typing import Callable


def _write_metrics(rewards: np.ndarray, regrets: np.ndarray, out_dir: str) -> str:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","reward","regret","cum_reward","cum_regret"])
        cum_r = 0.0
        cum_g = 0.0
        for t, (r, g) in enumerate(zip(rewards, regrets), start=1):
            cum_r += r
            cum_g += g
            w.writerow([t, r, g, cum_r, cum_g])
    return str(out_path)

def run_bandit(env, algo, steps: int, seed: int | None = None, out_dir: str | None = None):
    rng = np.random.default_rng(seed)
    rewards = np.zeros(steps, dtype=float)
    regrets = np.zeros(steps, dtype=float)

    for t in range(steps):
        arm = algo.select_arm()
        r = env.pull(arm)
        algo.update(arm, r)
        rewards[t] = r
        regrets[t] = env.optimal_p - r

    out = _write_metrics(rewards, regrets, out_dir) if out_dir else None
    return rewards, regrets, out


def run_contextual_bandit(
    env,
    algo,
    steps: int,
    seed: int | None = None,
    out_dir: str | None = None,
):
    rng = np.random.default_rng(seed)
    rewards = np.zeros(steps, dtype=float)
    regrets = np.zeros(steps, dtype=float)

    for t in range(steps):
        _ = rng  # keep signature parity even if env has its own RNG
        contexts = env.sample_contexts()
        arm = algo.select_arm(contexts)
        reward = env.pull(arm)
        algo.update(contexts, arm, reward)
        rewards[t] = reward
        regrets[t] = env.optimal_reward - env.expected_reward(arm)

    out = _write_metrics(rewards, regrets, out_dir) if out_dir else None
    return rewards, regrets, out


def run_ranking_bandit(
    env,
    algo,
    steps: int,
    seed: int | None = None,
    out_dir: str | None = None,
):
    rng = np.random.default_rng(seed)
    rewards = np.zeros(steps, dtype=float)
    regrets = np.zeros(steps, dtype=float)

    for t in range(steps):
        _ = rng  # maintain parity with other runners
        ranking = algo.select_ranking()
        click_index, reward = env.step(ranking)
        algo.update(ranking, click_index)
        rewards[t] = reward
        regrets[t] = env.optimal_click_prob - env.expected_click_prob(ranking)

    out = _write_metrics(rewards, regrets, out_dir) if out_dir else None
    return rewards, regrets, out
