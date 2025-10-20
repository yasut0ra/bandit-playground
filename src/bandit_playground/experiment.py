from __future__ import annotations
import numpy as np
import csv, pathlib
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class BanditRunResult:
    """Container for outcomes produced by the experiment runners."""

    rewards: np.ndarray
    pseudo_regret: np.ndarray
    instant_regret: np.ndarray | None
    logdir: str | None


def _write_metrics(
    rewards: np.ndarray,
    pseudo_regrets: np.ndarray,
    out_dir: str,
    instant_regrets: np.ndarray | None = None,
) -> str:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["t", "reward", "pseudo_regret", "cum_reward", "cum_pseudo_regret"]
        if instant_regrets is not None:
            header.extend(["instant_regret", "cum_instant_regret"])
        w.writerow(header)
        cum_r = 0.0
        cum_pseudo = 0.0
        cum_instant = 0.0
        for t, (r, g) in enumerate(zip(rewards, pseudo_regrets), start=1):
            cum_r += r
            cum_pseudo += g
            row = [t, r, g, cum_r, cum_pseudo]
            if instant_regrets is not None:
                inst = float(instant_regrets[t - 1])
                cum_instant += inst
                row.extend([inst, cum_instant])
            w.writerow(row)
    return str(out_path)

def run_bandit(env, algo, steps: int, seed: int | None = None, out_dir: str | None = None):
    """Run a classic bandit loop and log both pseudo- and instant-regrets."""
    rewards = np.zeros(steps, dtype=float)
    pseudo_regrets = np.zeros(steps, dtype=float)
    instant_regrets = np.zeros(steps, dtype=float)

    for t in range(steps):
        arm = algo.select_arm()
        r = env.pull(arm)
        algo.update(arm, r)
        rewards[t] = r
        pseudo_regrets[t] = env.optimal_reward - env.expected_reward(arm)
        instant_regrets[t] = env.optimal_reward - r

    out = _write_metrics(rewards, pseudo_regrets, out_dir, instant_regrets) if out_dir else None
    return BanditRunResult(rewards, pseudo_regrets, instant_regrets, out)


def run_contextual_bandit(
    env,
    algo,
    steps: int,
    seed: int | None = None,
    out_dir: str | None = None,
):
    """Run a contextual bandit loop, returning pseudo- and realised regret traces."""
    rng = np.random.default_rng(seed)
    rewards = np.zeros(steps, dtype=float)
    pseudo_regrets = np.zeros(steps, dtype=float)
    instant_regrets = np.zeros(steps, dtype=float)

    for t in range(steps):
        _ = rng  # keep signature parity even if env has its own RNG
        contexts = env.sample_contexts()
        arm = algo.select_arm(contexts)
        reward = env.pull(arm)
        algo.update(contexts, arm, reward)
        rewards[t] = reward
        pseudo_regrets[t] = env.optimal_reward - env.expected_reward(arm)
        instant_regrets[t] = env.optimal_reward - reward

    out = _write_metrics(rewards, pseudo_regrets, out_dir, instant_regrets) if out_dir else None
    return BanditRunResult(rewards, pseudo_regrets, instant_regrets, out)


def run_ranking_bandit(
    env,
    algo,
    steps: int,
    seed: int | None = None,
    out_dir: str | None = None,
):
    """Run a ranking bandit loop with cascade/PBM feedback models."""
    rng = np.random.default_rng(seed)
    rewards = np.zeros(steps, dtype=float)
    pseudo_regrets = np.zeros(steps, dtype=float)
    instant_regrets = np.zeros(steps, dtype=float)

    for t in range(steps):
        _ = rng  # maintain parity with other runners
        ranking = algo.select_ranking()
        feedback, reward = env.step(ranking)
        algo.update(ranking, feedback)
        rewards[t] = reward
        pseudo_regrets[t] = env.optimal_reward - env.expected_reward(ranking)
        instant_regrets[t] = env.optimal_reward - reward

    out = _write_metrics(rewards, pseudo_regrets, out_dir, instant_regrets) if out_dir else None
    return BanditRunResult(rewards, pseudo_regrets, instant_regrets, out)
