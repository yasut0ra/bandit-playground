from __future__ import annotations
import numpy as np
import csv, pathlib
from typing import Callable

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

    out = None
    if out_dir:
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
        out = str(out_path)
    return rewards, regrets, out
