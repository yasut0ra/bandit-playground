#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np, pathlib
from bandit_playground.envs.ranking import CascadeClickEnv
from bandit_playground.algorithms.ranking_cascade import CascadeUCB1, CascadeThompson
from bandit_playground.experiment import run_ranking_bandit


ALGOS = {
    "ucb": CascadeUCB1,
    "ts": CascadeThompson,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), default="ucb")
    ap.add_argument("--n-items", type=int, default=10)
    ap.add_argument("--list-size", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.5, help="UCB exploration strength")
    ap.add_argument("--ts-alpha", type=float, default=1.0, help="Thompson prior alpha")
    ap.add_argument("--ts-beta", type=float, default=1.0, help="Thompson prior beta")
    ap.add_argument("--out", type=str, default="runs_cascade")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    probs = np.clip(rng.beta(2.0, 4.0, size=args.n_items), 0.01, 0.9)
    env = CascadeClickEnv(probs, list_size=args.list_size, seed=args.seed)

    if args.algo == "ucb":
        algo = CascadeUCB1(args.n_items, args.list_size, alpha=args.alpha, seed=args.seed)
    else:
        algo = CascadeThompson(
            args.n_items,
            args.list_size,
            alpha_prior=args.ts_alpha,
            beta_prior=args.ts_beta,
            seed=args.seed,
        )

    _, _, out_dir = run_ranking_bandit(env, algo, args.steps, seed=args.seed, out_dir=args.out)

    metrics_path = pathlib.Path(out_dir) / "metrics.csv"
    print(f"Saved cascade bandit metrics to: {metrics_path}")
    print("Item click probabilities:", probs)


if __name__ == "__main__":
    main()
