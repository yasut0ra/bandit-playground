#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np, pathlib
from bandit_playground.envs.ranking import PositionBasedClickEnv
from bandit_playground.algorithms.ranking_pbm import PBMUCB1, PBMThompson
from bandit_playground.experiment import run_ranking_bandit


ALGOS = {
    "ucb": PBMUCB1,
    "ts": PBMThompson,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), default="ucb")
    ap.add_argument("--n-items", type=int, default=15)
    ap.add_argument("--list-size", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0, help="UCB exploration strength")
    ap.add_argument("--prior-mean", type=float, default=0.5)
    ap.add_argument("--prior-var", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="runs_pbm")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    attractiveness = np.clip(rng.beta(2.0, 4.0, size=args.n_items), 0.01, 0.95)
    exam = np.linspace(1.0, 0.3, num=args.list_size)

    env = PositionBasedClickEnv(attractiveness, exam_prob=exam, seed=args.seed)

    if args.algo == "ucb":
        algo = PBMUCB1(
            args.n_items,
            args.list_size,
            exam_prob=exam,
            alpha=args.alpha,
            seed=args.seed,
        )
    else:
        algo = PBMThompson(
            args.n_items,
            args.list_size,
            exam_prob=exam,
            prior_mean=args.prior_mean,
            prior_var=args.prior_var,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )

    _, _, out_dir = run_ranking_bandit(env, algo, args.steps, seed=args.seed, out_dir=args.out)

    metrics_path = pathlib.Path(out_dir) / "metrics.csv"
    print(f"Saved PBM bandit metrics to: {metrics_path}")
    print("Item attractiveness:", attractiveness)
    print("Exam probabilities:", exam)


if __name__ == "__main__":
    main()
