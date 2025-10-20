#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np, pathlib
from bandit_playground.envs.linear import LinearBanditEnv
from bandit_playground.algorithms.linear import LinUCB, LinThompson, OFUL
from bandit_playground.experiment import run_contextual_bandit

ALGOS = {
    "linucb": LinUCB,
    "lints": LinThompson,
    "oful": OFUL,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), default="linucb")
    ap.add_argument("--n-arms", type=int, default=5)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0, help="exploration for LinUCB")
    ap.add_argument("--v", type=float, default=0.2, help="posterior scale for LinTS")
    ap.add_argument("--oful-lambda", type=float, default=1.0, help="ridge regularization for OFUL")
    ap.add_argument("--oful-delta", type=float, default=0.1, help="confidence level for OFUL")
    ap.add_argument("--oful-context-bound", type=float, default=5.0, help="L-infty bound on context norms")
    ap.add_argument("--oful-theta-bound", type=float, default=None, help="bound on parameter norm (defaults to env max norm)")
    ap.add_argument("--oful-noise-scale", type=float, default=None, help="sub-Gaussian noise scale (defaults to env noise_std)")
    ap.add_argument("--out", type=str, default="runs_linear")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 1.0, size=(args.n_arms, args.dim))
    env = LinearBanditEnv(weights, noise_std=args.noise_std, seed=args.seed)

    if args.algo == "linucb":
        algo = LinUCB(args.n_arms, args.dim, alpha=args.alpha, seed=args.seed)
    elif args.algo == "lints":
        algo = LinThompson(args.n_arms, args.dim, v=args.v, seed=args.seed)
    else:
        theta_bound = args.oful_theta_bound
        if theta_bound is None:
            theta_bound = float(np.linalg.norm(weights, axis=1).max())
        noise_scale = args.oful_noise_scale
        if noise_scale is None:
            noise_scale = args.noise_std if args.noise_std > 0 else 0.1
        algo = OFUL(
            args.n_arms,
            args.dim,
            lambda_=args.oful_lambda,
            delta=args.oful_delta,
            context_bound=args.oful_context_bound,
            theta_bound=theta_bound,
            noise_scale=noise_scale,
            seed=args.seed,
        )

    result = run_contextual_bandit(env, algo, args.steps, seed=args.seed, out_dir=args.out)
    out_dir = result.logdir

    # simple aggregate stats
    metrics_path = pathlib.Path(out_dir) / "metrics.csv"
    print(f"Saved linear bandit metrics to: {metrics_path}")
    print("Weights (per arm):", weights)


if __name__ == "__main__":
    main()
