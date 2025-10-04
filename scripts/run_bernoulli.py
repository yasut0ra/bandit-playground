#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np, matplotlib.pyplot as plt, pathlib
from bandit_playground.envs.bernoulli import BernoulliKArms
from bandit_playground.algorithms.epsilon_greedy import EpsilonGreedy
from bandit_playground.algorithms.ucb1 import UCB1
from bandit_playground.algorithms.thompson_bernoulli import ThompsonBernoulli
from bandit_playground.algorithms.exp3 import Exp3
from bandit_playground.algorithms.softmax import Softmax
from bandit_playground.algorithms.gradient_bandit import GradientBandit
from bandit_playground.experiment import run_bandit

ALGOS = {
    "eps": EpsilonGreedy,
    "ucb1": UCB1,
    "ts": ThompsonBernoulli,
    "exp3": Exp3,
    "softmax": Softmax,
    "grad": GradientBandit,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS.keys(), default="ts")
    ap.add_argument("--n-arms", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epsilon", type=float, default=0.1, help="for eps-greedy")
    ap.add_argument("--gamma", type=float, default=0.07, help="for EXP3")
    ap.add_argument("--tau", type=float, default=0.1, help="for softmax exploration")
    ap.add_argument("--alpha", type=float, default=0.1, help="for gradient bandit")
    ap.add_argument("--no-baseline", action="store_true", help="disable baseline in gradient bandit")
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # sample K Bernoulli means, make one clearly best
    probs = np.clip(rng.beta(2,5, size=args.n_arms), 0.01, 0.99)
    # ensure one optimal near 0.7~0.9
    idx = int(rng.integers(args.n_arms))
    probs[idx] = float(rng.uniform(0.7, 0.9))

    env = BernoulliKArms(probs, seed=args.seed)
    if args.algo == "eps":
        algo = EpsilonGreedy(env.n_arms, epsilon=args.epsilon, seed=args.seed)
    elif args.algo == "exp3":
        algo = Exp3(env.n_arms, gamma=args.gamma, seed=args.seed)
    elif args.algo == "softmax":
        algo = Softmax(env.n_arms, tau=args.tau, seed=args.seed)
    elif args.algo == "grad":
        algo = GradientBandit(
            env.n_arms,
            alpha=args.alpha,
            use_baseline=not args.no_baseline,
            seed=args.seed,
        )
    else:
        algo = ALGOS[args.algo](env.n_arms, seed=args.seed)

    _, _, out_dir = run_bandit(env, algo, args.steps, seed=args.seed, out_dir=args.out)

    # Plot cumulative regret
    import csv
    ts, cum_regret = [], []
    with open(pathlib.Path(out_dir) / "metrics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(int(row["t"]))
            cum_regret.append(float(row["cum_regret"]))

    plt.figure()
    plt.plot(ts, cum_regret, label=f"{args.algo}")
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.title("Bernoulli K-Armed Bandit")
    plt.legend()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(pathlib.Path(out_dir) / "plot.png", dpi=160)
    print(f"Saved results to: {out_dir}/metrics.csv and {out_dir}/plot.png")
    print("Means:", probs)

if __name__ == "__main__":
    main()
