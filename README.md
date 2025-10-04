# bandit-playground

Reproducible sandbox for classic and contextual bandit experiments (ベースラインからランキング系まで拡張可能).

## Features (v0.1)
- Bernoulli K-armed, Linear contextual, and Cascade ranking environments
- Epsilon-Greedy, Softmax, UCB1, UCB-Tuned, KL-UCB, Thompson Sampling (Bernoulli), LinUCB, LinTS, CascadeUCB1, CascadeThompson, EXP3 (adversarial), Gradient Bandit
- Deterministic experiment runner + plots
- Simple logging (CSV) for regret/reward
- Ready for GitHub Actions + pytest

## Quickstart
```bash
# (Optional) Create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install (editable)
pip install -e .

# Run a sample experiment (Bernoulli 10 arms, 10,000 steps)
python scripts/run_bernoulli.py --algo ucb1 --n-arms 10 --steps 10000 --seed 42

# Run a linear bandit experiment
python scripts/run_linear.py --algo linucb --n-arms 5 --dim 4 --steps 5000 --seed 0

# Run a cascade ranking bandit experiment
python scripts/run_cascade.py --algo ucb --n-items 10 --list-size 5 --steps 10000 --seed 0
```

This will emit a CSV under `runs/` and a plot `runs/plot.png`.

## Roadmap
- [x] Contextual Linear Bandits (LinUCB / LinTS)
- [x] Ranking bandits (Cascade, PBM) with synthetic click models
- [ ] JSON/YAML configs (Hydra), multi-seed sweeps
- [ ] W&B (optional) integration
- [ ] More environments (Gaussian, adversarial)
- [ ] Off-policy evaluation utilities (IPS, SNIPS, DR)

## Repo layout
```
bandit-playground/
├─ pyproject.toml
├─ README.md
├─ src/bandit_playground/
│  ├─ algorithms/{base.py,contextual_base.py,ranking_base.py,linear.py,ranking_cascade.py,...}
│  ├─ envs/{bernoulli.py,linear.py,ranking.py}
│  └─ experiment.py
├─ scripts/{run_bernoulli.py,run_linear.py,run_cascade.py}
├─ tests/{test_envs.py,test_algorithms.py,test_linear_algorithms.py,test_ranking_algorithms.py}
└─ runs/ (created at runtime)
```

## License
MIT
