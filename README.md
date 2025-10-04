# bandit-playground

Reproducible sandbox for classic and contextual bandit experiments (ベースラインからランキング系まで拡張可能).

## Features (v0.1)
- Bernoulli K-armed environment
- Epsilon-Greedy, UCB1, Thompson Sampling (Bernoulli), EXP3 (adversarial)
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
```

This will emit a CSV under `runs/` and a plot `runs/plot.png`.

## Roadmap
- [ ] Contextual Linear Bandits (LinUCB / LinTS)
- [ ] Ranking bandits (Cascade, PBM) with synthetic click models
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
│  ├─ algorithms/{base.py,epsilon_greedy.py,ucb1.py,thompson_bernoulli.py}
│  ├─ envs/{bernoulli.py}
│  └─ experiment.py
├─ scripts/run_bernoulli.py
├─ tests/{test_envs.py,test_algorithms.py}
└─ runs/ (created at runtime)
```

## License
MIT
