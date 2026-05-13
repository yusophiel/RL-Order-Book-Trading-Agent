# Reinforcement Learning Trading Agents in a BSE Limit Order Book Environment

**Studying Order Book Information Representation in Reinforcement Learning Trading Agents**

This project trains and evaluates a PPO-based RL trading agent in a simulated continuous double auction market based on the [Bristol Stock Exchange](https://github.com/davecliff/BristolStockExchange) (BSE) developed by Dave Cliff at the University of Bristol. The agent trades alongside BSE background traders (ZIC and ZIP) on the original BSE matching engine, rather than on a hand-crafted order book implementation.

---

## Project Structure

```
rl_lob_project/
├── main.py                             ← CLI entry point
├── config.py                           ← MarketConfig / AgentConfig / ExperimentConfig
├── trading_env.py                      ← Gymnasium trading environment
├── rl_policy.py                        ← SB3 PPO factory + feature extractor
├── visualization.py                    ← Metrics + plotting helpers
├── market/
│   ├── BSE.py                          ← Dave Cliff's BSE engine
│   └── bse_market.py                   ← BSE integration layer
├── outputs/
│   ├── experiment1/                    ← Exp 1 results
│   ├── experiment2/                    ← Exp 2 results
│   └── experiment3/                    ← Exp 3 results
├── representations/
│   ├── state_builder.py                ← Observation builders
│   └── encoders.py                     ← MLP / CNN / Autoencoder encoders
├── requirements.txt
│
├── experiments/
│   ├── runner.py                       ← Shared train / eval / baseline utilities
│   ├── experiment1.py                  ← Exp 1: information level comparison
│   ├── experiment2.py                  ← Exp 2: representation method comparison
│   └── experiment3.py                  ← Exp 3: PPO agents vs non-learning baselines
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, PyTorch, Stable-Baselines3, Gymnasium, NumPy, Matplotlib, SciPy

---

## Usage

### Quick smoke test (fast — for debugging)
```bash
python smoke_test.py
```

### Run all three experiments
```bash
python main.py --all
```

Some formal experiment scripts override the default training budget and may use larger PPO training budgets, depending on the experiment configuration.

### Run a single experiment
```bash
python main.py --exp 1     # Information level comparison
python main.py --exp 2     # Representation method comparison
python main.py --exp 3     # PPO agents vs non-learning baselines
```

### Custom settings
```bash
python main.py --all --n_runs 5 --n_steps 500000 --n_eval 50

# Experiment 3 with a specific best config from Exp 1/2
python main.py --exp 3 --best_state_type handcrafted
```

`Experiment 3` currently supports only `handcrafted`, `best_bidask`, or the default `both`.

### Save a representative Experiment 3 handcrafted model
```bash
python save_exp3_handcrafted_model.py
```

This trains a single representative `RL(handcrafted)` model using the fixed seed `42` under the formal Experiment 3 configuration and saves it to:

- `outputs/experiment3/models/rl_handcrafted_seed42.zip`
- `outputs/experiment3/models/rl_handcrafted_seed42.txt`

---

## Outputs

Each experiment writes figures and a plain-text summary into its own directory:

- `outputs/experiment1/`
- `outputs/experiment2/`
- `outputs/experiment3/`

Experiment 3 model outputs may also include:

- `outputs/experiment3/models/rl_handcrafted_seed42.zip`
- `outputs/experiment3/models/rl_handcrafted_seed42.txt`

Typical files include:
- `lines_total_pnl.png`
- `lines_sharpe_ratio.png`
- `lines_max_drawdown.png`
- `significance_total_pnl.png` for Welch-based experiments
- `significance_total_pnl_paired.png` for paired-test experiment outputs
- `results.txt`

---

## Reproducibility Notes

The main configuration values are defined in `config.py`. Some experiment scripts override selected settings, such as training steps, episode length, market density, or evaluation episodes, to match the formal experiment settings used in the report.

All compared methods within the same experiment use the same environment dynamics, action space, reward structure, accounting rules, inventory limits and evaluation protocol.
