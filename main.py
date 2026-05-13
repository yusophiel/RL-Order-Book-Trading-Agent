"""
Entry point for all three experiments.

# Quick smoke test (fast, for CI / debugging)
python main.py --all --quick

# Full run: starts from the default config of 10 runs × 200k steps,
# but formal experiment scripts may raise training steps to at least 350k.
python main.py --all

# Single experiment
python main.py --exp 1
python main.py --exp 2
python main.py --exp 3

# Custom settings
python main.py --all --n_runs 5 --n_steps 100000 --n_eval 10
"""

import argparse
import sys
import os
from pathlib import Path


# Return the project root directory
def _project_root() -> Path:
    return Path(__file__).resolve().parent


# Make sure the required runtime environment is available
def _ensure_runtime_env() -> None:
    root = _project_root()

    mpl_cache = root / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))

    venv_python = root / ".venv" / "bin" / "python"
    if Path(sys.executable).resolve() == venv_python.resolve():
        return

    try:
        import torch
        import stable_baselines3
        if not getattr(torch, "__version__", None):
            raise ImportError("Broken torch install: missing __version__")
    except Exception:
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python), str(root / "main.py"), *sys.argv[1:]])
        raise


# Check and prepare the runtime environment before importing project modules.
_ensure_runtime_env()

# Make sure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AgentConfig, ExperimentConfig, MarketConfig
from experiments.experiment1 import run_experiment1
from experiments.experiment2 import run_experiment2
from experiments.experiment3 import run_experiment3


# Parse command-line arguments
def parse_args():
    p = argparse.ArgumentParser(description="RL LOB Trading Agent – BSE")
    p.add_argument('--all', action='store_true', help='Run all 3 experiments')
    p.add_argument('--exp', type=int, choices=[1, 2, 3], help='Run a single experiment')
    p.add_argument('--quick', action='store_true',
                   help='Quick smoke-test mode (few runs, few steps)')
    p.add_argument('--n_runs', type=int, default=None)
    p.add_argument('--n_steps', type=int, default=None)
    p.add_argument('--n_eval', type=int, default=None)
    p.add_argument(
        '--best_state_type',
        choices=['both', 'handcrafted', 'best_bidask'],
        default='both',
        help='State type for Experiment 3 RL agent',
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Default configurations
    market_cfg = MarketConfig(n_buyers=5, n_sellers=5)
    agent_cfg = AgentConfig()
    exp_cfg = ExperimentConfig()

    # If quick mode is enabled, use a smaller and faster test setup
    if args.quick:
        market_cfg.n_buyers = 3
        market_cfg.n_sellers = 3
        market_cfg.end_time = 200.0
        exp_cfg.n_runs = 2
        exp_cfg.n_training_steps = 5_000
        exp_cfg.eval_episodes = 3

    # Override default experiment settings if provided from the command line
    if args.n_runs is not None:
        exp_cfg.n_runs = args.n_runs
    if args.n_steps is not None:
        exp_cfg.n_training_steps = args.n_steps
    if args.n_eval is not None:
        exp_cfg.eval_episodes = args.n_eval

    # Run selected experiments
    if not args.all and args.exp is None:
        print("Please specify --all or --exp {1,2,3}")
        sys.exit(1)

    # This stores Experiment 1 results so they can be reused by Experiment 2
    exp1_payload = None
    if args.all or args.exp == 1:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Information Level Comparison")
        print("=" * 60)
        # Run Experiment 1 and save its output
        exp1_payload = run_experiment1(market_cfg, agent_cfg, exp_cfg)

    if args.all or args.exp == 2:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Representation Method Comparison")
        print("=" * 60)

        # If Experiment 1 was already run, reuse its best_bidask result
        # as a reference for Experiment 2
        best_bidask_reference = None
        if exp1_payload is not None:
            best_bidask_reference = {
                'aggregated': exp1_payload['results']['best_bidask']['aggregated'],
                'metric_series': {
                    metric: exp1_payload['metric_series'][metric]['best_bidask']
                    for metric in exp1_payload['metric_series']
                },
            }
        run_experiment2(
            market_cfg, agent_cfg, exp_cfg,
            best_bidask_reference=best_bidask_reference,
        )

    if args.all or args.exp == 3:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: RL vs Traditional Baselines")
        print("=" * 60)
        # Run Experiment 3 using the selected best state type
        run_experiment3(
            market_cfg, agent_cfg, exp_cfg,
            best_state_type=args.best_state_type,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
