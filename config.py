"""
All hyperparameters live here for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# Market environment parameters
@dataclass
class MarketConfig:
    """Parameters for the BSE-backed CDA market session."""

    # Background trader counts  (ZIC & ZIP mix)
    n_buyers: int = 5
    n_sellers: int = 5
    zic_ratio: float = 0.6          # fraction of background traders that are ZIC (rest ZIP)

    # Background customer order price range
    min_price: int = 10
    max_price: int = 200

    # Episode / session length (BSE time units)
    start_time: float = 0.0
    end_time: float = 500.0         # one episode = 500 BSE time-units

    # The RL agent can only see a maximum of depth-5 order book.
    lob_depth: int = 5

    # RL agent limits
    initial_cash: float = 10000.0
    max_inventory: int = 10         # position limit ±

    # Transaction cost rate applied to RL agent fills (fraction of trade value)
    transaction_cost_rate: float = 0.001
    force_flatten_on_end: bool = True


# PPO agent parameters
@dataclass
class AgentConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048             # rollout buffer size
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01          # entropy bonus (exploration)

    # Reward coefficients
    inventory_penalty: float = 0.015
    transaction_cost_rate: float = 0.001
    no_trade_penalty: float = 0.2
    trade_reward_bonus: float = 0.05
    drawdown_penalty: float = 0.0
    pnl_vol_penalty: float = 0.0
    pnl_vol_window: int = 10

    # Network architecture (shared for policy + value)
    net_arch: List[int] = field(default_factory=lambda: [128, 128])

    # Encoder (used when state_type == 'encoded')
    encoder_type: str = "mlp"
    encoder_hidden: int = 64
    encoder_latent: int = 8


# Experimental running parameters
@dataclass
class ExperimentConfig:
    n_runs: int = 10                # independent random seeds
    n_training_steps: int = 200000
    eval_episodes: int = 30
    verbose: int = 0                # SB3 verbosity (0=silent)
    seed_base: int = 42
