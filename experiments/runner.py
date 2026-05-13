"""
Training, Evaluation, and Baseline Execution Toolkit
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from config import AgentConfig, ExperimentConfig, MarketConfig
from trading_env import (
    ACT_HOLD,
    ACT_LIMIT_BUY,
    ACT_LIMIT_SELL,
    ACT_MARKET_BUY,
    ACT_MARKET_SELL,
    TradingEnv,
    N_ACTIONS,
)
from rl_policy import make_ppo_agent
from visualization import aggregate_metrics, sharpe_ratio, max_drawdown, welch_t_test


# Training PPO agents
def train_agent(
    market_cfg: MarketConfig,
    agent_cfg: AgentConfig,
    state_type: str,
    n_steps: int,
    seed: int = 42,
    encoder=None,
    verbose: int = 0,
) -> PPO:
    env = TradingEnv(market_cfg, agent_cfg, state_type=state_type,
                     encoder=encoder, seed=seed)
    agent = make_ppo_agent(env, agent_cfg, state_type=state_type, seed=seed)
    agent.learn(total_timesteps=n_steps, reset_num_timesteps=True)
    env.close()
    return agent


# Evaluation
def evaluate_agent(
    agent: PPO,
    market_cfg: MarketConfig,
    agent_cfg: AgentConfig,
    state_type: str,
    n_episodes: int,
    seed_offset: int = 1000,
    eval_seeds: Optional[Sequence[int]] = None,
    encoder=None,
    deterministic: bool = True,
) -> List[Dict]:
    results = []
    seeds = list(eval_seeds) if eval_seeds is not None else [seed_offset + ep for ep in range(n_episodes)]
    for ep, eval_seed in enumerate(seeds):
        env = TradingEnv(
            market_cfg,
            agent_cfg,
            state_type=state_type,
            encoder=encoder,
            seed=eval_seed,
        )
        obs, _ = env.reset()
        done = False
        ep_pnl_series = []
        ep_inv_series = []
        ep_tc = 0.0
        ep_trades = 0
        ep_steps = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_pnl_series.append(info['pnl'])
            ep_inv_series.append(info['inventory'])
            ep_tc = float(info.get('transaction_cost_cum', ep_tc))
            ep_trades = int(info.get('trade_count', ep_trades))
            ep_steps = int(info.get('step', ep_steps))

        env.close()
        final_pnl = ep_pnl_series[-1] if ep_pnl_series else 0.0
        results.append({
            'total_pnl': final_pnl,
            'sharpe_ratio': sharpe_ratio(
                np.diff(ep_pnl_series).tolist() if len(ep_pnl_series) > 1 else [0.0]
            ),
            'max_drawdown': max_drawdown(ep_pnl_series),
            'inventory_variance': float(np.var(ep_inv_series)),
            'transaction_costs': ep_tc,
            'trade_frequency': float(ep_trades / max(1, ep_steps)),
        })
    return results


# Baselines (all use TradingEnv → BSE exchange)
# Random quoting baseline under the same inventory/PnL rules as RL
class _ZICLikePolicy:

    def __init__(self, buy_prob: float = 0.34, sell_prob: float = 0.34):
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob

    def predict(self, obs, deterministic=False):
        inv = float(obs[-3])
        draw = random.random()
        if inv > 0.8:
            return ACT_LIMIT_SELL, None
        if inv < -0.8:
            return ACT_LIMIT_BUY, None
        if draw < self.buy_prob:
            return ACT_LIMIT_BUY, None
        if draw < self.buy_prob + self.sell_prob:
            return ACT_LIMIT_SELL, None
        return ACT_HOLD, None


# Adaptive quoting baseline inspired by ZIP-style quote adjustment
class _ZIPLikePolicy:

    def predict(self, obs, deterministic=False):
        spread = float(obs[0])
        mid_ret1 = float(obs[1])
        order_imb = float(obs[4])
        depth_imb = float(obs[5])
        inv = float(obs[-3])
        time_left = float(obs[-1])

        pressure = 0.5 * order_imb + 0.5 * depth_imb

        if inv > 0.7:
            return (ACT_MARKET_SELL if time_left < 0.15 else ACT_LIMIT_SELL), None
        if inv < -0.7:
            return (ACT_MARKET_BUY if time_left < 0.15 else ACT_LIMIT_BUY), None

        if pressure > 0.28:
            return (ACT_LIMIT_BUY if spread > 0.01 else ACT_HOLD), None
        if pressure < -0.28:
            return (ACT_LIMIT_SELL if spread > 0.01 else ACT_HOLD), None

        if mid_ret1 > 0.08 and spread > 0.015:
            return ACT_LIMIT_BUY, None
        if mid_ret1 < -0.08 and spread > 0.015:
            return ACT_LIMIT_SELL, None

        return ACT_HOLD, None


# Simple mean-reversion trader with an inventory cap
class _RuleBasedPolicy:

    def predict(self, obs, deterministic=False):
        spread = float(obs[0])
        mid_ret1 = float(obs[1])
        inv = float(obs[-3])
        time_left = float(obs[-1])

        if inv > 0.6:
            return ACT_LIMIT_SELL, None
        if inv < -0.6:
            return ACT_LIMIT_BUY, None

        if spread < 0.01:
            return ACT_HOLD, None

        if mid_ret1 < -0.03:
            return ACT_LIMIT_BUY, None
        if mid_ret1 > 0.03:
            return ACT_LIMIT_SELL, None

        if time_left < 0.1:
            if inv > 0.1:
                return ACT_MARKET_SELL, None
            if inv < -0.1:
                return ACT_MARKET_BUY, None

        return ACT_HOLD, None


# Evaluate a non-RL baseline policy using TradingEnv
def evaluate_baseline(
    policy,
    market_cfg: MarketConfig,
    agent_cfg: AgentConfig,
    state_type: str,
    n_episodes: int,
    seed_offset: int = 2000,
    eval_seeds: Optional[Sequence[int]] = None,
) -> List[Dict]:
    results = []
    seeds = list(eval_seeds) if eval_seeds is not None else [seed_offset + ep for ep in range(n_episodes)]
    for ep, eval_seed in enumerate(seeds):
        env = TradingEnv(market_cfg, agent_cfg, state_type=state_type,
                         seed=eval_seed)
        obs, _ = env.reset()
        done = False
        ep_pnl_series = []
        ep_inv_series = []
        ep_tc = 0.0
        ep_trades = 0
        ep_steps = 0

        if hasattr(policy, 'price_buf'):
            policy.price_buf = []   # reset momentum buffer

        while not done:
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_pnl_series.append(info['pnl'])
            ep_inv_series.append(info['inventory'])
            ep_tc = float(info.get('transaction_cost_cum', ep_tc))
            ep_trades = int(info.get('trade_count', ep_trades))
            ep_steps = int(info.get('step', ep_steps))

        env.close()
        final_pnl = ep_pnl_series[-1] if ep_pnl_series else 0.0
        results.append({
            'total_pnl': final_pnl,
            'sharpe_ratio': sharpe_ratio(
                np.diff(ep_pnl_series).tolist() if len(ep_pnl_series) > 1 else [0.0]
            ),
            'max_drawdown': max_drawdown(ep_pnl_series),
            'inventory_variance': float(np.var(ep_inv_series)),
            'transaction_costs': ep_tc,
            'trade_frequency': float(ep_trades / max(1, ep_steps)),
        })
    return results


# Full train+eval run (single seed)
def run_single(
    market_cfg: MarketConfig,
    agent_cfg: AgentConfig,
    exp_cfg: ExperimentConfig,
    state_type: str,
    seed: int,
    encoder=None,
) -> Dict:
    agent = train_agent(
        market_cfg, agent_cfg, state_type,
        n_steps=exp_cfg.n_training_steps,
        seed=seed, encoder=encoder, verbose=exp_cfg.verbose,
    )
    ep_results = evaluate_agent(
        agent, market_cfg, agent_cfg, state_type,
        n_episodes=exp_cfg.eval_episodes,
        seed_offset=seed + 10_000,
        encoder=encoder,
    )
    agg = aggregate_metrics(ep_results)
    return {
        'aggregated': agg,
        'episode_results': ep_results,
        'run_metrics': {k: v['mean'] for k, v in agg.items()},
        'agent': agent,
    }
