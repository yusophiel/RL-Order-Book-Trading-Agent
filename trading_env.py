"""
Gymnasium environment backed by the Bristol Stock Exchange (BSE), defining what the RL agent could see,
what it could do, and how to calculate the reward after it was completed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import AgentConfig, MarketConfig
from market.bse_market import BSEMarket, LOBSnapshot, RL_AGENT_TID, BSE_SYS_MAX_PRICE, BSE_SYS_MIN_PRICE
from representations.state_builder import StateBuilder, get_obs_dim

# Action constants
# Do not trade, and cancel the current pending order.
ACT_HOLD = 0

# Place a passive buy order.
ACT_LIMIT_BUY = 1

# Place a passive sell order.
ACT_LIMIT_SELL = 2

# Actively buy. Submit an extremely high buy price, guaranteeing a crossover,
# and immediately buy the current sell order.
ACT_MARKET_BUY = 3

# Actively sell. Submit an extremely low sell price, guaranteeing a crossover,
# and immediately sell the current buy order.
ACT_MARKET_SELL = 4

N_ACTIONS = 5


# Gymnasium environment wrapping BSEMarket.
class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        market_cfg: MarketConfig,
        agent_cfg: AgentConfig,
        state_type: str = "best_bidask",
        encoder=None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.market_cfg = market_cfg
        self.agent_cfg = agent_cfg
        self.state_type = state_type
        self._seed = seed

        # Create the underlying market
        self.market = BSEMarket(market_cfg)

        # State builder (combines market_cfg + agent_cfg lob_depth / encoder_latent)
        merged_cfg = _MergedCfg(market_cfg, agent_cfg)
        self.state_builder = StateBuilder(state_type, merged_cfg, encoder)

        # Define observation space
        obs_dim = get_obs_dim(state_type, market_cfg.lob_depth,
                              agent_cfg.encoder_latent)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Define action space
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state
        self._cash: float = 0.0  # Agent Current Cash
        self._inventory: int = 0  # Agent Current Positions
        self._prev_pnl: float = 0.0  # Previous PnL
        self._transaction_cost_step: float = 0.0  # Transaction Cost of Current Step
        self._transaction_cost_cum: float = 0.0  # Cumulative Transaction Costs
        self._snap: Optional[LOBSnapshot] = None  # Current LOBSnapshot
        self._step_count: int = 0  # Current Episode Step Number
        self._trade_count: int = 0  # Agent Number of Trades
        self._peak_pnl: float = 0.0  # Highest All-Time PnL
        # Recent PnL Changes, used to calculate PnL volatility
        self._recent_delta_pnl = deque(maxlen=max(1, agent_cfg.pnl_vol_window))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        use_seed = seed if seed is not None else self._seed

        # Reset agent account
        self._cash = self.market_cfg.initial_cash
        self._inventory = 0
        self._prev_pnl = 0.0

        # Clear transaction statistics
        self._transaction_cost_step = 0.0
        self._transaction_cost_cum = 0.0
        self._step_count = 0
        self._trade_count = 0
        self._peak_pnl = 0.0
        self._recent_delta_pnl.clear()

        # Resetting the BSE Market
        self._snap = self.market.reset(seed=use_seed)
        obs = self._make_obs(self._snap)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Check if the action is legal
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._transaction_cost_step = 0.0

        # Execute the RL action against the current visible market snapshot.
        # This keeps the action timing aligned with the observation the agent
        # actually received at the previous step/reset.
        current_snap = self._snap
        trade_executed = self._execute_action(action, current_snap)
        self._snap = self.market.publish_snapshot()

        # Then let the background market evolve once and settle any passive
        # fills the agent receives from resting orders.
        self._snap = self.market.step_background()
        passive_trades = self.market.consume_agent_trades()
        for trade, agent_otype in passive_trades:
            self._settle_trade(trade, agent_otype)
            trade_executed = True

        terminated = self.market.is_done
        if terminated and self.market_cfg.force_flatten_on_end:
            if self._liquidate_inventory(self._snap):
                trade_executed = True

        # Update PnL
        mid = self._snap.mid_price or (
            (self.market_cfg.min_price + self.market_cfg.max_price) / 2.0
        )
        pnl = self._cash + self._inventory * mid - self.market_cfg.initial_cash

        # Compute reward
        delta_pnl = pnl - self._prev_pnl
        self._peak_pnl = max(self._peak_pnl, pnl)
        self._recent_delta_pnl.append(delta_pnl)
        drawdown = self._peak_pnl - pnl
        pnl_vol = 0.0
        if len(self._recent_delta_pnl) >= 2:
            pnl_vol = float(np.std(self._recent_delta_pnl))
        inv_penalty = self.agent_cfg.inventory_penalty * abs(self._inventory)
        idle_penalty = 0.0
        if (not trade_executed) and self._inventory == 0:
            idle_penalty = self.agent_cfg.no_trade_penalty
        drawdown_penalty = self.agent_cfg.drawdown_penalty * drawdown
        vol_penalty = self.agent_cfg.pnl_vol_penalty * pnl_vol
        reward = float(
            delta_pnl
            - inv_penalty
            - self._transaction_cost_step
            - idle_penalty
            - drawdown_penalty
            - vol_penalty
        )
        if trade_executed:
            reward += self.agent_cfg.trade_reward_bonus

        self._prev_pnl = pnl
        self._step_count += 1

        truncated = False

        obs = self._make_obs(self._snap)
        info = {
            "pnl": pnl,
            "inventory": self._inventory,
            "cash": self._cash,
            "last_trade": self._snap.last_trade_price,
            "trade_executed": trade_executed,
            "trade_count": self._trade_count,
            "transaction_cost_step": self._transaction_cost_step,
            "transaction_cost_cum": self._transaction_cost_cum,
            "drawdown": drawdown,
            "pnl_vol": pnl_vol,
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    # Construction observation
    def _make_obs(self, snap: LOBSnapshot) -> np.ndarray:
        obs = self.state_builder.build(
            snap=snap,
            inventory=self._inventory,
            cash=self._cash,
            pnl=self._prev_pnl,
            price_history=self.market.price_history,
            market_volatility=self.market.short_term_volatility(10),
        )
        # Overwrite the time-remaining slot (3rd agent feature, index -1)
        obs[-1] = float(self.market.time_remaining_fraction)
        return obs

    # Convert action to BSE order
    def _execute_action(self, action: int, snap: LOBSnapshot) -> bool:
        cfg = self.market_cfg
        mid = snap.mid_price or ((cfg.min_price + cfg.max_price) / 2.0)
        tick = 1  # BSE uses integer prices

        # `at_max` indicates that too many have been bought and no more can be bought.
        # `at_min` indicates that too many have been sold out and no more can be sold.
        at_max = self._inventory >= cfg.max_inventory
        at_min = self._inventory <= -cfg.max_inventory

        # Cancel currently pending orders without submitting new orders.
        if action == ACT_HOLD:
            self.market.cancel_agent_order()
            return False

        elif action == ACT_LIMIT_BUY:
            if at_max:
                return False
            # Passive limit bid: join the best bid and wait to be lifted.
            if snap.best_bid is not None:
                price = int(snap.best_bid)
            else:
                price = int(mid - tick)
            price = max(cfg.min_price, price)
            trade = self.market.submit_agent_order('Bid', price)
            if trade is not None:
                self._settle_trade(trade, 'Bid')
                return True
            return False

        elif action == ACT_LIMIT_SELL:
            if at_min:
                return False
            # Passive limit ask: join the best ask and wait to be hit.
            if snap.best_ask is not None:
                price = int(snap.best_ask)
            else:
                price = int(mid + tick)
            price = min(cfg.max_price, price)
            trade = self.market.submit_agent_order('Ask', price)
            if trade is not None:
                self._settle_trade(trade, 'Ask')
                return True
            return False

        elif action == ACT_MARKET_BUY:
            if at_max:
                return False
            # Market buy: post Bid at system max → guaranteed cross
            trade = self.market.submit_agent_order('Bid', BSE_SYS_MAX_PRICE)
            if trade is not None:
                self._settle_trade(trade, 'Bid')
                return True
            return False

        elif action == ACT_MARKET_SELL:
            if at_min:
                return False
            # Market sell: post Ask at system min → guaranteed cross
            trade = self.market.submit_agent_order('Ask', BSE_SYS_MIN_PRICE)
            if trade is not None:
                self._settle_trade(trade, 'Ask')
                return True
            return False

        return False

    # Update cash and inventory when the RL agent is a party to a BSE trade.
    def _settle_trade(self, trade: dict, agent_otype: str) -> None:
        price = trade['price']
        qty = trade['qty']
        tc_rate = self.agent_cfg.transaction_cost_rate
        agent_is_party = (
            trade['party1'] == RL_AGENT_TID or
            trade['party2'] == RL_AGENT_TID
        )
        if not agent_is_party:
            return

        tc = tc_rate * price * qty
        if agent_otype == 'Bid':  # agent bought
            # Cash decreased, holdings increased
            self._cash -= price * qty + tc
            self._inventory += qty
        else:                      # agent sold
            # Cash increased, holdings decreased
            self._cash += price * qty - tc
            self._inventory -= qty
        self._transaction_cost_step += tc
        self._transaction_cost_cum += tc
        self._trade_count += 1

    def _liquidate_inventory(self, snap: LOBSnapshot) -> bool:
        if self._inventory == 0:
            return False

        if self._inventory > 0:
            price = snap.best_bid if snap.best_bid is not None else (
                snap.mid_price or ((self.market_cfg.min_price + self.market_cfg.max_price) / 2.0)
            )
            qty = self._inventory
            tc = self.agent_cfg.transaction_cost_rate * price * qty
            self._cash += price * qty - tc
            self._inventory = 0
        else:
            price = snap.best_ask if snap.best_ask is not None else (
                snap.mid_price or ((self.market_cfg.min_price + self.market_cfg.max_price) / 2.0)
            )
            qty = abs(self._inventory)
            tc = self.agent_cfg.transaction_cost_rate * price * qty
            self._cash -= price * qty + tc
            self._inventory = 0

        self._transaction_cost_step += tc
        self._transaction_cost_cum += tc
        self._trade_count += 1
        return True


# Combine MarketConfig + AgentConfig into a single namespace for StateBuilder
class _MergedCfg:

    def __init__(self, mcfg: MarketConfig, acfg: AgentConfig):
        self.lob_depth = mcfg.lob_depth
        self.min_price = mcfg.min_price
        self.max_price = mcfg.max_price
        self.initial_cash = mcfg.initial_cash
        self.max_inventory = mcfg.max_inventory
        self.encoder_latent = acfg.encoder_latent
