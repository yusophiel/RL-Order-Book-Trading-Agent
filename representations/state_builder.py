"""
Converts a BSE LOBSnapshot into a fixed-length numpy observation vector.

  Experiment 1 – information level
    'price_only'   – last price + lagged returns + volatility  (8 dims)
    'best_bidask'  – above + best bid/ask/spread/mid           (12 dims)
    'raw_lob'      – depth-D prices & quantities both sides + last trade

  Experiment 2 – representation method
    'handcrafted'  – engineered microstructure features        (16 dims)
    'encoded'      – raw LOB passed through learned encoder    (variable)

All vectors are augmented with 3 agent-level features:
    normalised inventory, normalised PnL, time remaining fraction.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from market.bse_market import LOBSnapshot

# Number of agent-level features appended to every state vector
AGENT_FEATURES = 3


# Raw LOB vector length: lob_depth × (bid_price, bid_qty, ask_price, ask_qty)
def raw_lob_dim(lob_depth: int) -> int:
    return 4 * lob_depth


# Return the observation length for different state types.
def get_obs_dim(state_type: str, lob_depth: int, encoder_latent: int = 8) -> int:
    base = {
        'price_only':  8,
        'best_bidask': 12,
        'raw_lob':     raw_lob_dim(lob_depth) + 1,  # +1 for last trade
        'handcrafted': 16,
        'encoded':     encoder_latent + AGENT_FEATURES,
    }
    if state_type == 'encoded':
        return base['encoded']
    return base[state_type] + AGENT_FEATURES


# Builds normalised observation vectors from BSE market state.
class StateBuilder:
    def __init__(self, state_type: str, cfg, encoder=None):
        self.state_type = state_type
        self.cfg = cfg
        self.encoder = encoder
        self.obs_dim = get_obs_dim(
            state_type, cfg.lob_depth,
            cfg.__dict__.get('encoder_latent', 8)
        )

        # Preserve price normalization upper and lower bounds
        self._p_min = float(cfg.min_price)
        self._p_max = float(cfg.max_price)

    # Generate the final observation based on state_type
    def build(
        self,
        snap: LOBSnapshot,
        inventory: int,
        cash: float,
        pnl: float,
        price_history: List[float],
        market_volatility: float,
    ) -> np.ndarray:
        builder = {
            'price_only':  self._price_only,
            'best_bidask': self._best_bidask,
            'raw_lob':     self._raw_lob,
            'handcrafted': self._handcrafted,
            'encoded':     self._encoded,
        }[self.state_type]

        agent_feats = self._agent_features(inventory, pnl)

        if self.state_type == 'encoded':
            # encoder already appends agent features
            vec = builder(snap, price_history, market_volatility, agent_feats)
        else:
            market_vec = builder(snap, price_history, market_volatility)
            vec = np.concatenate([market_vec, agent_feats])

        return vec.astype(np.float32)

    # Price normalization
    def _norm_price(self, p: Optional[float]) -> float:
        if p is None:
            return 0.5
        return (float(p) - self._p_min) / (self._p_max - self._p_min + 1e-8)

    # Calculate the lagged rate of return
    def _last_returns(self, price_history: List[float], lags=(1, 5, 10)) -> np.ndarray:
        ret = []
        for lag in lags:
            if len(price_history) > lag:
                r = math.log(
                    (price_history[-1] + 1e-6) / (price_history[-lag - 1] + 1e-6)
                )
                ret.append(np.clip(r, -1.0, 1.0))
            else:
                ret.append(0.0)
        return np.array(ret, dtype=np.float32)

    # Price base status
    def _price_only(self, snap, price_history, vol) -> np.ndarray:
        last = self._norm_price(snap.last_trade_price or snap.mid_price)
        rets = self._last_returns(price_history)
        norm_vol = min(vol / (self._p_max - self._p_min + 1e-8), 1.0)
        # pad to 8 dims
        return np.array([last, *rets, norm_vol, 0.0, 0.0, 0.0], dtype=np.float32)

    # top-of-book status
    # 12-dim: price_only(8) + best_bid + best_ask + spread + mid
    def _best_bidask(self, snap, price_history, vol) -> np.ndarray:
        base = self._price_only(snap, price_history, vol)
        mid = snap.mid_price
        spread = snap.spread
        extra = np.array([
            self._norm_price(snap.best_bid),
            self._norm_price(snap.best_ask),
            min((spread or 0) / (self._p_max - self._p_min + 1e-8), 1.0),
            self._norm_price(mid),
        ], dtype=np.float32)
        return np.concatenate([base, extra])

    # 4*D+1 dims: depth-D bid/ask prices & quantities + last trade
    def _raw_lob(self, snap, price_history, vol) -> np.ndarray:
        D = self.cfg.lob_depth
        vec = np.zeros(4 * D + 1, dtype=np.float32)
        for i, (p, q) in enumerate(snap.bid_levels[:D]):
            vec[2 * i] = self._norm_price(p)
            vec[2 * i + 1] = min(q / 10.0, 1.0)
        offset = 2 * D
        for i, (p, q) in enumerate(snap.ask_levels[:D]):
            vec[offset + 2 * i] = self._norm_price(p)
            vec[offset + 2 * i + 1] = min(q / 10.0, 1.0)
        vec[-1] = self._norm_price(snap.last_trade_price)
        return vec

    # 16-dim engineered microstructure features:
    #   spread, mid_return_1, mid_return_5, mid_return_10,
    #   order_imbalance, depth_imbalance, micro_price_deviation, volatility,
    #   bid_depth_1, ask_depth_1, total_bid_depth, total_ask_depth,
    #   bid_price_1, ask_price_1, last_trade, recent_trade_direction
    def _handcrafted(self, snap, price_history, vol) -> np.ndarray:
        mid = snap.mid_price or ((self._p_min + self._p_max) / 2)
        spread_norm = min((snap.spread or 0) / (self._p_max - self._p_min + 1e-8), 1.0)
        mid_ret1 = self._last_returns(price_history, lags=(1,))[0]
        mid_ret5 = self._last_returns(price_history, lags=(5,))[0]
        mid_ret10 = self._last_returns(price_history, lags=(10,))[0]

        # order imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
        b_qty = sum(q for _, q in snap.bid_levels) if snap.bid_levels else 0
        a_qty = sum(q for _, q in snap.ask_levels) if snap.ask_levels else 0
        total_qty = b_qty + a_qty
        order_imb = (b_qty - a_qty) / (total_qty + 1e-8)

        # depth imbalance: same but for top-1 level only
        b1 = snap.bid_levels[0][1] if snap.bid_levels else 0
        a1 = snap.ask_levels[0][1] if snap.ask_levels else 0
        depth_imb = (b1 - a1) / (b1 + a1 + 1e-8)

        micro_price = mid
        if snap.best_bid is not None and snap.best_ask is not None and (b1 + a1) > 0:
            micro_price = (
                snap.best_ask * b1 + snap.best_bid * a1
            ) / (b1 + a1 + 1e-8)
        micro_price_dev = np.clip(
            (micro_price - mid) / (self._p_max - self._p_min + 1e-8),
            -1.0,
            1.0,
        )

        norm_vol = min(vol / (self._p_max - self._p_min + 1e-8), 1.0)
        bid1_p = self._norm_price(snap.best_bid)
        ask1_p = self._norm_price(snap.best_ask)
        last = self._norm_price(snap.last_trade_price)
        total_bid_depth = min(b_qty / 50.0, 1.0)
        total_ask_depth = min(a_qty / 50.0, 1.0)

        recent_trade_direction = 0.0
        if snap.last_trade_price is not None:
            recent_trade_direction = 1.0 if snap.last_trade_price >= mid else -1.0

        return np.array([
            spread_norm,
            mid_ret1,
            mid_ret5,
            mid_ret10,
            order_imb,
            depth_imb,
            micro_price_dev,
            norm_vol,
            min(b1 / 10, 1.0),
            min(a1 / 10, 1.0),
            total_bid_depth,
            total_ask_depth,
            bid1_p,
            ask1_p,
            last,
            recent_trade_direction,
        ], dtype=np.float32)

    # Compress raw LOB with a learned encoder, then append agent features.
    #         Returns encoder_latent + AGENT_FEATURES dims.
    def _encoded(self, snap, price_history, vol, agent_feats) -> np.ndarray:
        raw = self._raw_lob(snap, price_history, vol)
        encoded = self.encoder.encode(raw)        # ndarray, shape (latent_dim,)
        return np.concatenate([encoded, agent_feats]).astype(np.float32)

    # 3-dim: normalised inventory, normalised PnL, time-remaining (set externally via snap.time)
    def _agent_features(self, inventory: int, pnl: float) -> np.ndarray:
        max_inv = float(self.cfg.max_inventory)
        norm_inv = np.clip(inventory / (max_inv + 1e-8), -1.0, 1.0)
        norm_pnl = np.clip(pnl / (self.cfg.initial_cash + 1e-8), -1.0, 1.0)
        # time_remaining is added by TradingEnv via snap attributes
        return np.array([norm_inv, norm_pnl, 0.0], dtype=np.float32)
