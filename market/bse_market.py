"""
The interface layer (wrapper) between RL and BSE
The RL agent is treated as a special trader whose ID is RL_AGENT_TID.
"""

from __future__ import annotations

import math
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Import BSE from this folder.
_BSE_DIR = os.path.dirname(__file__)
if _BSE_DIR not in sys.path:
    sys.path.insert(0, _BSE_DIR)

from BSE import (
    Exchange,
    Order,
    populate_market,
    customer_orders,
    bse_sys_maxprice,
    bse_sys_minprice,
)

# BSE hard limits (must match constants inside BSE.py)
BSE_SYS_MIN_PRICE = bse_sys_minprice
BSE_SYS_MAX_PRICE = bse_sys_maxprice

RL_AGENT_TID = "RL_AGENT"


# The original order book data of BSE is encapsulated into a structured market state.
@dataclass
class LOBSnapshot:

    time: float
    best_bid: Optional[int]
    best_ask: Optional[int]
    bid_levels: List[Tuple[int, int]]   # [(price, qty), ...] best-first, up to lob_depth
    ask_levels: List[Tuple[int, int]]   # [(price, qty), ...] best-first (ascending), up to lob_depth
    last_trade_price: Optional[int]     # most recent transaction price from tape
    tape: list                          # raw BSE tape (last N records)

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        if self.best_bid is not None:
            return float(self.best_bid)
        if self.best_ask is not None:
            return float(self.best_ask)
        return None

    @property
    def spread(self) -> Optional[int]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


# Create a background market
# Enable ZIC/ZIP background traders to continuously place orders
# Enable RL agents to insert orders as special traders
# Call the BSE matching engine to process transactions
# Convert the order book into LOBSnapshots for use by the RL environment
class BSEMarket:

    def __init__(self, cfg):
        self.cfg = cfg
        self.exchange: Exchange = None
        self.traders: Dict = {}
        self.trader_stats: Dict = {}
        self.pending_cust_orders: List = []
        self.time: float = 0.0

        # price history for volatility / returns features
        self.price_history: List[float] = []

        # order schedule fed to customer_orders()
        self._order_schedule = self._build_order_schedule()

        # quote-id counter for agent orders (BSE assigns real QID internally)
        self._agent_qid: int = 0

        # last order the agent posted (needed so we can cancel it)
        self.agent_last_order: Optional[Order] = None
        self._agent_trades: List[Tuple[dict, str]] = []

    # The number of ZIC/ZIP buyers and sellers in the market determines the number of buyers and sellers.
    def _build_trader_spec(self) -> dict:
        n_b = self.cfg.n_buyers
        n_s = self.cfg.n_sellers
        r = self.cfg.zic_ratio

        n_b_zic = max(1, round(n_b * r))
        n_b_zip = n_b - n_b_zic
        n_s_zic = max(1, round(n_s * r))
        n_s_zip = n_s - n_s_zic

        buyers_spec = [('ZIC', n_b_zic)]
        if n_b_zip > 0:
            buyers_spec.append(('ZIP', n_b_zip))

        sellers_spec = [('ZIC', n_s_zic)]
        if n_s_zip > 0:
            sellers_spec.append(('ZIP', n_s_zip))

        return {'buyers': buyers_spec, 'sellers': sellers_spec}

    # Rules for generating "customer task orders" for BSE background traders
    def _build_order_schedule(self) -> dict:
        lo = self.cfg.min_price
        hi = self.cfg.max_price
        t0 = self.cfg.start_time
        t1 = self.cfg.end_time
        order_interval = max(1.0, (t1 - t0) / 50.0)   # ~50 replenishment cycles per session
        return {
            'sup': [{'from': t0, 'to': t1, 'ranges': [(lo, hi)], 'stepmode': 'random'}],
            'dem': [{'from': t0, 'to': t1, 'ranges': [(lo, hi)], 'stepmode': 'random'}],
            'interval': order_interval,
            'timemode': 'drip-jitter',
        }

    def reset(self, seed: Optional[int] = None) -> LOBSnapshot:
        if seed is not None:
            random.seed(seed)

        self.exchange = Exchange()
        self.traders = {}
        self.pending_cust_orders = []
        self.time = self.cfg.start_time
        self.price_history = []
        self.agent_last_order = None
        self._agent_trades = []
        self._agent_qid = 0

        trader_spec = self._build_trader_spec()
        self.trader_stats = populate_market(trader_spec, self.traders, True, False)

        return self.publish_snapshot()

    # With each background trader processed, market time advances a little bit.
    def _timestep_size(self) -> float:
        n_total = (self.trader_stats.get('n_buyers', self.cfg.n_buyers) +
                   self.trader_stats.get('n_sellers', self.cfg.n_sellers))
        return 1.0 / max(n_total, 1)

    # Before placing an order with the RL agent, let the ZIC/ZIP background trader participate in a promotional event.
    def step_background(self) -> LOBSnapshot:
        dt = self._timestep_size()
        self._agent_trades = []

        # Refresh customer orders
        lob_pub = self.exchange.publish_lob(self.time, None, False)
        [self.pending_cust_orders, kills] = customer_orders(
            self.time, self.traders, self.trader_stats,
            self._order_schedule, self.pending_cust_orders, False
        )
        for kill_tid in kills:
            if (kill_tid in self.traders and
                    self.traders[kill_tid].lastquote is not None):
                self.exchange.del_order(
                    self.time, self.traders[kill_tid].lastquote, None, False
                )

        # Each background trader submits an order
        tids = list(self.traders.keys())
        random.shuffle(tids)
        for tid in tids:
            trader = self.traders[tid]
            lob_pub = self.exchange.publish_lob(self.time, None, False)
            time_left = max(0.0, (self.cfg.end_time - self.time) /
                            (self.cfg.end_time - self.cfg.start_time))
            order = trader.getorder(self.time, time_left, lob_pub)
            if order is not None:
                trader.n_quotes = 1
                trade = self.exchange.process_order(self.time, order, None, False)
                if trade is not None:
                    self._record_trade(trade)
                    if (trade['party1'] == RL_AGENT_TID or trade['party2'] == RL_AGENT_TID) and self.agent_last_order is not None:
                        self._agent_trades.append((trade, self.agent_last_order.otype))
                        self.agent_last_order = None
                    for party in ('party1', 'party2'):
                        pid = trade[party]
                        if pid in self.traders:
                            self.traders[pid].bookkeep(
                                self.time, trade, order, False
                            )
                # all traders respond (ZIP updates learning)
                lob_pub = self.exchange.publish_lob(self.time, None, False)
                for t in self.traders:
                    self.traders[t].respond(self.time, lob_pub, trade, False)

            self.time += dt

        return self.publish_snapshot()

    # Convert the RL action into a BSE order and send it to the real order book for matching.
    def submit_agent_order(
        self,
        otype: str,           # 'Bid' or 'Ask'
        price: int,           # limit price (integer, BSE convention)
        qty: int = 1,
    ) -> Optional[dict]:
        # Clip price to BSE system limits
        price = int(max(BSE_SYS_MIN_PRICE, min(BSE_SYS_MAX_PRICE, price)))

        order = Order(
            tid=RL_AGENT_TID,
            otype=otype,
            price=price,
            qty=qty,
            time=self.time,
            qid=self._agent_qid,
        )
        self._agent_qid += 1

        trade = self.exchange.process_order(self.time, order, None, False)
        if trade is not None:
            self.agent_last_order = None
            self._record_trade(trade)
            # notify the background trader that was the counterparty
            for party in ('party1', 'party2'):
                pid = trade[party]
                if pid in self.traders:
                    self.traders[pid].bookkeep(
                        self.time, trade, order, False
                    )
            # let background traders respond
            lob_pub = self.exchange.publish_lob(self.time, None, False)
            for t in self.traders:
                self.traders[t].respond(self.time, lob_pub, trade, False)
        else:
            self.agent_last_order = order
        return trade

    # The RL agent had previously placed a limit order, which was later hit by a background trader.
    def consume_agent_trades(self) -> List[Tuple[dict, str]]:
        trades = list(self._agent_trades)
        self._agent_trades = []
        return trades

    # Cancel the previous orders placed by the RL agent.
    def cancel_agent_order(self) -> None:
        if self.agent_last_order is not None:
            self.exchange.del_order(
                self.time, self.agent_last_order, None, False
            )
            self.agent_last_order = None

    # Generate market states observed by RL
    def publish_snapshot(self) -> LOBSnapshot:
        pub = self.exchange.publish_lob(self.time, None, False)
        depth = self.cfg.lob_depth

        # BSE bids sorted ascending (worst→best), we want best-first → reverse
        raw_bids = pub['bids']['lob']          # list of [price, qty]
        raw_asks = pub['asks']['lob']          # list of [price, qty] ascending

        bid_levels = [(p, q) for p, q in reversed(raw_bids)][:depth]
        ask_levels = [(p, q) for p, q in raw_asks][:depth]

        # last trade price from tape
        last_trade = None
        for record in reversed(pub['tape']):
            if record['type'] == 'Trade':
                last_trade = record['price']
                break

        return LOBSnapshot(
            time=self.time,
            best_bid=pub['bids']['best'],
            best_ask=pub['asks']['best'],
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            last_trade_price=last_trade,
            tape=pub['tape'],
        )

    # Save transaction price
    def _record_trade(self, trade: dict) -> None:
        self.price_history.append(trade['price'])

    # Short-term volatility
    def short_term_volatility(self, window: int = 10) -> float:
        prices = self.price_history[-window:]
        if len(prices) < 2:
            return 0.0
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / (len(prices) - 1)
        return math.sqrt(var)

    @property
    def is_done(self) -> bool:
        return self.time >= self.cfg.end_time

    # Return remaining time percentage
    @property
    def time_remaining_fraction(self) -> float:
        duration = self.cfg.end_time - self.cfg.start_time
        return max(0.0, (self.cfg.end_time - self.time) / duration)
