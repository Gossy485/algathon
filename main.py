#!/usr/bin/env python

"""Mean reversion strategy using z-score of price relative to a moving average.
The function getMyPosition is called every day with the price history. We trade
only when we have at least MA_WINDOW days of data and at most once every
REBALANCE_DAYS. Position sizes are scaled inversely with historical volatility
and capped by dollar limits.
"""

import numpy as np

# Strategy parameters
MA_WINDOW = 20          # lookback for mean and volatility
REBALANCE_DAYS = 5      # how often to rebalance
Z_THRESHOLD = 2.0       # enter trades when |z| > threshold
RISK_DOLLARS = 300      # risk budget per leg
MAX_LEG_DOLLARS = 10000

# Internal state
_last_rebalance = 0
_positions: np.ndarray | None = None


def getMyPosition(price_history: np.ndarray):
    """Return target positions for each instrument."""
    global _last_rebalance, _positions

    prices = np.asarray(price_history, dtype=float)
    if prices.shape[0] != 50:
        prices = prices.T

    n_inst, n_days = prices.shape
    today = n_days - 1

    if _positions is None:
        _positions = np.zeros(n_inst, dtype=int)

    # insufficient history
    if n_days <= MA_WINDOW:
        return _positions.tolist()

    # only rebalance every REBALANCE_DAYS
    if today - _last_rebalance < REBALANCE_DAYS:
        return _positions.tolist()

    window = prices[:, today - MA_WINDOW + 1 : today + 1]
    ma = window.mean(axis=1)
    vol = window.std(axis=1)
    # avoid zero volatility
    fallback = np.median(vol[vol > 0])
    vol[vol == 0] = fallback if fallback > 0 else 1.0

    z = (prices[:, today] - ma) / vol

    new_pos = np.zeros(n_inst, dtype=int)
    for i in range(n_inst):
        if z[i] < -Z_THRESHOLD:
            size = int(RISK_DOLLARS / (vol[i] * prices[i, today]))
            size = max(1, min(size, int(MAX_LEG_DOLLARS / prices[i, today])))
            new_pos[i] = size
        elif z[i] > Z_THRESHOLD:
            size = int(RISK_DOLLARS / (vol[i] * prices[i, today]))
            size = max(1, min(size, int(MAX_LEG_DOLLARS / prices[i, today])))
            new_pos[i] = -size

    _positions = new_pos
    _last_rebalance = today
    return new_pos.tolist()
