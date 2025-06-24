#!/usr/bin/env python
"""Index momentum with cross-sectional tilt."""
import numpy as np

LOOKBACK = 10
THRESH = 0.002
INDEX_DOLLARS = 744
CS_DOLLARS = 30
CROSS_COUNT = 1


def getMyPosition(price_history: np.ndarray) -> list[int]:
    prices = np.asarray(price_history, dtype=float)
    if prices.shape[0] != 50:
        prices = prices.T
    n_inst, n_days = prices.shape
    if n_days <= LOOKBACK:
        return [0] * n_inst

    index = prices.mean(axis=0)
    mom = index[-1] / index[-LOOKBACK - 1] - 1.0
    if abs(mom) < THRESH:
        return [0] * n_inst

    direction = 1 if mom > 0 else -1
    inst_mom = prices[:, -1] / prices[:, -LOOKBACK - 1] - 1.0
    ranks = np.argsort(inst_mom)
    longs = ranks[-CROSS_COUNT:]
    shorts = ranks[:CROSS_COUNT]

    price_today = prices[:, -1]
    base = np.floor(INDEX_DOLLARS / price_today).astype(int)
    tilt = np.floor(CS_DOLLARS / price_today).astype(int)

    pos = np.full(n_inst, direction) * base
    if direction > 0:
        pos[longs] += tilt[longs]
        pos[shorts] -= tilt[shorts]
    else:
        pos[longs] -= tilt[longs]
        pos[shorts] += tilt[shorts]

    return pos.tolist()
