#!/usr/bin/env python
"""Index momentum strategy with larger position size."""
import numpy as np

LOOKBACK = 10
THRESH = 0.002
DOLLARS_PER_INST = 1500


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
    price_today = prices[:, -1]
    shares = np.floor(DOLLARS_PER_INST / price_today).astype(int)
    return (direction * shares).tolist()
