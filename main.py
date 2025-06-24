#!/usr/bin/env python
"""Index momentum with cross-sectional tilt."""
import numpy as np

LOOKBACK = 10
THRESH = 0.001
INDEX_DOLLARS = 2000
CS_DOLLARS = 40
CROSS_COUNT = 1
HOLD_DAYS = 20
SCALE = 0.005  # scale factor for momentum strength

_last_dir = 0
_last_pos: np.ndarray | None = None
_last_day = -1


def getMyPosition(price_history: np.ndarray) -> list[int]:
    global _last_dir, _last_pos, _last_day

    prices = np.asarray(price_history, dtype=float)
    if prices.shape[0] != 50:
        prices = prices.T
    n, t = prices.shape
    today = t - 1

    if _last_pos is None:
        _last_pos = np.zeros(n, dtype=int)

    if t <= LOOKBACK:
        return _last_pos.tolist()

    index = prices.mean(axis=0)
    mom = index[-1] / index[-LOOKBACK - 1] - 1
    if abs(mom) <= THRESH:
        strength = 0.0
    else:
        strength = min(1.0, (abs(mom) - THRESH) / SCALE)
    direction = 0
    if strength > 0:
        direction = 1 if mom > 0 else -1

    if today - _last_day < HOLD_DAYS and direction == _last_dir:
        return _last_pos.tolist()

    if direction == 0:
        _last_pos = np.zeros(n, dtype=int)
        _last_dir = 0
        _last_day = today
        return _last_pos.tolist()

    inst_mom = prices[:, -1] / prices[:, -LOOKBACK - 1] - 1
    ranks = np.argsort(inst_mom)
    longs = ranks[-CROSS_COUNT:]
    shorts = ranks[:CROSS_COUNT]
    price_today = prices[:, -1]
    base = np.floor(INDEX_DOLLARS / price_today).astype(int)
    tilt = np.floor(CS_DOLLARS / price_today).astype(int)
    pos = np.full(n, direction) * (base * strength).astype(int)
    if direction > 0:
        pos[longs] += (tilt * strength).astype(int)[longs]
        pos[shorts] -= (tilt * strength).astype(int)[shorts]
    else:
        pos[longs] -= (tilt * strength).astype(int)[longs]
        pos[shorts] += (tilt * strength).astype(int)[shorts]

    _last_pos = pos.astype(int)
    _last_dir = direction
    _last_day = today
    return _last_pos.tolist()
