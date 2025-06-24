#!/usr/bin/env python
import numpy as np

DOLLAR_LIMIT = 10000
SHORT_LOOK = 3
LONG_LOOK = 20
_last_pos = None


def getMyPosition(price_history: np.ndarray):
    global _last_pos
    prices = np.asarray(price_history, dtype=float)
    if prices.shape[0] != 50:
        prices = prices.T
    n_inst, n_days = prices.shape
    today = n_days - 1
    if _last_pos is None:
        _last_pos = np.zeros(n_inst, dtype=int)
    if n_days <= LONG_LOOK:
        return _last_pos.tolist()
    ma_short = prices[:, today-SHORT_LOOK:today].mean()
    ma_long = prices[:, today-LONG_LOOK:today].mean()
    sign = 1 if ma_short > ma_long else -1
    tgt = sign * np.floor(DOLLAR_LIMIT / prices[:, today]).astype(int)
    _last_pos = tgt
    return tgt.tolist()
