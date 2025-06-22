# ─────────────── main.py ───────────────

from __future__ import annotations
import numpy as np

# ── strategy knobs ────────────────────────────
_WINDOW_SMA      = 13          # look-back for mean-reversion signal
_REBAL_EVERY     = 5           # rebalance cadence (days)
_K_LONG_SHORT    = 6           # number of longs and shorts
_GAP_THRESH      = 0.018       # 1.8 % from SMA before we act
_VOL_WIN         = 20          # σ horizon for risk sizing
_SIGMA_RISK_USD  = 2_000.0     # \$ P&L budget at ±1 σ
_LEG_CAP_USD     = 9_900.0     # ≤ \$10 000 per competition rules
# ──────────────────────────────────────────────

_last_reb_day: int | float = -10**9
_pos:           np.ndarray | None = None


def _as_matrix(p: np.ndarray) -> np.ndarray:
    """Ensure the data are (50 × nDays), transposing if needed."""
    return p if p.shape[0] == 50 else p.T


def getMyPosition(prcSoFar) -> list[int]:
    """
    Parameters
    ----------
    prcSoFar : np.ndarray
        Price history *up to today* inclusive.  Shape (50 × nDays) or its
        transpose.

    Returns
    -------
    list[int] – desired share positions for all 50 instruments.
    """
    global _last_reb_day, _pos

    prc = _as_matrix(np.asarray(prcSoFar, dtype=float))
    n_inst, n_days = prc.shape
    today          = n_days - 1

    # one-off initialisation
    if _pos is None:
        _pos = np.zeros(n_inst, dtype=int)

    # need enough history for both SMA and σ
    if n_days <= max(_WINDOW_SMA, _VOL_WIN):
        return _pos.tolist()

    # honour the rebalance cadence
    if (today - _last_reb_day) < _REBAL_EVERY:
        return _pos.tolist()

    # ── build the mean-reversion signal ───────────────────────────────
    sma  = prc[:, today - _WINDOW_SMA + 1 : today + 1].mean(axis=1)
    gap  = prc[:, today] / sma - 1.0

    order   = np.argsort(gap)
    longs   = [i for i in order[:_K_LONG_SHORT]  if gap[i] <= -_GAP_THRESH]
    shorts  = [i for i in order[-_K_LONG_SHORT:] if gap[i] >=  _GAP_THRESH]

    new_pos = np.zeros(n_inst, dtype=int)

    # proceed only if we have names on *both* sides (keeps book neutral)
    if longs and shorts:
        # 20-day realised σ for risk-parity sizing
        ret20 = prc[:, today - _VOL_WIN + 1 : today + 1] \
                / prc[:, today - _VOL_WIN : today] - 1.0
        sigma = ret20.std(axis=1, ddof=0)
        # guard against zero-vol names
        fallback = np.median(sigma[sigma > 0])
        sigma    = np.where(sigma == 0, fallback, sigma)

        prices = prc[:, today]

        for i in longs:
            sh = int(_SIGMA_RISK_USD / (sigma[i] * prices[i]))
            sh = max(1, min(sh, int(_LEG_CAP_USD / prices[i])))
            new_pos[i] =  sh

        for i in shorts:
            sh = int(_SIGMA_RISK_USD / (sigma[i] * prices[i]))
            sh = max(1, min(sh, int(_LEG_CAP_USD / prices[i])))
            new_pos[i] = -sh

    # persist state & return
    _pos           = new_pos
    _last_reb_day  = today
    return new_pos.tolist()

