import numpy as np

_IDX = 4                # instrument to short (0-based index)
_CAP_USD = 10_000       # capital per instrument


def _matrix(p: np.ndarray) -> np.ndarray:
    """Ensure shape (n_inst, T)."""
    p = np.asarray(p, dtype=float)
    return p if p.shape[0] == 50 else p.T


class Strategy:
    def fit(self, *_):
        pass

    def trade(self, hist: np.ndarray) -> list[int]:
        prc = _matrix(hist)
        px = prc[:, -1]
        cap = np.floor(_CAP_USD / px).astype(int)
        pos = np.zeros(prc.shape[0], dtype=int)
        pos[_IDX] = -cap[_IDX]
        return pos.tolist()


def getMyPosition(prcSoFar) -> list[int]:
    global _default
    try:
        _default
    except NameError:
        _default = Strategy()
    return _default.trade(prcSoFar)

