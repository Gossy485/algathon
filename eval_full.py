#!/usr/bin/env python
"""
eval_full.py

A robust evaluation suite focusing on key metrics:
1. Expanding walk-forward analysis for consistency.
2. Noise test to check for sensitivity to minor data changes.
3. Label-shuffle reality check to validate the strategic logic.
4. Includes the primary plot of fold scores for visual analysis.

"""

import importlib, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────── constants ─────────
PRICE_FILE = Path("prices.txt")
MODULE     = "main"

CAP_USD      = 10_000           # position cap
BASE_COMM    = 5e-4             # 5 bp
TRAIN_MIN    = 250
TEST_LEN     = 50
NOISE_PCT    = 0.5              # ±0.5 % on daily returns

# ───────── utils ──────────
def load_prices(p): return pd.read_csv(p, sep=r"\s+", header=None).values.T

def score(pl): return pl.mean() - 0.1 * pl.std(ddof=0)

def make_factory(mod):
    if hasattr(mod, "Strategy"):
        return lambda: mod.Strategy()
    f = mod.getMyPosition
    class _W:
        def fit(self, *_): pass
        def trade(self,h): return f(h)
    return lambda: _W()

def add_noise(prc, pct, rng):
    ret   = prc[:,1:] / prc[:, :-1] - 1.0
    noisy = prc[:, :-1] * (1.0 + ret*(1.0 + rng.normal(0, pct/100, ret.shape)))
    return np.concatenate([prc[:, :1], noisy], axis=1)

def circ_shift(arr, rng):
    out = arr.copy()
    for i in range(arr.shape[0]):
        out[i] = np.roll(out[i], rng.integers(arr.shape[1]))
    return out

# ───────── walk-forward ─────────
def walk(prc, fac, comm=BASE_COMM, slip_bp=0.0, expanding=True):
    n, T = prc.shape
    folds, mdds = [], []
    start = 0
    while start + TRAIN_MIN + TEST_LEN <= T:
        s_train = 0 if expanding else start
        e_train = start + TRAIN_MIN
        s_test  = e_train
        e_test  = s_test + TEST_LEN

        st = fac(); st.fit(prc[:, s_train:e_train])

        cash, pos, val = 0.0, np.zeros(n), 0.0
        pl = []

        for t in range(s_test, e_test):
            px = prc[:, t]
            if t < e_test-1:
                tgt = np.asarray(st.trade(prc[:, :t+1]), int)
                cap = np.floor(CAP_USD / px).astype(int)
                tgt = np.clip(tgt, -cap, cap)

                delta = tgt - pos
                traded = np.abs(delta) * px
                cash -= delta @ (px*(1+np.sign(delta)*slip_bp*1e-4))
                cash -= traded.sum() * comm
                pos   = tgt

            new_val = cash + pos @ px
            pl.append(new_val - val); val = new_val

        pl = np.array(pl[1:])
        eq = np.cumsum(pl)
        dd = eq - np.maximum.accumulate(eq)

        folds.append(score(pl))
        mdds.append(float(dd.min()))
        start += TEST_LEN
    return np.array(folds), mdds

# ───────── evaluation suite ─────────
def evaluate():
    prc = load_prices(PRICE_FILE)
    mod = importlib.import_module(MODULE)
    fac = make_factory(mod)
    rng = np.random.default_rng(42)

    # Base expanding walk
    folds, mdds = walk(prc, fac)
    wts = np.arange(1, len(folds) + 1)
    w_mean = (folds * wts).sum() / wts.sum()
    worst_fold = folds.min()

    # Last 50-day block test
    last_block_score, _ = walk(prc[:, -(TRAIN_MIN + TEST_LEN):], fac, expanding=False)
    last_block = last_block_score[-1]

    # Noise test
    noise_vals = [walk(add_noise(prc, NOISE_PCT, rng), fac)[0].mean() for _ in range(10)] # Reduced runs for speed
    noise_mean = float(np.mean(noise_vals))

    # Label-shuffle reality check
    shuffle_vals = [walk(circ_shift(prc, rng), fac)[0].mean() for _ in range(20)] # Reduced runs for speed
    shuffle_mean = float(np.mean(shuffle_vals))


    # ───── print table ─────
    print("\nfold scores        :", np.round(folds,2))
    print("time-weighted mean :", f"{w_mean:+.2f}")
    print("worst single fold  :", f"{worst_fold:+.2f}")
    print("last unseen block  :", f"{last_block:+.2f}")
    print(f"\n±{NOISE_PCT}% noise mean    : {noise_mean:+.2f}")
    print(f"shuffle mean       : {shuffle_mean:+.2f}")

    # ───── robustness gates ─────
    fail_reasons = []
    if worst_fold < 0:        fail_reasons.append("negative fold")
    if noise_mean < 0:        fail_reasons.append("noise test")
    if shuffle_mean >= w_mean/2 and shuffle_mean > 0: # Check if shuffle score is suspiciously high
        fail_reasons.append("shuffle reality")

    if fail_reasons:
        print(f"\n⚠️  Strategy fails robustness gate: {', '.join(fail_reasons)}")
    else:
        print("\n✅  Strategy passes all key robustness gates!")

    # ───────── visualizations ─────────
    # Plot fold scores
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(folds, label="Fold Scores", marker='o', linestyle='-', color='royalblue', alpha=0.8)
    plt.axhline(y=w_mean, color='green', linestyle='--', label=f"Time-weighted mean ({w_mean:.2f})")
    plt.axhline(y=worst_fold, color='red', linestyle=':', label=f"Worst fold ({worst_fold:.2f})")
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.7)
    plt.xlabel("Fold Number")
    plt.ylabel("Score (Mean PL - 0.1 * Stdev PL)")
    plt.title("Walk-Forward Fold Scores", fontsize=16)
    plt.legend()
    plt.show()


# ───────── main ─────────
if __name__ == "__main__":
    t0 = time.time()
    evaluate()
    print(f"\n⟡  completed in {time.time()-t0:.1f}s")