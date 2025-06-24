#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from pathlib import Path
import main

# ───────── constants ─────────
PRICE_FILE      = Path("prices.txt")  # or prices.txt for synthetic
COMM_RATE       = 0.0005    # commission rate (5 bps)
POS_LIMIT_USD   = 10000
TRAIN_DAYS      = 250
TEST_DAYS       = 50
USE_DAYS        = 700      # only use first 700 days for evaluation

# noise test parameters
NOISE_PCT       = 0.001   # 0.1% noise
RANDOM_SEED     = 42

# ───────── data loader ─────────
def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None)
    prc = df.values.T  # shape: (nInst, nt)
    nInst, nt = prc.shape
    print(f"Loaded {nInst} instruments × {nt} total days")
    if nt < USE_DAYS:
        raise ValueError(f"Need at least {USE_DAYS} days, but got {nt}")
    return prc

# ───────── P/L calculator (with optional noise) ─────────
def calcPL(prcHist: np.ndarray,
           numTestDays: int,
           noise_pct: float = 0.0,
           seed: int | None = None):
    """
    noise_pct > 0 only affects when explicitly passed; defaults to 0 for all other tests.
    """
    rng = np.random.RandomState(seed) if noise_pct and seed is not None else None
    prc_exec = prcHist
    nInst, nt = prc_exec.shape

    cash = 0.0
    curPos = np.zeros(nInst, dtype=int)
    totDVolume = 0.0
    value = 0.0
    dailyPL = []
    start_day = nt - numTestDays + 1

    for t in range(start_day, nt + 1):
        price_ex = prc_exec[:, t-1]
        hist_true = prc_exec[:, :t]

        # simulate noisy fill price only if noise_pct>0
        if noise_pct and rng is not None:
            noise = noise_pct * rng.randn(nInst)
            price_sig = price_ex * (1.0 + noise)
        else:
            price_sig = price_ex

        # build history passed to strategy
        hist_sig = hist_true.copy()
        hist_sig[:, -1] = price_sig

        if t < nt:
            # decision based on potentially noisy last price
            tgt = np.asarray(main.getMyPosition(hist_sig), int)
            cap = np.floor(POS_LIMIT_USD / price_ex).astype(int)
            tgt = np.clip(tgt, -cap, cap)

            delta = tgt - curPos
            traded = np.abs(delta) * price_ex
            totDVolume += traded.sum()
            cash -= price_ex.dot(delta) + COMM_RATE * traded.sum()
            curPos = tgt.copy()

        # mark-to-market
        posValue = curPos.dot(price_ex)
        todayPL = cash + posValue - value
        value = cash + posValue

        if t > start_day:
            dailyPL.append(todayPL)

    pll = np.array(dailyPL)
    mu = pll.mean()
    sigma = pll.std(ddof=0)
    sharpe = np.sqrt(249) * mu / sigma if sigma > 0 else 0.0
    ret = value / totDVolume if totDVolume > 0 else 0.0
    return mu, ret, sigma, sharpe, totDVolume, pll

# ───────── evaluation functions ─────────
def walkForward(prc):
    """walkForward with zero noise"""
    _, nt = prc.shape
    scores = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS <= nt:
        importlib.reload(main)
        block = prc[:, : start + TRAIN_DAYS + TEST_DAYS]
        # explicitly no noise for CV folds
        m, r, s, sh, dv, _ = calcPL(block, TEST_DAYS, noise_pct=0.0)
        scores.append(m - 0.1 * s)
        start += TEST_DAYS
    return np.array(scores)


def shuffleTest(prc, repeats=20):
    """shuffleTest with zero noise"""
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(repeats):
        sh = prc.copy()
        for i in range(sh.shape[0]):
            sh[i] = np.roll(sh[i], rng.integers(sh.shape[1]))
        importlib.reload(main)
        # explicitly no noise for shuffle test
        vals.append(walkForward(sh).mean())
    return float(np.mean(vals))

# ───────── main script ─────────
if __name__ == "__main__":
    prcAll = loadPrices(PRICE_FILE)
    prcTrain = prcAll[:, :USE_DAYS]
    print(f"Evaluating on first {USE_DAYS} days")

    # 1) Walk-forward (no noise)
    wf_scores = walkForward(prcTrain)
    #wts = np.arange(1, len(wf_scores)+1)
    #tw_mean = wf_scores.mean()#(wf_scores * wts).sum() / wts.sum()
    λ = 0.1
    exps = np.exp(λ * np.arange(len(wf_scores))) 
    tw_mean = (wf_scores * exps).sum() / exps.sum()

    # 2) Shuffle (no noise)
    shuffle_mean = shuffleTest(prcTrain)

    # 3) Noise test (only here)
    importlib.reload(main)
    mu_n, ret_n, sigma_n, sharpe_n, dvol_n, pll_n = calcPL(
        prcTrain,
        TEST_DAYS,
        noise_pct=NOISE_PCT,
        seed=RANDOM_SEED
    )
    score_n = mu_n - 0.1 * sigma_n

    # ───────── summary display ─────────
    print("========== Evaluation Summary (first 700 days) ==========")
    print(f"Walk-forward folds : {wf_scores.round(2)}")
    print(f"Time-weighted mean : {tw_mean:.2f}\n")
    print("---- Robustness Test ----")
    print(f"Shuffle test         : {shuffle_mean:+.2f}")
    print(f"Noise test (pct={NOISE_PCT:.4f}) : score {score_n:.2f}, meanPL {mu_n:.2f}, σ {sigma_n:.2f}, Sharpe {sharpe_n:.2f}\n")
    print("========================================================")
