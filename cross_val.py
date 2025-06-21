#!/usr/bin/env python
"""Run rolling backtests over the price history to gauge stability."""

from eval import loadPrices, calcPL


def rolling_backtest(prices, window=200, step=150):
    results = []
    for start in range(0, prices.shape[1] - window + 1, step):
        seg = prices[:, start : start + window]
        print(f"\nSegment {start}-{start+window-1}")
        metrics = calcPL(seg, window, verbose=False)
        results.append(metrics)
        meanpl, ret, plstd, sharpe, dvol = metrics
        score = meanpl - 0.1 * plstd
        print(
            f"meanPL={meanpl:.2f} ret={ret:.5f} std={plstd:.2f} sharpe={sharpe:.2f} $-traded={dvol:.0f} score={score:.2f}"
        )
    return results


if __name__ == "__main__":
    prices = loadPrices("prices.txt")
    rolling_backtest(prices)
