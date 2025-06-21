#!/usr/bin/env python

import numpy as np
import pandas as pd
from main import getMyPosition as getPosition

commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df.values.T

def main(num_days: int = 200, prices_file: str = "prices.txt", verbose: bool = True):
    prcAll = loadPrices(prices_file)
    nInst, nt = prcAll.shape
    print(f"Loaded {nInst} instruments for {nt} days")
    meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, num_days, verbose)
    score = meanpl - 0.1 * plstd
    print("=====")
    print("mean(PL): %.1lf" % meanpl)
    print("return: %.5lf" % ret)
    print("StdDev(PL): %.2lf" % plstd)
    print("annSharpe(PL): %.2lf " % sharpe)
    print("totDvolume: %.0lf " % dvol)
    print("Score: %.2lf" % score)
    return score

def calcPL(prcHist, numTestDays, verbose=True):
    cash = 0.0
    nInst, nt = prcHist.shape
    curPos = np.zeros(nInst)
    totDVolume = 0.0
    value = 0.0
    todayPLL = []
    startDay = nt + 1 - numTestDays
    for t in range(startDay, nt + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = value / totDVolume if totDVolume > 0 else 0.0
        if t > startDay:
            if verbose:
                print(
                    "Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf"
                    % (t, value, todayPL, totDVolume, ret)
                )
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    plmu, plstd = np.mean(pll), np.std(pll)
    annSharpe = np.sqrt(249) * plmu / plstd if plstd > 0 else 0.0
    return plmu, ret, plstd, annSharpe, totDVolume



if __name__ == "__main__":
    main()
