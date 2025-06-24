import numpy as np
import pandas as pd
import yfinance as yf

# 1) Read your original to get shape (days × instruments)
orig = np.loadtxt("prices.txt")
ndays, ninstruments = orig.shape
print(f"Original: {ndays} days × {ninstruments} instruments")

# 2) Pick exactly ninstruments tickers
tickers = [
    "AAPL","MSFT","AMZN","GOOGL","BRK-B","NVDA","META","TSLA","JPM","V",
    "UNH","JNJ","PG","MA","BAC","HD","XOM","PFE","KO","PEP",
    "MRK","CSCO","CVX","ADBE","CMCSA","ORCL","T","VZ","ABBV","ABT",
    "CRM","NKE","LLY","MCD","COST","MDT","TXN","ACN","QCOM","UPS",
    "UNP","AMGN","INTC","HON","LIN","NEE","PM","IBM","BMY","LOW"
][:ninstruments]

# 3) Download ample history (>= ndays)
df = yf.download(
    tickers,
    start="2022-06-01",
    end="2025-06-01",
    auto_adjust=True,
)["Close"]

# 4) Drop any tickers with missing data, then pick exactly ninstruments columns
df = df.dropna(axis=1)
df = df.iloc[:, :ninstruments]

# 5) Build array of shape (instruments × fetched_days), then truncate to ndays
arr = df.T.values                    # shape = (ninstruments, fetched_days)
if arr.shape[1] < ndays:
    raise ValueError(f"Fetched only {arr.shape[1]} days, need {ndays}")
arr = arr[:, :ndays]                 # now (ninstruments, ndays)

# 6) Transpose to (days × instruments) = (ndays, ninstruments) and save
out = arr.T                          # shape = (ndays, ninstruments)
np.savetxt("real_prices.txt", out, fmt="%.2f", delimiter=" ")
print(f"Written real_prices.txt with shape {out.shape}")
