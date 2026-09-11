import yfinance as yf
import numpy as np
import pandas as pd

ticker_symbol = "GOLDBEES.NS"
print(f"Analyzing metrics for {ticker_symbol}...")

# 1. Fetch 30 days of data to compute rolling metrics
df = yf.download(ticker_symbol, period="1mo", progress=False)

# 2. Clean MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 3. Calculate daily percentage return: (Today - Yesterday) / Yesterday
df["Daily_Return"] = df["Close"].pct_change()

# 4. Calculate 7-day Rolling Volatility (annualized risk metric)
df["Rolling_Vol"] = df["Daily_Return"].rolling(window=7).std() * np.sqrt(252)

# Reset index so 'Date' becomes a standard column
df.reset_index(inplace=True)

print("\n--- Processed Financial Metrics (Last 5 Days) ---")
print(df[["Date", "Close", "Daily_Return", "Rolling_Vol"]].tail())