import yfinance as yf
import numpy as np
import pandas as pd
from database import engine, init_db

# Assets to track: Benchmark index, a Gold ETF, and two large-cap equities
WATCHLIST = ["^NSEI", "GOLDBEES.NS", "TCS.NS", "HDFCBANK.NS"]

def fetch_and_process(ticker: str) -> pd.DataFrame:
    """Extracts 6 months of data and computes financial features."""
    print(f"Extracting & processing: {ticker}")
    df = yf.download(ticker, period="6mo", progress=False)
    
    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df["ticker"] = ticker
    df["trade_date"] = df.index.date
    df["close_price"] = df["Close"]

    # Transformation: Log returns and 30-day annualized volatility
    df["daily_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    df["rolling_vol"] = df["daily_return"].rolling(window=30).std() * np.sqrt(252)

    # Select only the columns matching our database table
    clean_df = df[["ticker", "trade_date", "close_price", "daily_return", "rolling_vol"]].dropna(subset=["daily_return"])
    return clean_df

def run_pipeline():
    init_db()  # Ensure tables exist
    
    all_data = []
    for ticker in WATCHLIST:
        df_processed = fetch_and_process(ticker)
        if not df_processed.empty:
            all_data.append(df_processed)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Load: write straight to the SQLite table
        final_df.to_sql("asset_prices", con=engine, if_exists="replace", index=False)
        print(f"\nETL complete: Successfully loaded {len(final_df)} records into 'asset_prices' table.")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    run_pipeline()