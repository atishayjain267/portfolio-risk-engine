import yfinance as yf
import pandas as pd
from database import engine

DEFAULT_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "AAPL"]

def fetch_data(tickers: list = None, period: str = "2y") -> pd.DataFrame:
    """Extracts adjusted close prices from Yahoo Finance."""
    if tickers is None:
        tickers = DEFAULT_TICKERS
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data
    return prices.dropna()

def process_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw prices into daily log/percentage returns."""
    returns = prices.pct_change().dropna()
    return returns

def load_to_db(returns_df: pd.DataFrame):
    """Loads transformed returns into SQLite database."""
    returns_df.to_sql("asset_returns", con=engine, if_exists="replace", index=True)

def run_pipeline(tickers: list = None) -> int:
    """Executes the complete ETL sequence."""
    prices = fetch_data(tickers)
    returns = process_data(prices)
    load_to_db(returns)
    return len(returns)

if __name__ == "__main__":
    count = run_pipeline()
    print(f"ETL Pipeline successfully processed {count} records.")