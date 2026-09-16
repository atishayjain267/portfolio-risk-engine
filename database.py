import pandas as pd
from sqlalchemy import create_engine, inspect

# Initialize SQLite database engine
engine = create_engine("sqlite:///portfolio.db")


def get_available_tickers() -> list:
    """Queries SQLite and returns a list of unique tickers stored in the cache."""
    inspector = inspect(engine)

    # If the table hasn't been created yet, return an empty list
    if not inspector.has_table("asset_prices"):
        return []

    query = "SELECT DISTINCT ticker FROM asset_prices"
    try:
        df = pd.read_sql(query, con=engine)
        return df["ticker"].dropna().tolist()
    except Exception:
        return []