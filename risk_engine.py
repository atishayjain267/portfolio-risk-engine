import numpy as np
import pandas as pd
from database import engine

def load_returns(db_engine=engine) -> pd.DataFrame:
    """Queries SQLite asset_returns table and returns a clean daily returns DataFrame."""
    query = "SELECT * FROM asset_returns"
    with db_engine.connect() as conn:
        df = pd.read_sql(query, con=conn)
    
    if "Date" in df.columns:
        df = df.set_index("Date")
    
    return df.dropna()

def analyze_portfolio(
    returns_df: pd.DataFrame, 
    risk_free_rate: float = 0.065, 
    num_simulations: int = 3000,
    investor_profile: str = "Balanced (Growth + Hedge)"
) -> dict:
    """
    Monte Carlo simulation with profile-specific allocation constraints:
    - Conservative: 30-50% Gold, lower equities (5-20%)
    - Balanced: 10-35% across all assets
    - Aggressive: 5-15% Gold, higher equities (15-45%)
    """
    mean_daily_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)
    assets = list(returns_df.columns)

    # Set asset-level bounds depending on investor profile
    bounds = {}
    for asset in assets:
        if "Conservative" in investor_profile:
            bounds[asset] = (0.30, 0.50) if "GOLDBEES" in asset else (0.05, 0.20)
        elif "Aggressive" in investor_profile:
            bounds[asset] = (0.05, 0.15) if "GOLDBEES" in asset else (0.15, 0.45)
        else:  # Balanced
            bounds[asset] = (0.10, 0.35)

    lows = np.array([bounds[a][0] for a in assets])
    highs = np.array([bounds[a][1] for a in assets])

    results = np.zeros((3, num_simulations))
    weights_record = []
    np.random.seed(42)

    for i in range(num_simulations):
        raw = np.random.uniform(lows, highs)
        w = raw / np.sum(raw)
        weights_record.append(w)

        port_return = np.sum(mean_daily_returns * w) * 252
        port_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
        sharpe_ratio = (port_return - risk_free_rate) / (port_volatility + 1e-8)

        results[0, i] = port_volatility
        results[1, i] = port_return
        results[2, i] = sharpe_ratio

    best_idx = int(np.argmax(results[2]))
    best_weights = weights_record[best_idx]
    optimal_weights = dict(zip(returns_df.columns, np.round(best_weights, 4)))

    return {
        "optimal_weights": optimal_weights,
        "max_sharpe": round(float(results[2, best_idx]), 2),
        "expected_annual_return": round(float(results[1, best_idx]) * 100, 2),
        "annual_volatility_risk": round(float(results[0, best_idx]) * 100, 2)
    }