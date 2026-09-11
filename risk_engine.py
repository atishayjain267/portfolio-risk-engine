import pandas as pd
import numpy as np
from database import engine

def load_returns():
    """Queries SQLite and pivots returns into a clean time-series matrix."""
    query = "SELECT ticker, trade_date, daily_return FROM asset_prices"
    df = pd.read_sql(query, con=engine)
    
    # Pivot so each column is an asset and rows are trading dates
    returns_matrix = df.pivot(index="trade_date", columns="ticker", values="daily_return").dropna()
    return returns_matrix

def analyze_portfolio(risk_free_rate=0.065, num_simulations=3000):
    """
    Runs a Monte Carlo simulation across random portfolio weightings
    to locate the Markowitz Efficient Frontier and optimal Sharpe Ratio.
    """
    returns = load_returns()
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    results = np.zeros((3, num_simulations))
    weights_record = []

    for i in range(num_simulations):
        # Generate random weights summing to 1.0 (100%)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # 252 trading days per year
        port_return = np.sum(mean_daily_returns * weights) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        results[0, i] = port_volatility
        results[1, i] = port_return
        results[2, i] = sharpe_ratio

    # Locate the portfolio with the highest Sharpe Ratio
    best_idx = np.argmax(results[2])
    best_weights = dict(zip(returns.columns, np.round(weights_record[best_idx], 4)))

    return {
        "correlation_matrix": returns.corr().round(3),
        "optimal_weights": best_weights,
        "max_sharpe": round(results[2, best_idx], 2),
        "expected_annual_return": round(results[1, best_idx] * 100, 2),
        "annual_volatility_risk": round(results[0, best_idx] * 100, 2)
    }

if __name__ == "__main__":
    print("Running Quantitative Risk Engine & Optimization...\n")
    report = analyze_portfolio()

    print("--- Asset Correlation Matrix ---")
    print(report["correlation_matrix"])
    print("\n--- Optimal Portfolio Weights (Max Sharpe) ---")
    for ticker, weight in report["optimal_weights"].items():
        print(f"{ticker}: {weight * 100:.1f}%")
    print(f"\nExpected Return: {report['expected_annual_return']}%")
    print(f"Annual Volatility (Risk): {report['annual_volatility_risk']}%")
    print(f"Sharpe Ratio: {report['max_sharpe']}")