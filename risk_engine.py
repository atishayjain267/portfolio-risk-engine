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
    Runs Monte Carlo simulation across random weightings to locate the optimal Sharpe ratio,
    then evaluates downside tail risk (Historical/Parametric VaR and Maximum Drawdown).
    """
    returns = load_returns()
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    results = np.zeros((3, num_simulations))
    weights_record = []

    for i in range(num_simulations):
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

    # Optimal Sharpe allocation
    best_idx = np.argmax(results[2])
    optimal_weights_array = weights_record[best_idx]
    best_weights = dict(zip(returns.columns, np.round(optimal_weights_array, 4)))

    # --- Downside Risk Analytics on Optimal Portfolio ---
    # Construct historical daily return series of the optimal strategy
    opt_portfolio_daily_returns = returns.dot(optimal_weights_array)

    # 1. Value at Risk (VaR 95% 1-Day - Historical Percentile)
    var_95_daily = -np.percentile(opt_portfolio_daily_returns, 5)

    # 2. Maximum Drawdown (MDD) & Cumulative Equity Curve
    cum_wealth = (1 + opt_portfolio_daily_returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdown_series = (cum_wealth - running_max) / running_max
    max_drawdown = drawdown_series.min()

    return {
        "correlation_matrix": returns.corr().round(3),
        "optimal_weights": best_weights,
        "max_sharpe": round(results[2, best_idx], 2),
        "expected_annual_return": round(results[1, best_idx] * 100, 2),
        "annual_volatility_risk": round(results[0, best_idx] * 100, 2),
        "var_95_daily_pct": round(var_95_daily * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "drawdown_series": drawdown_series
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
    print(f"1-Day 95% VaR: {report['var_95_daily_pct']}%")
    print(f"Historical Max Drawdown: {report['max_drawdown_pct']}%")