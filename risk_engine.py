import numpy as np
import pandas as pd
from scipy.optimize import minimize

def load_returns(engine) -> pd.DataFrame:
    """Loads transformed asset returns table from the database."""
    query = "SELECT * FROM asset_returns"
    df = pd.read_sql(query, con=engine)
    
    # Handle Date index if stored as a column
    date_cols = [c for c in df.columns if "date" in c.lower() or "index" in c.lower()]
    if date_cols:
        df = df.set_index(date_cols[0])
    
    # Keep only numeric return columns
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.dropna()

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.065):
    """Calculates annualized expected return, volatility, and Sharpe ratio."""
    port_return = np.sum(mean_returns * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility != 0 else 0
    return port_return, port_volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.065):
    """Objective function to minimize (negative Sharpe ratio)."""
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def analyze_portfolio(returns_df: pd.DataFrame, risk_free_rate: float = 0.065) -> dict:
    """
    Executes Markowitz Mean-Variance Optimization and computes risk statistics.
    """
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)
    
    # Boundary conditions: weights sum to 1, no short selling (0 <= w <= 1)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]
    
    # Numerical optimization
    opt_results = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = opt_results.x if opt_results.success else initial_weights
    exp_return, volatility, sharpe = portfolio_performance(
        optimal_weights, mean_returns, cov_matrix, risk_free_rate
    )
    
    # Calculate historical 1-day 95% Value at Risk (VaR)
    portfolio_daily_returns = (returns_df * optimal_weights).sum(axis=1)
    var_95 = np.percentile(portfolio_daily_returns, 5)
    
    # Calculate Maximum Drawdown
    cumulative = (1 + portfolio_daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    weight_dict = {
        col: round(float(w), 4)
        for col, w in zip(returns_df.columns, optimal_weights)
    }
    
    return {
        "expected_annual_return": round(float(exp_return) * 100, 2),
        "annual_volatility_risk": round(float(volatility) * 100, 2),
        "max_sharpe": round(float(sharpe), 2),
        "var_95_daily_pct": round(float(var_95) * 100, 2),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
        "optimal_weights": weight_dict,
        "weights": optimal_weights
    }