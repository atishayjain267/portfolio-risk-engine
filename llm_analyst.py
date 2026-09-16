from google import genai


def generate_risk_briefing(report: dict, api_key: str) -> str:
    """Generates an executive risk committee briefing using Gemini."""
    client = genai.Client(api_key=api_key)

    exp_return = report.get(
        "expected_annual_return",
        round(report.get("expected_return", 0) * 100, 2),
    )
    volatility = report.get(
        "annual_volatility_risk", round(report.get("volatility", 0) * 100, 2)
    )
    sharpe = report.get("max_sharpe", round(report.get("sharpe_ratio", 0), 2))
    var_95 = report.get(
        "var_95_daily_pct", round(report.get("var_95", 0) * 100, 2)
    )
    mdd = report.get(
        "max_drawdown_pct", round(report.get("max_drawdown", 0) * 100, 2)
    )
    weights = report.get("optimal_weights", {})

    prompt = f"""
    You are a Senior Quantitative Risk Officer at an investment fund.
    Analyze the following quantitative portfolio report and provide an executive committee briefing:

    - Expected Annualized Return: {exp_return}%
    - Annual Volatility (Risk): {volatility}%
    - Maximum Sharpe Ratio: {sharpe}
    - 1-Day 95% Value at Risk (VaR): {var_95}%
    - Historical Maximum Drawdown: {mdd}%
    - Optimal Asset Allocation: {weights}

    Structure the briefing into these three concise sections:
    1. **Portfolio Health & Efficiency**: Interpret the risk-adjusted return and Sharpe ratio.
    2. **Downside & Tail Risk Evaluation**: Explain the 1-day 95% VaR and worst-case historical drawdown in plain financial terms.
    3. **Actionable Recommendations**: Give 2 concrete tactical hedge or rebalancing tips based on the allocation.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt
    )
    return response.text