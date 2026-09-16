# Quantitative Assets & Portfolio Risk Engine

An end-to-end financial data engineering and quantitative analytics platform. The application automates ingestion of multi-asset market data, models portfolio optimization using Modern Portfolio Theory (Markowitz Efficient Frontier), quantifies tail risk via Value at Risk (VaR) and Maximum Drawdown, performs historical macroeconomic stress testing, and synthesizes executive risk memorandums using Google Gemini 3.6 Flash.

---

## Architecture Pipeline

```text
[ Financial Market Data (yfinance) ]
                 │
                 ▼
[ Data Engineering & Ingestion Pipeline ]
  • Log returns computation: R_t = ln(P_t / P_{t-1})
  • Time-series timestamp reconciliation & corporate action adjustment
  • Rolling volatility & covariance matrix generation
                 │
                 ▼
[ Relational Persistence Layer (SQLite & SQLAlchemy) ]
  • Schema: `asset_prices` indexed on composite key (ticker, trade_date)
                 │
                 ▼
[ Quantitative Modeling & Risk Analytics (`risk_engine.py`) ]
  • 3,000-iteration Monte Carlo simulation for Markowitz Mean-Variance Optimization
  • Tangency portfolio isolation (Maximum Sharpe Ratio)
  • Downside tail risk: Parametric & Historical Value at Risk (1-Day 95% VaR)
  • Macroeconomic shock testing (2008 GFC, 2020 COVID crash, 2000 Dot-Com, 2022 Rates/Tech)
                 │
                 ▼
[ Interactive Presentation Dashboard (`app.py`) ]
  • Efficient Frontier scatter visualizer & optimal allocation donut charts
  • Real-time capital risk exposure calculators
                 │
                 ▼
[ GenAI Decision Layer (`llm_analyst.py`) ]
  • Automated risk committee memorandums via Google Gemini 3.6 Flash
```
