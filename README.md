# Financial Assets & Portfolio Risk Engine

An end-to-end quantitative analytics and data engineering application that ingests multi-asset financial time-series data, computes rolling risk metrics, and calculates optimal portfolio allocations using Modern Portfolio Theory.

---

## Architecture Overview ```

[ Financial APIs (yfinance) ]
│
▼
[ ETL Pipeline (etl_pipeline.py) ]
• Automated extraction of historical prices
• Vectorized calculations: Log returns & 30-day rolling volatility
│
▼
[ Structured Relational Storage (portfolio.db) ]
• SQLAlchemy ORM schema mapping asset prices and metrics
│
▼
[ Quantitative Engine (risk_engine.py) ]
• Covariance & correlation matrix computation
• Monte Carlo simulation (3,000+ portfolio weightings)
• Sharpe Ratio optimization (Markowitz Efficient Frontier)
│
▼
[ Interactive UI (app.py via Streamlit) ]
• Real-time parameter tweaking, allocation charts, and heatmaps

````

---

## Features

- **Automated Data Engineering Pipeline:** Extracts daily market sessions for multi-asset baskets (Equities, Indices, ETFs) and standardizes timestamps.
- **Feature Engineering:** Computes continuous daily log returns and annualized 30-day rolling volatility metrics.
- **Relational Data Modeling:** Persists normalized tables locally using SQLite and SQLAlchemy ORM.
- **Modern Portfolio Theory:** Evaluates inter-asset correlations and identifies the asset weighting distribution that maximizes the Sharpe Ratio against a risk-free benchmark.
- **Interactive Visualizations:** Built with Streamlit and Plotly to explore risk-return profiles and correlation heatmaps dynamically.

---

## Tech Stack

- **Language:** Python 3
- **Data Engineering:** Pandas, NumPy, SQLAlchemy, SQLite
- **Quantitative Analytics:** SciPy, Statsmodels
- **Visualization & UI:** Streamlit, Plotly
- **Data Ingestion:** yfinance API
- **Version Control:** Git, GitHub

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/portfolio-risk-engine.git](https://github.com/YOUR_GITHUB_USERNAME/portfolio-risk-engine.git)
   cd portfolio-risk-engine
````

2. **Create and Activate Virtual Environment:**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Database Setup & ETL Pipeline:**

   ```bash
   python etl_pipeline.py
   ```

5. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```
