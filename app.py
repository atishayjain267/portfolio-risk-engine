import os
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sqlalchemy import inspect
from database import engine
from etl_pipeline import run_pipeline
from risk_engine import analyze_portfolio, load_returns

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide", initial_sidebar_state="expanded")

# Auto-initialize database on first run
inspector = inspect(engine)
if not os.path.exists("portfolio.db") or not inspector.has_table("asset_prices"):
    with st.spinner("Initializing database and extracting market feeds..."):
        run_pipeline()

# Title & Abstract Header
st.title("Financial Assets & Portfolio Risk Engine")
st.markdown("""
*An end-to-end quantitative analytics engine integrating automated time-series ETL, 
relational SQLite persistence, and Modern Portfolio Theory (MPT) optimization.*
""")

# Sidebar Controls
st.sidebar.header("Quantitative Parameters")
risk_free_rate = st.sidebar.slider("Benchmark Risk-Free Rate (%)", min_value=3.0, max_value=8.5, value=6.5, step=0.1) / 100
num_sims = st.sidebar.slider("Monte Carlo Iterations", min_value=1000, max_value=5000, value=3000, step=500)

if st.sidebar.button("Re-run Simulation"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Data extracted via automated pipeline: NIFTY 50 (`^NSEI`), Nippon India Gold BeES (`GOLDBEES.NS`), TCS (`TCS.NS`), and HDFC Bank (`HDFCBANK.NS`).")

# Execute Engine
report = analyze_portfolio(risk_free_rate=risk_free_rate, num_simulations=num_sims)
returns_df = load_returns()

# Top KPI Metric Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimized Sharpe Ratio", f"{report['max_sharpe']}")
c2.metric("Expected Annual Return", f"{report['expected_annual_return']}%")
c3.metric("Annualized Volatility (Risk)", f"{report['annual_volatility_risk']}%")
c4.metric("Assets Analyzed", f"{len(report['optimal_weights'])}")

st.divider()

# Section 1: Optimization & Correlation
col_alloc, col_corr = st.columns(2)

with col_alloc:
    st.subheader("Optimal Asset Allocation")
    st.caption("Weightings selected by Markowitz Efficient Frontier optimization.")
    weights_data = pd.DataFrame({
        "Asset": list(report["optimal_weights"].keys()),
        "Weight (%)": [round(v * 100, 2) for v in report["optimal_weights"].values()]
    })
    fig_pie = px.pie(
        weights_data, 
        values="Weight (%)", 
        names="Asset", 
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_corr:
    st.subheader("Asset Correlation Matrix")
    st.caption("Quantifying cross-asset hedging and diversification.")
    corr = report["correlation_matrix"]
    fig_heat = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale="Blues"
    )
    fig_heat.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# Section 2: Historical Performance & Risk Spectrum
col_trend, col_scatter = st.columns(2)

with col_trend:
    st.subheader("Cumulative Growth Trajectory")
    st.caption("Normalized compound growth derived from continuous log returns.")
    cumulative_returns = np.exp(returns_df.cumsum()) - 1
    fig_cum = px.line(
        cumulative_returns, 
        labels={"value": "Cumulative Return (1.0 = 100%)", "trade_date": "Date"},
        line_shape="spline"
    )
    fig_cum.update_layout(legend_title_text="Ticker", margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_cum, use_container_width=True)

with col_scatter:
    st.subheader("Asset Risk vs. Expected Return")
    st.caption("Annualized performance distribution of individual basket components.")
    annual_returns = (returns_df.mean() * 252 * 100).round(2)
    annual_risks = (returns_df.std() * np.sqrt(252) * 100).round(2)
    risk_df = pd.DataFrame({
        "Ticker": returns_df.columns,
        "Annual Return (%)": annual_returns.values,
        "Annual Risk (%)": annual_risks.values
    })
    fig_scatter = px.scatter(
        risk_df, 
        x="Annual Risk (%)", 
        y="Annual Return (%)", 
        text="Ticker",
        size=[15]*len(risk_df),
        color="Annual Return (%)",
        color_continuous_scale="Viridis"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

# Section 3: Data Engineering Explorer
with st.expander("Database Records Explorer (SQLite: asset_prices)"):
    raw_df = pd.read_sql("SELECT ticker, trade_date, close_price, daily_return, rolling_vol FROM asset_prices ORDER BY trade_date DESC LIMIT 100", con=engine)
    st.dataframe(raw_df, use_container_width=True)