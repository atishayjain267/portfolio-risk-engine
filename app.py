import os
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sqlalchemy import inspect
from database import engine
from etl_pipeline import run_pipeline
from risk_engine import analyze_portfolio, load_returns

st.set_page_config(
    page_title="Portfolio Risk Engine", 
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Auto-initialize database on first run
inspector = inspect(engine)
if not os.path.exists("portfolio.db") or not inspector.has_table("asset_prices"):
    with st.spinner("Initializing database and extracting market feeds..."):
        run_pipeline()

# Title & Abstract Header
st.title("📊 Financial Assets & Portfolio Risk Engine")
st.markdown("""
*An end-to-end quantitative analytics platform integrating automated time-series ETL, 
relational SQLite persistence, and Modern Portfolio Theory (MPT) optimization.*
""")

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
st.sidebar.header("⚙️ Portfolio Controls")

initial_investment = st.sidebar.number_input(
    "Investment Capital (₹)", 
    min_value=10000, 
    max_value=100000000, 
    value=100000, 
    step=10000
)

risk_free_rate = st.sidebar.slider(
    "Benchmark Risk-Free Rate (%)", 
    min_value=3.0, 
    max_value=8.5, 
    value=6.5, 
    step=0.1
) / 100

num_sims = st.sidebar.slider(
    "Monte Carlo Iterations", 
    min_value=1000, 
    max_value=5000, 
    value=3000, 
    step=500
)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("Re-run Sim", use_container_width=True):
        st.rerun()

with col_btn2:
    if st.button("Sync Data", use_container_width=True):
        with st.spinner("Fetching market feeds..."):
            run_pipeline()
            st.success("Updated!")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Tracked Assets: NIFTY 50 (`^NSEI`), Nippon India Gold BeES (`GOLDBEES.NS`), TCS (`TCS.NS`), and HDFC Bank (`HDFCBANK.NS`).")

# -------------------------------------------------------------
# Quantitative Calculations
# -------------------------------------------------------------
report = analyze_portfolio(risk_free_rate=risk_free_rate, num_simulations=num_sims)
returns_df = load_returns()

# -------------------------------------------------------------
# Top KPI Metric Cards
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Optimized Sharpe Ratio", 
    f"{report['max_sharpe']}",
    help="Higher Sharpe indicates superior risk-adjusted returns."
)
c2.metric(
    "Expected Annual Return", 
    f"{report['expected_annual_return']}%"
)
c3.metric(
    "Annual Volatility (Risk)", 
    f"{report['annual_volatility_risk']}%",
    delta_color="inverse"
)
c4.metric(
    "Capital Deployed", 
    f"₹{initial_investment:,.0f}",
    help="Total capital allocated across assets."
)

st.divider()

# -------------------------------------------------------------
# Tabbed Layout
# -------------------------------------------------------------
tab_alloc, tab_perf, tab_db = st.tabs([
    "🎯 Optimal Allocation & Correlation", 
    "📈 Growth Trajectory & Risk Spectrum", 
    "🗄️ SQLite Data Explorer"
])

# --- TAB 1: ALLOCATION & CORRELATION ---
with tab_alloc:
    col_alloc, col_corr = st.columns([1.2, 1])

    with col_alloc:
        st.subheader("Markowitz Efficient Frontier Allocation")
        st.caption(f"Asset distribution calculated on a ₹{initial_investment:,.0f} baseline.")
        
        weights_data = pd.DataFrame({
            "Asset": list(report["optimal_weights"].keys()),
            "Weight (%)": [round(v * 100, 2) for v in report["optimal_weights"].values()],
            "Capital Allocation (₹)": [round(v * initial_investment, 2) for v in report["optimal_weights"].values()]
        })
        
        sub_col1, sub_col2 = st.columns([1.2, 1])
        with sub_col1:
            fig_pie = px.pie(
                weights_data, 
                values="Weight (%)", 
                names="Asset", 
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with sub_col2:
            st.dataframe(
                weights_data.style.format({
                    "Weight (%)": "{:.2f}%", 
                    "Capital Allocation (₹)": "₹{:,.2f}"
                }),
                hide_index=True,
                use_container_width=True
            )

    with col_corr:
        st.subheader("Cross-Asset Correlation Matrix")
        st.caption("Quantifies hedging efficacy and co-movement between assets.")
        corr = report["correlation_matrix"]
        fig_heat = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Blues"
        )
        fig_heat.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 2: PERFORMANCE & RISK SPECTRUM ---
with tab_perf:
    # Downside Risk Cards
    r1, r2, r3 = st.columns(3)
    r1.metric(
        "1-Day 95% Value at Risk (VaR)", 
        f"{report['var_95_daily_pct']}%",
        help="Expected maximum 1-day loss with 95% confidence."
    )
    r2.metric(
        "Max 1-Day Capital at Risk", 
        f"₹{initial_investment * (report['var_95_daily_pct'] / 100):,.2f}",
        help="Potential daily capital loss under normal market movements."
    )
    r3.metric(
        "Max Historical Drawdown", 
        f"{report['max_drawdown_pct']}%",
        delta_color="inverse",
        help="Worst peak-to-trough decline experienced by this allocation."
    )

    st.markdown("---")

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
        fig_cum.update_layout(
            legend_title_text="Ticker", 
            margin=dict(t=20, b=20, l=20, r=20),
            height=340
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with col_scatter:
        st.subheader("Asset Risk vs. Expected Return")
        st.caption("Annualized performance distribution of basket components.")
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
            size=[18] * len(risk_df),
            color="Annual Return (%)",
            color_continuous_scale="Viridis"
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=340)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Underwater Drawdown Curve
    st.subheader("Underwater Portfolio Drawdown")
    st.caption("Visualizing capital drawdown depth and recovery periods over time.")
    fig_dd = px.area(
        report["drawdown_series"],
        labels={"value": "Drawdown (%)", "trade_date": "Date"},
        color_discrete_sequence=["#EF553B"]
    )
    fig_dd.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=260)
    st.plotly_chart(fig_dd, use_container_width=True)

# --- TAB 3: DATA ENGINEERING EXPLORER ---
with tab_db:
    st.subheader("Relational Database Records (`portfolio.db`)")
    st.caption("Inspecting the latest 100 transformed time-series records loaded via SQLAlchemy.")
    
    raw_df = pd.read_sql(
        "SELECT ticker, trade_date, close_price, daily_return, rolling_vol FROM asset_prices ORDER BY trade_date DESC LIMIT 100", 
        con=engine
    )
    st.dataframe(raw_df, use_container_width=True, height=400)