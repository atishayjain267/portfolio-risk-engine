import os
import streamlit as st
import plotly.express as px
import pandas as pd
from sqlalchemy import inspect
from database import engine
from etl_pipeline import run_pipeline
from risk_engine import analyze_portfolio, load_returns

# 1. Page Configuration
st.set_page_config(
    page_title="Portfolio Risk Engine", 
    page_icon="📈",
    layout="wide"
)

# Friendly display names for Indian market assets
NAME_MAP = {
    "^NSEI": "NIFTY 50",
    "GOLDBEES.NS": "Gold BeES (Gold ETF)",
    "HDFCBANK.NS": "HDFC Bank",
    "TCS.NS": "TCS (IT)",
    "ITC.NS": "ITC Ltd"
}

# 2. Auto-initialize Database on boot
inspector = inspect(engine)
if not os.path.exists("portfolio.db") or not inspector.has_table("asset_returns"):
    with st.spinner("Setting up database and downloading market data..."):
        run_pipeline()

# 3. Simple Header
st.title("📈 Smart Portfolio Optimizer")
st.write(
    "This tool analyzes historical market prices to find the **safest and most rewarding investment recipe** "
    "across equities, market indices, and gold."
)

# 4. Sidebar Controls (Simple, Friendly Labels)
st.sidebar.header("⚙️ Settings")
rfr_input = st.sidebar.slider(
    "Risk-Free Interest Rate (FD / Govt Bond %)", 
    min_value=3.0, 
    max_value=8.5, 
    value=6.5, 
    step=0.5,
    help="The guaranteed interest rate you would get without taking market risk."
)
risk_free_rate = rfr_input / 100

sim_count = st.sidebar.selectbox(
    "Simulation Speed", 
    options=[1000, 2000, 3000], 
    index=1,
    help="How many random weight combinations the engine will test."
)

if st.sidebar.button("🔄 Recalculate Portfolio"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Basket: NIFTY 50, Gold BeES, HDFC Bank, TCS, and ITC.")

# 5. Run Backend Engine
returns_df = load_returns(engine)
report = analyze_portfolio(returns_df, risk_free_rate=risk_free_rate)

# 6. Key Metrics (Easy to understand cards)
col1, col2, col3 = st.columns(3)
col1.metric("Expected Annual Return", f"{report['expected_annual_return']}%", help="Estimated average gain per year.")
col2.metric("Annual Risk (Volatility)", f"{report['annual_volatility_risk']}%", help="How much the portfolio swings up or down.")
col3.metric("Sharpe Score", f"{report['max_sharpe']}", help="Higher is better: measures reward per unit of risk.")

st.markdown("---")

# 7. Main Visuals: Clean 2-Column Split
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🎯 Recommended Investment Split")
    st.caption("How much of your money should go into each asset for the best risk-reward balance.")
    
    # Map raw tickers to clean names
    raw_tickers = list(report["optimal_weights"].keys())
    clean_names = [NAME_MAP.get(ticker, ticker) for ticker in raw_tickers]
    
    weights_df = pd.DataFrame({
        "Asset": clean_names,
        "Percentage": [round(v * 100, 1) for v in report["optimal_weights"].values()]
    })
    
    fig_donut = px.pie(
        weights_df, 
        names="Asset", 
        values="Percentage", 
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_donut, use_container_width=True)

with right_col:
    st.subheader("🔗 Asset Relationships (Correlation)")
    st.caption("Values close to 0 mean the assets protect each other when the market drops.")
    
    corr_matrix = returns_df.corr().rename(columns=NAME_MAP, index=NAME_MAP)
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="Blues"
    )
    fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_corr, use_container_width=True)

# 8. Collapsible Data Preview (Keeps page clean unless opened)
with st.expander("🔍 View Raw Database Records"):
    raw_preview = pd.read_sql("SELECT * FROM asset_returns ORDER BY Date DESC LIMIT 20", con=engine)
    # Also rename the columns in the preview table for better readability
    raw_preview = raw_preview.rename(columns=NAME_MAP)
    st.dataframe(raw_preview, use_container_width=True)