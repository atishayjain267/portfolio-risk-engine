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
    page_title="Smart Portfolio Optimizer & Risk Engine", 
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="expanded"
)

NAME_MAP = {
    "^NSEI": "NIFTY 50",
    "GOLDBEES.NS": "Gold BeES (Gold ETF)",
    "HDFCBANK.NS": "HDFC Bank",
    "TCS.NS": "TCS (IT)",
    "ITC.NS": "ITC Ltd"
}

# Auto-initialize database on boot
inspector = inspect(engine)
if not os.path.exists("portfolio.db") or not inspector.has_table("asset_returns"):
    with st.spinner("Setting up database and downloading market data..."):
        run_pipeline()

# Title & Header
st.title("📈 Smart Portfolio Optimizer & Risk Engine")
st.caption("Quantitative Modern Portfolio Theory (MPT) optimization tailored for Indian markets.")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Investor Profile & Settings")

investor_profile = st.sidebar.radio(
    "Investor Risk Profile",
    ["Balanced (Growth + Hedge)", "Conservative (Capital Preservation)", "Aggressive (Maximum Growth)"],
    index=0,
    help="Conservative enforces high Gold weight; Aggressive tilts heavily into Equities."
)

initial_investment = st.sidebar.number_input(
    "Lump Sum Capital (₹)", 
    min_value=10000, 
    max_value=10000000, 
    value=100000, 
    step=10000,
    help="Initial capital available to allocate today."
)

monthly_sip = st.sidebar.number_input(
    "Monthly SIP Addition (₹)", 
    min_value=0, 
    max_value=100000, 
    value=5000, 
    step=1000,
    help="Recurring investment added at the end of each month."
)

horizon_years = st.sidebar.slider(
    "Investment Horizon (Years)", 
    min_value=1, 
    max_value=15, 
    value=5
)

rfr_input = st.sidebar.slider(
    "Risk-Free Return Rate (FD / Sovereign Bond %)", 
    min_value=3.0, 
    max_value=8.5, 
    value=6.5, 
    step=0.25,
    help="The benchmark risk-free hurdle rate (e.g. RBI bonds or bank FDs)."
)
risk_free_rate = rfr_input / 100

sim_count = st.sidebar.selectbox(
    "Monte Carlo Iterations", 
    options=[1000, 2000, 3000], 
    index=1
)

col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("🔄 Recalculate", use_container_width=True):
        st.rerun()
with col_b2:
    if st.button("⚡ Sync Feeds", use_container_width=True):
        with st.spinner("Fetching latest market feeds..."):
            run_pipeline()
            st.success("Feeds Refreshed!")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Tracked Basket: NIFTY 50, Gold BeES, HDFC Bank, TCS, and ITC.")

# --- ANALYTICS EXECUTION ---
returns_df = load_returns(engine)
report = analyze_portfolio(
    returns_df, 
    risk_free_rate=risk_free_rate, 
    num_simulations=sim_count,
    investor_profile=investor_profile
)

# --- SECTION 1: TOP KPI CARDS ---
c1, c2, c3, c4 = st.columns(4)

vol = report['annual_volatility_risk']
risk_tag = "Low Risk" if vol < 15 else ("Moderate Risk" if vol < 22 else "High Risk")

c1.metric("Sharpe Score", f"{report['max_sharpe']}", help="Measures return earned per unit of risk. Scores > 1.0 indicate strong efficiency.")
c2.metric("Expected Annual Return", f"{report['expected_annual_return']}%")
c3.metric("Annual Volatility", f"{report['annual_volatility_risk']}%", delta=risk_tag, delta_color="off")
c4.metric("Starting Capital", f"₹{initial_investment:,.0f}")

st.info(
    f"💡 **Takeaway:** Configured for **{investor_profile.split(' ')[0]}** strategy. "
    f"For every **1% of volatility**, this portfolio targets **{report['max_sharpe']}% excess return** over risk-free instruments."
)

st.markdown("---")

# --- SECTION 2: ALLOCATION, RUPEE BREAKDOWN & CHECKLIST ---
col_pie, col_table, col_corr = st.columns([1.1, 1.2, 1.1])

weights_raw = report["optimal_weights"]
df_weights = pd.DataFrame({
    "Asset": [NAME_MAP.get(k, k) for k in weights_raw.keys()],
    "Allocation (%)": [round(v * 100, 1) for v in weights_raw.values()],
    "Amount (₹)": [round(v * initial_investment, 2) for v in weights_raw.values()]
})

with col_pie:
    st.subheader("🎯 Optimal Split")
    fig_donut = px.pie(
        df_weights, 
        names="Asset", 
        values="Allocation (%)", 
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=310)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_table:
    st.subheader("📋 Capital Breakdown")
    st.dataframe(
        df_weights.style.format({
            "Allocation (%)": "{:.1f}%",
            "Amount (₹)": "₹{:,.0f}"
        }),
        hide_index=True,
        use_container_width=True
    )
    
    csv_data = df_weights.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Allocation Plan (CSV)",
        data=csv_data,
        file_name="Optimal_Portfolio_Allocation.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_corr:
    st.subheader("🔗 Inter-Asset Correlation")
    corr_matrix = returns_df.corr().rename(columns=NAME_MAP, index=NAME_MAP)
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="Blues"
    )
    fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=310)
    st.plotly_chart(fig_corr, use_container_width=True)

# Feature 3: One-Click Execution Checklist
with st.expander("✅ Broker Execution Checklist (Mark as you place orders)"):
    st.caption("Track your order execution on your trading terminal (Zerodha Kite / Groww):")
    cols_check = st.columns(len(df_weights))
    for idx, row in df_weights.iterrows():
        with cols_check[idx]:
            st.checkbox(f"Buy **{row['Asset']}**\n\n₹{row['Amount (₹)']:,.0f}", key=f"chk_{idx}")

st.markdown("---")

# --- SECTION 3: SIP WEALTH PROJECTION ---
st.subheader("🌱 Long-Term SIP Wealth Projection")
st.caption(f"Simulating compound growth of ₹{initial_investment:,.0f} lump sum + ₹{monthly_sip:,.0f}/month SIP over a {horizon_years}-year horizon.")

months = horizon_years * 12
monthly_rate = (report['expected_annual_return'] / 100) / 12

projection_data = []
current_value = float(initial_investment)
total_invested = float(initial_investment)

for m in range(1, months + 1):
    current_value = current_value * (1 + monthly_rate) + monthly_sip
    total_invested += monthly_sip
    
    if m % 12 == 0:
        projection_data.append({
            "Year": f"Yr {m // 12}",
            "Invested Capital (₹)": round(total_invested),
            "Projected Portfolio (₹)": round(current_value)
        })

proj_df = pd.DataFrame(projection_data)

col_sip_chart, col_sip_metrics = st.columns([2, 1])

with col_sip_chart:
    fig_sip = px.bar(
        proj_df, 
        x="Year", 
        y=["Invested Capital (₹)", "Projected Portfolio (₹)"],
        barmode="group",
        color_discrete_sequence=["#94A3B8", "#10B981"]
    )
    fig_sip.update_layout(
        legend_title_text="", 
        margin=dict(t=10, b=10, l=10, r=10), 
        height=320,
        hovermode="x unified"
    )
    st.plotly_chart(fig_sip, use_container_width=True)

with col_sip_metrics:
    st.metric("Total Out-of-Pocket", f"₹{total_invested:,.0f}")
    st.metric(
        "Estimated Final Value", 
        f"₹{current_value:,.0f}", 
        delta=f"+ ₹{current_value - total_invested:,.0f} Wealth Created"
    )
    st.caption("Projections assume continuous reinvestment under normalized annualized compound returns.")

st.markdown("---")

# Feature 2: Historical Crisis Stress Testing
st.subheader("🛡️ Historical Crisis Stress-Test Engine")
st.caption("Simulated portfolio drawdown vs. NIFTY 50 during major market shocks.")

# Weight array aligned with columns
w_array = np.array([weights_raw.get(col, 0.0) for col in returns_df.columns])

# Stress scenarios: Asset impact shocks
crisis_scenarios = {
    "COVID-19 Crash (March 2020)": {
        "^NSEI": -0.38, "GOLDBEES.NS": 0.08, "HDFCBANK.NS": -0.42, "TCS.NS": -0.22, "ITC.NS": -0.28
    },
    "Global Inflation & Rate Hikes (2022)": {
        "^NSEI": -0.15, "GOLDBEES.NS": 0.14, "HDFCBANK.NS": -0.18, "TCS.NS": -0.26, "ITC.NS": 0.52
    },
    "Budget Day / Election Volatility Shock": {
        "^NSEI": -0.06, "GOLDBEES.NS": 0.02, "HDFCBANK.NS": -0.08, "TCS.NS": -0.04, "ITC.NS": -0.05
    }
}

stress_results = []
for event_name, shocks in crisis_scenarios.items():
    port_impact = sum(weights_raw.get(t, 0.0) * shocks.get(t, 0.0) for t in weights_raw)
    nifty_impact = shocks.get("^NSEI", -0.20)
    capital_loss = initial_investment * port_impact
    stress_results.append({
        "Stress Scenario": event_name,
        "Portfolio Impact": f"{port_impact * 100:+.1f}%",
        "NIFTY 50 Benchmark": f"{nifty_impact * 100:+.1f}%",
        "Capital Impact (₹)": f"₹{capital_loss:+,.0f}",
        "Hedging Buffer": f"{(port_impact - nifty_impact) * 100:+.1f}% vs Index"
    })

st.table(pd.DataFrame(stress_results))

st.markdown("---")

# --- SECTION 5: HISTORICAL TRAJECTORY & DATABASE ---
col_growth, col_scatter = st.columns(2)

with col_growth:
    st.subheader("📈 Historical Growth Trajectory")
    cum_growth = (1 + returns_df).cumprod() - 1
    cum_growth_named = cum_growth.rename(columns=NAME_MAP)
    fig_growth = px.line(cum_growth_named, labels={"value": "Growth (1.0 = +100%)", "Date": "Trade Date"})
    fig_growth.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig_growth, use_container_width=True)

with col_scatter:
    st.subheader("⚖️ Risk vs. Return Spectrum")
    annual_returns = (returns_df.mean() * 252 * 100).round(2)
    annual_risks = (returns_df.std() * (252 ** 0.5) * 100).round(2)
    
    scatter_df = pd.DataFrame({
        "Ticker": [NAME_MAP.get(t, t) for t in returns_df.columns],
        "Annual Return (%)": annual_returns.values,
        "Annual Volatility (%)": annual_risks.values
    })
    
    fig_scatter = px.scatter(
        scatter_df, 
        x="Annual Volatility (%)", 
        y="Annual Return (%)", 
        text="Ticker",
        size=[20] * len(scatter_df),
        color="Annual Return (%)",
        color_continuous_scale="Viridis"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

with st.expander("🔍 View Raw Historical Database Records (SQLite)"):
    raw_preview = pd.read_sql("SELECT * FROM asset_returns ORDER BY Date DESC LIMIT 20", con=engine)
    raw_preview = raw_preview.rename(columns=NAME_MAP)
    st.dataframe(raw_preview, use_container_width=True)