import streamlit as st
import plotly.express as px
from risk_engine import analyze_portfolio, load_returns

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide")

st.title("Financial Assets & Portfolio Risk Engine")
st.caption("Automated Time-Series Data Engineering & Portfolio Optimization")

# Sidebar settings
st.sidebar.header("Model Parameters")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=3.0, max_value=8.0, value=6.5, step=0.1) / 100
num_sims = st.sidebar.slider("Simulated Portfolios", min_value=1000, max_value=5000, value=3000, step=500)

if st.sidebar.button("Re-run Optimization"):
    st.rerun()

# Run Engine
report = analyze_portfolio(risk_free_rate=risk_free_rate, num_simulations=num_sims)

# Summary Cards
c1, c2, c3 = st.columns(3)
c1.metric("Expected Annual Return", f"{report['expected_annual_return']}%")
c2.metric("Annual Volatility (Risk)", f"{report['annual_volatility_risk']}%")
c3.metric("Max Sharpe Ratio", f"{report['max_sharpe']}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Optimal Asset Allocation (Markowitz Frontier)")
    weights_data = {"Asset": list(report["optimal_weights"].keys()), "Weight": list(report["optimal_weights"].values())}
    fig_pie = px.pie(weights_data, values="Weight", names="Asset", hole=0.45)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Asset Correlation Heatmap")
    corr = report["correlation_matrix"]
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_heat, use_container_width=True)