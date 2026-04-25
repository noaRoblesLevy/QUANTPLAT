import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="QUANTPLAT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 QUANTPLAT")
st.markdown("**Quantitative Trading Platform** — Write, optimize, and backtest trading algorithms.")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Platform", "QUANTPLAT v1")
with col2:
    st.metric("Engine", "LEAN (QuantConnect)")
with col3:
    st.metric("Status", "Ready")

st.markdown("""
### Getting Started
1. **Backtest** — select a Python strategy file and run a backtest
2. **Analytics** — view equity curve, Monte Carlo simulations, and metrics for any past run
3. **History** — browse all past backtest runs

Use the sidebar to navigate.
""")
