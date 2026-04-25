import ui.bootstrap  # noqa: F401 — adds project root to sys.path
import streamlit as st

from ui.data_loader import load_all_runs, load_report
from ui.chart_builder import equity_curve_chart, monte_carlo_chart, trade_distribution_chart

st.set_page_config(page_title="Analytics — QUANTPLAT", layout="wide")
st.title("📊 Analytics")
st.markdown("View detailed analytics for a past backtest run.")

runs = load_all_runs()

if not runs:
    st.info("No backtest runs yet. Go to **Backtest** to run your first strategy.")
    st.stop()

run_labels = [
    f"{r.strategy_name}  |  {r.created_at.strftime('%Y-%m-%d %H:%M')}  |  Sharpe {r.sharpe_ratio:.2f}"
    for r in runs
]
selected_idx = st.selectbox("Select backtest run", range(len(run_labels)),
                             format_func=lambda i: run_labels[i])
selected_run = runs[selected_idx]

report = load_report(selected_run.results_path)
if report is None:
    st.error(f"Report file not found: `{selected_run.results_path}`")
    st.stop()

metrics = report.get("metrics", {})
mc = report.get("monte_carlo", {})
equity = report.get("equity_curve", [])
pl_list = report.get("pl_list", [])

st.subheader(f"Strategy: `{selected_run.strategy_name}`")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sharpe Ratio",   f"{metrics.get('sharpe_ratio', 0):.2f}")
c2.metric("Sortino Ratio",  f"{metrics.get('sortino_ratio', 0):.2f}")
c3.metric("Max Drawdown",   f"{metrics.get('max_drawdown', 0)*100:.1f}%")
c4.metric("Win Rate",       f"{metrics.get('win_rate', 0)*100:.1f}%")
c5.metric("Profit Factor",  f"{metrics.get('profit_factor', 0):.2f}")

tab1, tab2, tab3 = st.tabs(["Equity Curve", "Monte Carlo", "Trade Distribution"])

with tab1:
    if equity:
        st.plotly_chart(
            equity_curve_chart(equity, title=f"{selected_run.strategy_name} — Equity Curve"),
            use_container_width=True,
        )
    else:
        st.warning("No equity curve data in this report.")

with tab2:
    if mc and mc.get("percentiles"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probability of Ruin", f"{mc.get('probability_of_ruin', 0)*100:.1f}%")
        col_b.metric("Median Final Equity", f"${mc.get('final_equity', {}).get('p50', 0):,.0f}")
        col_c.metric("95th pct Max DD",     f"{mc.get('max_drawdown', {}).get('p95', 0)*100:.1f}%")
        st.plotly_chart(
            monte_carlo_chart(mc, title=f"{selected_run.strategy_name} — Monte Carlo ({mc.get('n_simulations', 0)} sims)"),
            use_container_width=True,
        )
    else:
        st.warning("No Monte Carlo data in this report.")

with tab3:
    if pl_list:
        st.plotly_chart(trade_distribution_chart(pl_list), use_container_width=True)
    else:
        st.warning("No trade P&L data in this report.")

with st.expander("Full Metrics JSON"):
    st.json(metrics)
