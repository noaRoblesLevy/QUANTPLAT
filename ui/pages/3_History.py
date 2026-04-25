import ui.bootstrap  # noqa: F401 — adds project root to sys.path
import pandas as pd
import streamlit as st

from ui.data_loader import load_all_runs

st.set_page_config(page_title="History — QUANTPLAT", layout="wide")
st.title("📋 History")
st.markdown("All past backtest runs, most recent first.")

def _r(v, n=3):
    return round(v, n) if v is not None else None


runs = load_all_runs()

if not runs:
    st.info("No backtest runs yet. Go to **Backtest** to run your first strategy.")
    st.stop()

rows = [
    {
        "Strategy":          r.strategy_name,
        "Date":              r.created_at.strftime("%Y-%m-%d %H:%M"),
        "Sharpe":            _r(r.sharpe_ratio),
        "Sortino":           _r(r.sortino_ratio),
        "Max DD (%)":        _r(r.max_drawdown * 100, 2) if r.max_drawdown is not None else None,
        "Win Rate (%)":      _r(r.win_rate * 100, 1) if r.win_rate is not None else None,
        "Profit Factor":     _r(r.profit_factor),
        "Expectancy ($)":    _r(r.expectancy, 2),
        "Annual Return (%)": _r(r.annual_return * 100, 2) if r.annual_return is not None else None,
        "Trades":            r.total_trades,
        "Report Path":       r.results_path,
    }
    for r in runs
]

df = pd.DataFrame(rows)

search = st.text_input("Filter by strategy name", "")
if search:
    df = df[df["Strategy"].str.contains(search, case=False, na=False)]

st.dataframe(
    df.drop(columns=["Report Path"]),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Showing {len(df)} of {len(rows)} runs")

if len(df) > 0:
    with st.expander("Selected run details"):
        selected_name = st.selectbox("Strategy", df["Strategy"].unique())
        selected_row = df[df["Strategy"] == selected_name].iloc[0]
        st.json(selected_row.to_dict())
