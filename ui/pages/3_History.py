import pandas as pd
import streamlit as st

from ui.data_loader import load_all_runs

st.set_page_config(page_title="History — QUANTPLAT", layout="wide")
st.title("📋 History")
st.markdown("All past backtest runs, most recent first.")

runs = load_all_runs()

if not runs:
    st.info("No backtest runs yet. Go to **Backtest** to run your first strategy.")
    st.stop()

rows = [
    {
        "Strategy":        r.strategy_name,
        "Date":            r.created_at.strftime("%Y-%m-%d %H:%M"),
        "Sharpe":          round(r.sharpe_ratio, 3),
        "Sortino":         round(r.sortino_ratio, 3),
        "Max DD (%)":      round(r.max_drawdown * 100, 2),
        "Win Rate (%)":    round(r.win_rate * 100, 1),
        "Profit Factor":   round(r.profit_factor, 3),
        "Expectancy ($)":  round(r.expectancy, 2),
        "Annual Return (%)": round(r.annual_return * 100, 2),
        "Trades":          r.total_trades,
        "Report Path":     r.results_path,
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
