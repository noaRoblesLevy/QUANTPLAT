import ui.bootstrap  # noqa: F401 — adds project root to sys.path
import streamlit as st
import json

from ai_agent.copilot import StrategyCopilot
from ui.data_loader import load_all_runs

st.set_page_config(page_title="AI Chat — QUANTPLAT", layout="wide")
st.title("🤖 AI Chat")
st.markdown("Strategy code review and post-backtest analysis.")

tab_copilot, tab_history = st.tabs(["Copilot", "Analysis History"])

with tab_copilot:
    st.subheader("Strategy Code Review")
    st.markdown(
        "Paste your strategy code below. "
        "Optionally add backtest metrics (JSON) to get context-aware suggestions."
    )

    code = st.text_area(
        "Strategy code",
        placeholder="Paste your Python strategy code here...",
        height=300,
    )

    metrics_json = st.text_area(
        "Backtest metrics (optional JSON)",
        placeholder='{"sharpe_ratio": 1.2, "max_drawdown": -0.15}',
        height=80,
    )

    review_btn = st.button("🔍 Review Strategy", type="primary", disabled=not code.strip())

    if review_btn and code.strip():
        metrics = None
        if metrics_json.strip():
            try:
                metrics = json.loads(metrics_json)
            except json.JSONDecodeError:
                st.warning("Invalid metrics JSON — running review without metrics.")

        with st.spinner("Reviewing strategy..."):
            copilot = StrategyCopilot()
            st.session_state["last_review"] = copilot.review(code, metrics=metrics)

    if "last_review" in st.session_state:
        st.subheader("Review")
        st.markdown(st.session_state["last_review"])

with tab_history:
    st.subheader("Analysis History")
    st.markdown("AI summaries generated after each backtest run.")

    runs = load_all_runs()
    runs_with_summary = [r for r in runs if r.ai_summary]

    if not runs_with_summary:
        st.info(
            "No AI analyses yet. Run a backtest to generate automatic analysis."
        )
    else:
        for run in runs_with_summary:
            date_str = run.created_at.strftime("%Y-%m-%d %H:%M")
            sharpe_str = f"{run.sharpe_ratio:.2f}" if run.sharpe_ratio is not None else "N/A"
            with st.expander(f"{run.strategy_name} — {date_str} | Sharpe {sharpe_str}"):
                st.markdown(run.ai_summary)
                st.caption(f"Results: `{run.results_path}`")
