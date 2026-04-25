import streamlit as st
from pathlib import Path
from datetime import datetime

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from ui.chart_builder import equity_curve_chart, trade_distribution_chart
from ai_agent.analyzer import PostBacktestAnalyzer
from vault_sync.writer import VaultWriter
from ui.data_loader import update_ai_summary

st.set_page_config(page_title="Backtest — QUANTPLAT", layout="wide")
st.title("🚀 Backtest")
st.markdown("Run a backtest on a Python strategy file using the LEAN engine.")

strategy_path = st.text_input(
    "Strategy file path",
    placeholder="C:/Users/noaro/QUANTPLAT/strategies/my_algo.py",
    help="Absolute path to your .py strategy file",
)
starting_capital = st.number_input(
    "Starting capital ($)", min_value=1000, max_value=10_000_000,
    value=50_000, step=1000,
)
n_mc = st.slider("Monte Carlo simulations", min_value=100, max_value=5000, value=1000, step=100)

run_btn = st.button("▶ Run Backtest", type="primary", disabled=not strategy_path)

if run_btn and strategy_path:
    path = Path(strategy_path)
    if not path.exists():
        st.error(f"File not found: `{strategy_path}`")
        st.stop()

    loader = StrategyLoader()
    try:
        adapter = loader.load(path)
    except (ValueError, NotImplementedError) as e:
        st.error(str(e))
        st.stop()

    project_dir = adapter.prepare(path)
    runner = LeanRunner()
    report_gen = ReportGenerator()

    output_box = st.empty()
    log_lines = []

    def on_output(line: str):
        log_lines.append(line)
        output_box.code("".join(log_lines[-30:]), language="text")

    status = st.status("Running LEAN backtest...", expanded=True)
    try:
        with status:
            lean_output = runner.run(project_dir, on_output=on_output)
            st.write("Generating report...")
            report = report_gen.generate(path.stem, str(path), lean_output, n_mc=n_mc)
    except LeanRunError as e:
        st.error(f"LEAN failed: {e}")
        adapter.cleanup()
        st.stop()
    finally:
        adapter.cleanup()

    status.update(label="Backtest complete!", state="complete")

    metrics = report.get("metrics", {})
    equity = lean_output.get("equity_curve", [])
    pl_list = lean_output.get("pl_list", [])

    # Run AI analyzer and write vault note
    ai_summary = None
    with st.spinner("Generating AI analysis..."):
        try:
            analyzer = PostBacktestAnalyzer()
            ai_summary = analyzer.analyze(metrics, pl_list=pl_list)
            update_ai_summary(report["report_path"], ai_summary)
            VaultWriter().write_backtest(
                path.stem,
                datetime.now(),
                metrics,
                ai_summary=ai_summary,
                results_path=report["report_path"],
            )
        except Exception as e:
            st.warning(f"AI analysis failed: {e}")

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    c2.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
    c3.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
    c4.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

    if equity:
        st.plotly_chart(equity_curve_chart(equity, title=f"{path.stem} — Equity Curve"),
                        use_container_width=True)
    if pl_list:
        st.plotly_chart(trade_distribution_chart(pl_list), use_container_width=True)

    with st.expander("Full Metrics"):
        st.json(metrics)

    st.success(f"Report saved to `{report['report_path']}`")

    if ai_summary:
        st.subheader("🤖 AI Analysis")
        st.markdown(ai_summary)
