import ui.bootstrap  # noqa: F401 — adds project root to sys.path
import json
import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path

from optimizer.grid_search import GridSearchOptimizer
from optimizer.walk_forward import WalkForwardOptimizer
from optimizer.ai_optimizer import AIOptimizer
from ui.chart_builder import optimizer_scatter_chart

st.set_page_config(page_title="Optimizer — QUANTPLAT", layout="wide")
st.title("⚙️ Optimizer")
st.markdown("Optimize strategy parameters using grid search, walk-forward, or AI-guided Optuna.")

mode = st.selectbox("Optimization mode", ["Grid Search", "Walk-Forward", "AI (Optuna + LLM)"])
strategy_path = st.text_input(
    "Strategy file path",
    placeholder="C:/Users/noaro/QUANTPLAT/strategies/my_algo.py",
)
starting_capital = st.number_input(
    "Starting capital ($)", min_value=1000, max_value=10_000_000, value=50_000, step=1000
)

st.subheader("Parameters")

if mode == "Grid Search":
    st.markdown("Enter a JSON dict mapping parameter names to lists of values.")
    params_json = st.text_area(
        "Parameter grid (JSON)",
        value='{"sma_fast": [5, 10, 20], "sma_slow": [50, 100, 200]}',
        height=100,
    )
elif mode == "Walk-Forward":
    st.markdown("Enter a JSON dict mapping parameter names to lists of values.")
    params_json = st.text_area(
        "Parameter grid (JSON)",
        value='{"sma_fast": [5, 10, 20], "sma_slow": [50, 100]}',
        height=100,
    )
    col1, col2 = st.columns(2)
    wf_start = col1.date_input("Start date", value=date(2020, 1, 1))
    wf_end = col2.date_input("End date", value=date(2023, 1, 1))
    n_windows = st.slider("Number of windows", min_value=2, max_value=10, value=5)
else:  # AI (Optuna + LLM)
    st.markdown(
        "Enter a JSON dict mapping parameter names to `[min, max, type]` "
        "where type is `\"int\"` or `\"float\"`."
    )
    params_json = st.text_area(
        "Parameter ranges (JSON)",
        value='{"sma_fast": [5, 50, "int"], "sma_slow": [20, 200, "int"]}',
        height=100,
    )
    n_trials = st.slider("Number of trials", min_value=10, max_value=200, value=50, step=10)
    llm_interval = st.slider("LLM suggestion every N trials", min_value=5, max_value=50, value=10)

run_btn = st.button("▶ Run Optimization", type="primary", disabled=not strategy_path)

if run_btn and strategy_path:
    path = Path(strategy_path)
    if not path.exists():
        st.error(f"File not found: `{strategy_path}`")
        st.stop()

    try:
        raw_params = json.loads(params_json)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in parameters: {e}")
        st.stop()

    trial_log = []
    progress_placeholder = st.empty()

    def on_trial(trial_num: int, metrics: dict):
        trial_log.append({"trial": trial_num, **metrics})
        sharpe = metrics.get("sharpe_ratio") or 0.0
        progress_placeholder.caption(
            f"Trial {trial_num + 1} — Sharpe: {sharpe:.3f}"
        )

    status = st.status("Running optimization...", expanded=True)
    try:
        with status:
            if mode == "Grid Search":
                opt = GridSearchOptimizer()
                result = opt.run(
                    str(path),
                    param_grid=raw_params,
                    starting_capital=starting_capital,
                    on_trial=on_trial,
                )
            elif mode == "Walk-Forward":
                opt = WalkForwardOptimizer()
                result = opt.run(
                    str(path),
                    param_grid=raw_params,
                    start_date=wf_start,
                    end_date=wf_end,
                    n_windows=n_windows,
                    starting_capital=starting_capital,
                    on_trial=on_trial,
                )
            else:
                opt = AIOptimizer()
                result = opt.run(
                    str(path),
                    param_ranges=raw_params,
                    n_trials=n_trials,
                    starting_capital=starting_capital,
                    llm_interval=llm_interval,
                    on_trial=on_trial,
                )
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()

    status.update(label="Optimization complete!", state="complete")
    progress_placeholder.empty()

    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode", result.mode)
    col2.metric("Trials", result.n_trials)
    col3.metric("Best Sharpe", f"{result.best_sharpe:.3f}" if result.best_sharpe is not None else "N/A")

    if result.best_params:
        st.markdown("**Best parameters:**")
        st.json(json.loads(result.best_params))

    if trial_log:
        df = pd.DataFrame(trial_log)

        if mode in ("Grid Search", "Walk-Forward"):
            first_param = list(raw_params.keys())[0]
            if "params" in df.columns:
                df[first_param] = df["params"].apply(
                    lambda p: p.get(first_param) if isinstance(p, dict) else None
                )
            if first_param in df.columns and "sharpe_ratio" in df.columns:
                vals = df[first_param].dropna().tolist()
                sharpes = df.loc[df[first_param].notna(), "sharpe_ratio"].fillna(0).tolist()
                st.plotly_chart(
                    optimizer_scatter_chart(vals, sharpes, x_label=first_param,
                                           title=f"{first_param} vs Sharpe"),
                    use_container_width=True,
                )
        elif "sharpe_ratio" in df.columns:
            st.plotly_chart(
                optimizer_scatter_chart(
                    list(range(len(df))),
                    df["sharpe_ratio"].fillna(0).tolist(),
                    x_label="Trial #",
                    title="Sharpe Ratio per Trial",
                ),
                use_container_width=True,
            )

        with st.expander("All trials"):
            display_cols = ["trial", "sharpe_ratio", "win_rate", "max_drawdown", "profit_factor"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], use_container_width=True, hide_index=True)
