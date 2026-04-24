# QUANTPLAT — Platform Design Spec
**Date:** 2026-04-24  
**Status:** Approved

---

## Goal

Build a local + web-based quantitative trading platform that allows writing, optimizing, and backtesting trading algorithms — with the aim of consistently generating alpha over the long term. The platform uses AI to assist with strategy improvement and automated analytics reporting.

---

## Architecture

Modular Python services connected by a shared results store (SQLite + JSON files). No microservices overhead — all modules run in the same process or as subprocesses on localhost. Web UI is served via Streamlit on `localhost:8501`.

```
strategies/  ──►  core/  ──►  results/  ──►  analytics/  ──►  ui/
                                                  │
                                          ai_agent/ + vault_sync/
```

---

## Folder Structure

```
QUANTPLAT/
├── core/
│   ├── lean_runner.py          # triggers LEAN backtest as subprocess
│   ├── strategy_loader.py      # detects language, picks adapter
│   └── adapters/
│       ├── python_adapter.py   # runs directly in LEAN
│       ├── mt5_adapter.py      # MetaTrader5 Python package bridge
│       ├── cpp_adapter.py      # subprocess + JSON interface
│       └── rust_adapter.py     # subprocess + JSON interface
│
├── optimizer/
│   ├── grid_search.py          # brute-force parameter sweep
│   ├── walk_forward.py         # train/test split optimization
│   └── ai_optimizer.py         # Optuna + LLM-guided search
│
├── analytics/
│   ├── metrics.py              # Sharpe, Sortino, drawdown, expectancy, etc.
│   ├── monte_carlo.py          # N-simulation trade reshuffling
│   └── report.py               # generates JSON + chart-ready data
│
├── ai_agent/
│   ├── copilot.py              # strategy code review + suggestions
│   └── analyzer.py             # post-backtest automatic analysis
│
├── vault_sync/
│   └── writer.py               # writes .md notes to Obsidian vault
│
├── ui/
│   ├── app.py                  # Streamlit entrypoint
│   └── pages/
│       ├── backtest.py         # run backtests, view equity curve
│       ├── optimizer.py        # parameter optimization UI
│       ├── analytics.py        # Monte Carlo, stats, charts
│       └── ai_chat.py          # copilot chat + analyzer summaries
│
├── data/                       # local market data cache
├── results/                    # backtest outputs (JSON per run)
├── strategies/                 # user strategy files
│   └── examples/
├── docs/
│   └── superpowers/specs/
├── vault/                      # Obsidian vault
│   ├── 00-Hub.md
│   ├── backtests/
│   ├── strategies/
│   ├── optimizations/
│   └── research/
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Backtesting engine | LEAN (QuantConnect open-source) | Local, free, intraday futures data via QC account |
| Data | QC data via `lean` CLI + yfinance fallback | Free with QC account |
| Web UI | Streamlit | 100% Python, no JS required |
| Charts | Plotly | Interactive, free |
| AI | Ollama (local, free) with optional Claude/OpenAI API | Configured via `.env` |
| Optimization | Optuna | State-of-the-art HPO, free |
| Monte Carlo | NumPy custom | No license needed |
| Database | SQLite | Local, no server |
| MT5 adapter | MetaTrader5 Python package | Free with MT5 install |
| C++/Rust adapters | Subprocess + JSON interface | No extra tooling |

---

## Module Details

### `core/lean_runner.py`
- Accepts: strategy file path, date range, starting capital, resolution
- Starts LEAN as a subprocess with generated config
- Streams stdout for live progress
- Returns: path to results JSON on completion
- Error handling: captures LEAN stderr, surfaces readable error in UI

### `core/strategy_loader.py`
- Detects file extension: `.py` → python_adapter, `.mq5` → mt5_adapter, `.cpp` → cpp_adapter, `.rs` → rust_adapter
- Each adapter exposes a common interface: `run(config) -> results_path`
- MT5/C++/Rust adapters compile/run the strategy and output a standardized JSON results format

### `optimizer/`
- **Grid search:** iterates over all parameter combinations, runs a backtest per combination, stores results in SQLite
- **Walk-forward:** splits date range into N windows, optimizes on in-sample, validates on out-of-sample
- **AI optimizer:** uses Optuna for sampling + calls LLM after each trial batch to suggest promising parameter directions

### `analytics/metrics.py`
Computes per backtest run:
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown, drawdown duration, recovery factor
- Win rate, profit factor, expectancy
- Average win / average loss, profit-loss ratio

### `analytics/monte_carlo.py`
- Reshuffles trade P&L sequence N times (default: 1000)
- Computes equity curve per simulation
- Outputs: 5th / 50th / 95th percentile equity curves, probability of ruin, max drawdown distribution

### `ai_agent/copilot.py`
- User pastes strategy code in UI chat
- Copilot reviews code against backtest metrics and returns concrete suggestions
- Uses Ollama (local) by default; falls back to Claude/OpenAI API if configured

### `ai_agent/analyzer.py`
- Runs automatically after every backtest
- Receives: metrics dict + trade log
- Returns: short summary (what worked, what didn't, recommended next step)
- Output is saved to vault and shown in UI

### `vault_sync/writer.py`
- Triggered after every backtest and optimization run
- Writes to `vault/backtests/YYYY-MM-DD-{strategy_name}.md`
- Template includes: strategy name, date, key metrics table, AI summary, link to results JSON
- Updates `vault/00-Hub.md` index automatically

---

## Web UI Pages

### Backtest page
- File picker for strategy
- Date range selector, starting capital input, resolution dropdown
- "Run" button → live progress bar
- Results: equity curve chart, metrics table, trade list

### Optimizer page
- Mode selector: Grid / Walk-Forward / AI
- Parameter range inputs (min, max, step per parameter)
- Live trial counter
- Results: scatter plot (parameter value vs Sharpe), best config highlighted

### Analytics page
- Equity curve with drawdown overlay
- Monte Carlo fan chart (5th/50th/95th percentile)
- Trade distribution histogram
- Full metrics table with benchmark comparison

### AI Chat page
- Chat interface for copilot
- Automatic analyzer summaries listed per backtest run
- Copy button to paste current strategy code into chat

---

## Obsidian Vault Structure

```
vault/
├── 00-Hub.md                    # auto-updated index of all strategies
├── backtests/
│   └── YYYY-MM-DD-{name}.md     # auto-generated per run
├── strategies/
│   └── {name}.md                # manual strategy documentation
├── optimizations/
│   └── YYYY-MM-DD-{name}.md     # auto-generated per optimization run
└── research/
    └── (user's own notes)
```

---

## Data Flow (end-to-end)

1. User writes strategy in VS Code → saves to `strategies/`
2. User opens Streamlit UI → selects strategy, sets params, clicks Run
3. `core/strategy_loader.py` detects language → picks adapter
4. `core/lean_runner.py` starts LEAN subprocess → streams progress to UI
5. LEAN outputs results to `results/run_{id}.json`
6. `analytics/metrics.py` + `analytics/monte_carlo.py` process results
7. `ai_agent/analyzer.py` generates text summary via LLM
8. `vault_sync/writer.py` writes Obsidian note
9. UI displays equity curve, metrics, Monte Carlo chart, AI summary

---

## Out of Scope (v1)

- Live trading / broker execution
- Multi-user / cloud deployment
- Real-time data streaming
- Portfolio-level optimization (multi-strategy)
