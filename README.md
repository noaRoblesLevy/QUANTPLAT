# QUANTPLAT

Local + web-based quantitative trading platform for writing, optimizing, and backtesting trading algorithms.

## Features

- **Backtesting** via LEAN (QuantConnect open-source engine) with intraday futures data
- **Multi-language strategies** — Python, MT5, C++, Rust
- **Parameter optimization** — grid search, walk-forward, AI-guided (Optuna + LLM)
- **Analytics** — Sharpe, Sortino, max drawdown, Monte Carlo simulations (1000 runs)
- **AI co-pilot** — strategy code review + automatic post-backtest analysis (Ollama local LLM)
- **Obsidian vault** — auto-generated notes after every backtest and optimization run
- **Streamlit web UI** — equity curves, Monte Carlo fan charts, optimizer scatter plots

## Setup

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

## Structure

```
core/          LEAN runner + strategy adapters
optimizer/     Grid search, walk-forward, AI optimizer
analytics/     Metrics, Monte Carlo
ai_agent/      LLM copilot + analyzer
vault_sync/    Obsidian vault writer
ui/            Streamlit web dashboard
strategies/    Your strategy files go here
vault/         Obsidian vault (open with Obsidian)
results/       Backtest output JSON files
data/          Local market data cache
```

## Design Spec

See `docs/superpowers/specs/2026-04-24-quantplat-design.md` for the full architecture and module design.
