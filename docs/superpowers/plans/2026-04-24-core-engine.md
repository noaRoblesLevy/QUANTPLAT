# Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the QUANTPLAT core engine — LEAN backtest runner, result parser, analytics (Sharpe/Sortino/drawdown/Monte Carlo), and report generator — so that a Python strategy file can be backtested end-to-end and produce a structured results dict.

**Architecture:** `core/strategy_loader.py` detects the strategy language and picks an adapter. `core/lean_runner.py` runs a LEAN backtest as a subprocess using the adapter's generated config. `analytics/` parses the LEAN output JSON into metrics and Monte Carlo data. `analytics/report.py` assembles everything into a single results dict saved to `results/`.

**Tech Stack:** Python 3.11+, LEAN CLI (`pip install lean`), NumPy, SQLAlchemy (SQLite), pytest

---

## File Map

| File | Responsibility |
|---|---|
| `tests/conftest.py` | Shared fixtures (sample equity curve, sample P&L list, sample LEAN output JSON) |
| `tests/test_metrics.py` | Unit tests for all metric functions |
| `tests/test_monte_carlo.py` | Unit tests for Monte Carlo simulation |
| `tests/test_python_adapter.py` | Unit tests for LEAN config generation |
| `tests/test_lean_runner.py` | Unit tests for subprocess runner (mocked) |
| `tests/test_strategy_loader.py` | Unit tests for language detection + adapter selection |
| `tests/test_report.py` | Unit tests for report assembly + SQLite storage |
| `analytics/__init__.py` | Empty |
| `analytics/metrics.py` | Sharpe, Sortino, max drawdown, Calmar, profit factor, win rate, expectancy, `compute_all()` |
| `analytics/monte_carlo.py` | N-simulation equity curve reshuffling, percentile output |
| `analytics/report.py` | Assemble metrics + Monte Carlo into results dict, save to SQLite + JSON |
| `core/__init__.py` | Empty |
| `core/adapters/__init__.py` | Empty |
| `core/adapters/python_adapter.py` | Generate LEAN project folder + config.json for a Python strategy |
| `core/lean_runner.py` | Run `lean backtest` subprocess, stream stdout, return parsed results |
| `core/strategy_loader.py` | Detect `.py`/`.mq5`/`.cpp`/`.rs`, return correct adapter instance |
| `db/models.py` | SQLAlchemy `BacktestRun` model |
| `db/__init__.py` | Engine + session factory |

---

## Task 1: Project Setup

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `analytics/__init__.py`
- Create: `core/__init__.py`
- Create: `core/adapters/__init__.py`
- Create: `db/__init__.py`

- [ ] **Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 2: Create empty __init__.py files**

Run:
```bash
touch tests/__init__.py analytics/__init__.py core/__init__.py core/adapters/__init__.py db/__init__.py
```

- [ ] **Step 3: Create tests/conftest.py**

```python
import pytest
import json

@pytest.fixture
def sample_equity_curve():
    # 20 data points starting at 50000, trending up with noise
    return [
        50000, 50350, 50120, 50800, 51200, 50900, 51500, 51300,
        52000, 51600, 52400, 52100, 53000, 52700, 53500, 53200,
        54000, 53600, 54500, 55000
    ]

@pytest.fixture
def sample_pl_list():
    # 10 trades: 4 winners, 6 losers (matches typical strategy profile)
    return [350.0, -120.0, 680.0, -200.0, -150.0, 1100.0, -300.0, -180.0, 420.0, -90.0]

@pytest.fixture
def sample_lean_output():
    # Minimal structure of a real LEAN backtestResults.json
    return {
        "statistics": {
            "Total Orders": "10",
            "Net Profit": "22.043%",
            "Sharpe Ratio": "0.499",
            "Start Equity": "50000",
            "End Equity": "61021.56",
        },
        "profitLoss": {
            "2025-01-17T18:06:00Z": 353.58,
            "2025-01-21T14:33:00Z": 218.36,
            "2025-01-23T17:33:00Z": -589.12,
            "2025-02-03T14:33:00Z": 785.36,
            "2025-02-10T18:54:00Z": 317.44,
        },
        "charts": {
            "Strategy Equity": {
                "series": {
                    "Equity": {
                        "values": [
                            {"x": 1737000000, "y": 50000.0},
                            {"x": 1737100000, "y": 50353.58},
                            {"x": 1737200000, "y": 50571.94},
                            {"x": 1737300000, "y": 49982.82},
                            {"x": 1737400000, "y": 50768.18},
                            {"x": 1737500000, "y": 51085.62},
                        ]
                    }
                }
            }
        },
        "runtimeStatistics": {
            "Equity": "$61,021.56",
            "Net Profit": "$11,021.56",
            "Return": "22.04 %",
        }
    }
```

- [ ] **Step 4: Verify pytest finds tests (no tests yet, should show 0 collected)**

Run:
```bash
pytest --collect-only
```
Expected: `no tests ran` or `0 items`

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/ analytics/__init__.py core/__init__.py core/adapters/__init__.py db/__init__.py
git commit -m "feat: project test setup and fixtures"
```

---

## Task 2: Analytics — Metrics

**Files:**
- Create: `analytics/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics.py`:

```python
import pytest
import numpy as np
from analytics.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown,
    calmar_ratio, profit_factor, win_rate, expectancy, compute_all
)


def test_sharpe_ratio_positive_returns():
    returns = [0.01, 0.02, -0.005, 0.015, 0.008]
    result = sharpe_ratio(returns)
    assert result > 0

def test_sharpe_ratio_zero_std_returns_zero():
    returns = [0.01, 0.01, 0.01, 0.01]
    assert sharpe_ratio(returns) == 0.0

def test_sortino_ratio_no_losses_returns_zero():
    returns = [0.01, 0.02, 0.005, 0.015]
    # All positive returns → no downside deviation → return 0.0
    assert sortino_ratio(returns) == 0.0

def test_sortino_ratio_with_losses():
    returns = [0.01, -0.02, 0.005, -0.015, 0.008]
    result = sortino_ratio(returns)
    assert isinstance(result, float)

def test_max_drawdown_returns_negative():
    equity = [100, 110, 105, 95, 100, 115]
    result = max_drawdown(equity)
    assert result < 0
    # Peak is 110, trough is 95 → drawdown = (95-110)/110 ≈ -0.136
    assert abs(result - (-0.1364)) < 0.001

def test_max_drawdown_no_drawdown():
    equity = [100, 110, 120, 130]
    assert max_drawdown(equity) == 0.0

def test_profit_factor_basic(sample_pl_list):
    result = profit_factor(sample_pl_list)
    wins = 350 + 680 + 1100 + 420
    losses = 120 + 200 + 150 + 300 + 180 + 90
    assert abs(result - wins / losses) < 0.001

def test_profit_factor_no_losses():
    assert profit_factor([100.0, 200.0]) == float('inf')

def test_profit_factor_no_wins():
    assert profit_factor([-100.0, -200.0]) == 0.0

def test_win_rate_basic(sample_pl_list):
    result = win_rate(sample_pl_list)
    assert abs(result - 0.4) < 0.001  # 4 wins out of 10

def test_win_rate_empty():
    assert win_rate([]) == 0.0

def test_expectancy_basic(sample_pl_list):
    result = expectancy(sample_pl_list)
    assert isinstance(result, float)
    # 4 wins avg = (350+680+1100+420)/4 = 637.5
    # 6 losses avg = (120+200+150+300+180+90)/6 = 173.3
    # expectancy = 0.4*637.5 - 0.6*173.3 = 255 - 103.98 = 151.02
    assert abs(result - 151.02) < 1.0

def test_compute_all_returns_all_keys(sample_equity_curve, sample_pl_list):
    result = compute_all(sample_equity_curve, sample_pl_list)
    expected_keys = [
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
        "profit_factor", "win_rate", "expectancy", "annual_return",
        "total_trades", "avg_win", "avg_loss"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

def test_compute_all_total_trades(sample_equity_curve, sample_pl_list):
    result = compute_all(sample_equity_curve, sample_pl_list)
    assert result["total_trades"] == len(sample_pl_list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```
Expected: `ModuleNotFoundError: No module named 'analytics.metrics'`

- [ ] **Step 3: Implement analytics/metrics.py**

```python
import numpy as np
from typing import List


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    arr = np.array(returns, dtype=float)
    if arr.std() == 0:
        return 0.0
    excess = arr - risk_free_rate / 252
    return float(np.sqrt(252) * excess.mean() / arr.std())


def sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate / 252
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / downside_std)


def max_drawdown(equity_curve: List[float]) -> float:
    arr = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    with np.errstate(invalid='ignore'):
        dd = np.where(peak > 0, (arr - peak) / peak, 0.0)
    return float(dd.min())


def calmar_ratio(equity_curve: List[float], annual_return: float) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(annual_return / mdd)


def profit_factor(pl_list: List[float]) -> float:
    wins = sum(p for p in pl_list if p > 0)
    losses = abs(sum(p for p in pl_list if p < 0))
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(pl_list: List[float]) -> float:
    if not pl_list:
        return 0.0
    return float(sum(1 for p in pl_list if p > 0) / len(pl_list))


def expectancy(pl_list: List[float]) -> float:
    if not pl_list:
        return 0.0
    wr = win_rate(pl_list)
    wins = [p for p in pl_list if p > 0]
    losses = [p for p in pl_list if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(abs(np.mean(losses))) if losses else 0.0
    return float(wr * avg_win - (1 - wr) * avg_loss)


def compute_all(equity_curve: List[float], pl_list: List[float],
                risk_free_rate: float = 0.0) -> dict:
    arr = np.array(equity_curve, dtype=float)
    returns = (np.diff(arr) / arr[:-1]).tolist()
    n_days = len(returns)
    annual_return = float((arr[-1] / arr[0]) ** (252 / max(n_days, 1)) - 1)
    wins = [p for p in pl_list if p > 0]
    losses = [p for p in pl_list if p < 0]
    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve, annual_return),
        "profit_factor": profit_factor(pl_list),
        "win_rate": win_rate(pl_list),
        "expectancy": expectancy(pl_list),
        "annual_return": annual_return,
        "total_trades": len(pl_list),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add analytics/metrics.py tests/test_metrics.py
git commit -m "feat: analytics metrics (Sharpe, Sortino, drawdown, expectancy)"
```

---

## Task 3: Analytics — Monte Carlo

**Files:**
- Create: `analytics/monte_carlo.py`
- Create: `tests/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_monte_carlo.py`:

```python
import pytest
from analytics.monte_carlo import run


def test_run_returns_required_keys(sample_pl_list):
    result = run(sample_pl_list, n_simulations=100)
    assert "percentiles" in result
    assert "final_equity" in result
    assert "max_drawdown" in result
    assert "probability_of_ruin" in result
    assert "n_simulations" in result
    assert "n_trades" in result

def test_run_percentile_keys(sample_pl_list):
    result = run(sample_pl_list, n_simulations=100)
    assert "p5" in result["percentiles"]
    assert "p50" in result["percentiles"]
    assert "p95" in result["percentiles"]

def test_run_percentile_ordering(sample_pl_list):
    result = run(sample_pl_list, n_simulations=500)
    p5_end = result["percentiles"]["p5"][-1]
    p50_end = result["percentiles"]["p50"][-1]
    p95_end = result["percentiles"]["p95"][-1]
    assert p5_end <= p50_end <= p95_end

def test_run_equity_curves_start_at_starting_equity(sample_pl_list):
    starting = 75000.0
    result = run(sample_pl_list, n_simulations=100, starting_equity=starting)
    assert result["percentiles"]["p5"][0] == starting
    assert result["percentiles"]["p50"][0] == starting
    assert result["percentiles"]["p95"][0] == starting

def test_run_n_trades_correct(sample_pl_list):
    result = run(sample_pl_list, n_simulations=100)
    assert result["n_trades"] == len(sample_pl_list)

def test_run_equity_curve_length(sample_pl_list):
    result = run(sample_pl_list, n_simulations=100)
    # n_trades + 1 (starting point)
    assert len(result["percentiles"]["p50"]) == len(sample_pl_list) + 1

def test_probability_of_ruin_all_winners():
    # All winning trades → ruin probability should be 0
    pl = [100.0] * 20
    result = run(pl, n_simulations=200)
    assert result["probability_of_ruin"] == 0.0

def test_probability_of_ruin_all_losers():
    # All losing trades → ruin probability should be 1.0
    pl = [-5000.0] * 20
    result = run(pl, n_simulations=200, starting_equity=50000.0)
    assert result["probability_of_ruin"] == 1.0

def test_run_deterministic_with_seed(sample_pl_list):
    r1 = run(sample_pl_list, n_simulations=100)
    r2 = run(sample_pl_list, n_simulations=100)
    assert r1["final_equity"]["p50"] == r2["final_equity"]["p50"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_monte_carlo.py -v
```
Expected: `ModuleNotFoundError: No module named 'analytics.monte_carlo'`

- [ ] **Step 3: Implement analytics/monte_carlo.py**

```python
import numpy as np
from typing import List, Dict


def run(pl_list: List[float], n_simulations: int = 1000,
        starting_equity: float = 50000.0) -> Dict:
    rng = np.random.default_rng(42)
    pl_arr = np.array(pl_list, dtype=float)
    n_trades = len(pl_arr)

    sim_equity = np.empty((n_simulations, n_trades + 1))
    sim_equity[:, 0] = starting_equity
    for i in range(n_simulations):
        shuffled = rng.permutation(pl_arr)
        sim_equity[i, 1:] = starting_equity + np.cumsum(shuffled)

    final_equities = sim_equity[:, -1]
    sorted_idx = np.argsort(final_equities)

    max_drawdowns = np.array([_max_drawdown(sim_equity[i]) for i in range(n_simulations)])
    probability_of_ruin = float(np.mean(final_equities < starting_equity * 0.5))

    return {
        "percentiles": {
            "p5":  sim_equity[sorted_idx[int(0.05 * n_simulations)]].tolist(),
            "p50": sim_equity[sorted_idx[int(0.50 * n_simulations)]].tolist(),
            "p95": sim_equity[sorted_idx[int(0.95 * n_simulations)]].tolist(),
        },
        "final_equity": {
            "mean": float(final_equities.mean()),
            "std":  float(final_equities.std()),
            "p5":   float(np.percentile(final_equities, 5)),
            "p50":  float(np.percentile(final_equities, 50)),
            "p95":  float(np.percentile(final_equities, 95)),
        },
        "max_drawdown": {
            "mean": float(max_drawdowns.mean()),
            "p95":  float(np.percentile(max_drawdowns, 95)),
        },
        "probability_of_ruin": probability_of_ruin,
        "n_simulations": n_simulations,
        "n_trades": n_trades,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    with np.errstate(invalid='ignore'):
        dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    return float(dd.min())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_monte_carlo.py -v
```
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add analytics/monte_carlo.py tests/test_monte_carlo.py
git commit -m "feat: Monte Carlo simulation (1000-run equity reshuffling)"
```

---

## Task 4: Core — Python Adapter

**Files:**
- Create: `core/adapters/python_adapter.py`
- Create: `tests/test_python_adapter.py`

The Python adapter creates a temporary LEAN project folder from a strategy `.py` file, writes a `config.json`, and returns the project path. LEAN CLI expects this structure to run a backtest.

- [ ] **Step 1: Write failing tests**

Create `tests/test_python_adapter.py`:

```python
import json
import pytest
from pathlib import Path
import tempfile
from core.adapters.python_adapter import PythonAdapter


@pytest.fixture
def sample_strategy_file(tmp_path):
    f = tmp_path / "my_strategy.py"
    f.write_text("# dummy strategy\nclass MyAlgo: pass\n")
    return f


def test_prepare_creates_project_dir(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    assert project_dir.is_dir()


def test_prepare_copies_strategy_as_main(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    main_file = project_dir / "main.py"
    assert main_file.exists()
    assert "dummy strategy" in main_file.read_text()


def test_prepare_creates_config_json(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    config_file = project_dir / "config.json"
    assert config_file.exists()
    config = json.loads(config_file.read_text())
    assert config["algorithm-language"] == "Python"
    assert "parameters" in config


def test_prepare_config_includes_strategy_name(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    config = json.loads((project_dir / "config.json").read_text())
    assert config["algorithm-name"] == "my_strategy"


def test_cleanup_removes_project_dir(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    assert project_dir.exists()
    adapter.cleanup()
    assert not project_dir.exists()


def test_prepare_with_parameters(sample_strategy_file):
    adapter = PythonAdapter()
    params = {"fast_ema": 30, "slow_ema": 200}
    project_dir = adapter.prepare(sample_strategy_file, parameters=params)
    config = json.loads((project_dir / "config.json").read_text())
    assert config["parameters"]["fast_ema"] == 30
    assert config["parameters"]["slow_ema"] == 200
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_python_adapter.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.adapters.python_adapter'`

- [ ] **Step 3: Implement core/adapters/python_adapter.py**

```python
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any


class PythonAdapter:
    def __init__(self):
        self._project_dir: Optional[Path] = None

    def prepare(self, strategy_path: Path,
                parameters: Optional[Dict[str, Any]] = None) -> Path:
        strategy_path = Path(strategy_path)
        self._project_dir = Path(tempfile.mkdtemp(prefix="quantplat_"))
        main_file = self._project_dir / "main.py"
        shutil.copy2(strategy_path, main_file)
        config = {
            "algorithm-language": "Python",
            "algorithm-name": strategy_path.stem,
            "description": "",
            "parameters": parameters or {},
            "cloud-id": 0,
            "local-id": 0,
        }
        (self._project_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        return self._project_dir

    def cleanup(self) -> None:
        if self._project_dir and self._project_dir.exists():
            shutil.rmtree(self._project_dir)
            self._project_dir = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_python_adapter.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add core/adapters/python_adapter.py tests/test_python_adapter.py
git commit -m "feat: Python strategy adapter for LEAN project generation"
```

---

## Task 5: Core — LEAN Runner

**Files:**
- Create: `core/lean_runner.py`
- Create: `tests/test_lean_runner.py`

The runner calls `lean backtest <project_dir>`, streams stdout to a callback, then finds and parses the output `backtestResults.json`. Subprocess is injected so tests can mock it without calling real LEAN.

- [ ] **Step 1: Write failing tests**

Create `tests/test_lean_runner.py`:

```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from core.lean_runner import LeanRunner, LeanRunError


@pytest.fixture
def mock_project_dir(tmp_path):
    project = tmp_path / "my_strategy"
    project.mkdir()
    (project / "main.py").write_text("# algo")
    (project / "config.json").write_text('{"algorithm-name": "my_strategy"}')
    return project


@pytest.fixture
def mock_backtest_output(tmp_path, sample_lean_output):
    # Simulates LEAN writing results to backtests/{timestamp}/backtestResults.json
    output_dir = tmp_path / "backtests" / "20250101_120000"
    output_dir.mkdir(parents=True)
    result_file = output_dir / "backtestResults.json"
    result_file.write_text(json.dumps(sample_lean_output))
    return tmp_path


def test_run_calls_lean_subprocess(mock_project_dir, mock_backtest_output, sample_lean_output):
    runner = LeanRunner(lean_workspace=mock_backtest_output)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter([b"Launching LEAN...\n", b"Backtest complete.\n"]))
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        result = runner.run(mock_project_dir)
    mock_popen.assert_called_once()
    args = mock_popen.call_args[0][0]
    assert args[0] == "lean"
    assert args[1] == "backtest"


def test_run_returns_parsed_results(mock_project_dir, mock_backtest_output, sample_lean_output):
    runner = LeanRunner(lean_workspace=mock_backtest_output)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter([]))
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        result = runner.run(mock_project_dir)
    assert "pl_list" in result
    assert "equity_curve" in result
    assert "raw_statistics" in result
    assert "results_path" in result


def test_run_raises_on_nonzero_exit(mock_project_dir, tmp_path):
    runner = LeanRunner(lean_workspace=tmp_path)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter([b"Error: strategy failed\n"]))
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        with pytest.raises(LeanRunError):
            runner.run(mock_project_dir)


def test_run_streams_output_to_callback(mock_project_dir, mock_backtest_output):
    runner = LeanRunner(lean_workspace=mock_backtest_output)
    received_lines = []
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(
            return_value=iter([b"line one\n", b"line two\n"])
        )
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        runner.run(mock_project_dir, on_output=received_lines.append)
    assert "line one\n" in received_lines
    assert "line two\n" in received_lines


def test_parse_lean_output_extracts_pl_list(sample_lean_output):
    runner = LeanRunner()
    result = runner._parse_lean_output(sample_lean_output, results_path=Path("."))
    assert isinstance(result["pl_list"], list)
    assert len(result["pl_list"]) == len(sample_lean_output["profitLoss"])

def test_parse_lean_output_extracts_equity_curve(sample_lean_output):
    runner = LeanRunner()
    result = runner._parse_lean_output(sample_lean_output, results_path=Path("."))
    assert isinstance(result["equity_curve"], list)
    assert len(result["equity_curve"]) > 0
    assert result["equity_curve"][0] == 50000.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_lean_runner.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.lean_runner'`

- [ ] **Step 3: Implement core/lean_runner.py**

```python
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, Any, Optional


class LeanRunError(Exception):
    pass


class LeanRunner:
    def __init__(self, lean_workspace: Optional[Path] = None):
        self._workspace = Path(lean_workspace) if lean_workspace else Path.cwd()

    def run(self, project_dir: Path,
            on_output: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        project_dir = Path(project_dir)
        cmd = ["lean", "backtest", str(project_dir)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self._workspace),
        )
        output_lines = []
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            output_lines.append(line)
            if on_output:
                on_output(line)
        proc.wait()
        if proc.returncode != 0:
            raise LeanRunError(
                f"LEAN exited with code {proc.returncode}.\n" + "".join(output_lines)
            )
        results_file = self._find_results_file()
        raw = json.loads(results_file.read_text(encoding="utf-8"))
        return self._parse_lean_output(raw, results_path=results_file)

    def _find_results_file(self) -> Path:
        backtest_dirs = sorted(
            (self._workspace / "backtests").glob("*/backtestResults.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not backtest_dirs:
            raise LeanRunError("No backtestResults.json found in workspace/backtests/")
        return backtest_dirs[0]

    def _parse_lean_output(self, raw: Dict, results_path: Path) -> Dict[str, Any]:
        pl_list = list(raw.get("profitLoss", {}).values())
        equity_points = (
            raw.get("charts", {})
            .get("Strategy Equity", {})
            .get("series", {})
            .get("Equity", {})
            .get("values", [])
        )
        equity_curve = [pt["y"] for pt in equity_points] if equity_points else []
        return {
            "pl_list": pl_list,
            "equity_curve": equity_curve,
            "raw_statistics": raw.get("statistics", {}),
            "runtime_statistics": raw.get("runtimeStatistics", {}),
            "results_path": str(results_path),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_lean_runner.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add core/lean_runner.py tests/test_lean_runner.py
git commit -m "feat: LEAN subprocess runner with output streaming and result parsing"
```

---

## Task 6: Core — Strategy Loader

**Files:**
- Create: `core/strategy_loader.py`
- Create: `tests/test_strategy_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_strategy_loader.py`:

```python
import pytest
from pathlib import Path
from core.strategy_loader import StrategyLoader
from core.adapters.python_adapter import PythonAdapter


@pytest.fixture
def loader():
    return StrategyLoader()


def test_load_python_strategy_returns_python_adapter(tmp_path, loader):
    f = tmp_path / "my_algo.py"
    f.write_text("# algo")
    adapter = loader.load(f)
    assert isinstance(adapter, PythonAdapter)


def test_load_detects_by_extension(tmp_path, loader):
    f = tmp_path / "algo.py"
    f.write_text("# algo")
    adapter = loader.load(f)
    assert adapter is not None


def test_load_raises_for_unsupported_extension(tmp_path, loader):
    f = tmp_path / "algo.java"
    f.write_text("// java")
    with pytest.raises(ValueError, match="Unsupported strategy language"):
        loader.load(f)


def test_load_raises_if_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load(Path("/nonexistent/strategy.py"))


def test_load_mt5_not_yet_implemented(tmp_path, loader):
    f = tmp_path / "algo.mq5"
    f.write_text("// mt5")
    with pytest.raises(NotImplementedError, match="MT5"):
        loader.load(f)


def test_load_cpp_not_yet_implemented(tmp_path, loader):
    f = tmp_path / "algo.cpp"
    f.write_text("// cpp")
    with pytest.raises(NotImplementedError, match="C++"):
        loader.load(f)


def test_load_rust_not_yet_implemented(tmp_path, loader):
    f = tmp_path / "algo.rs"
    f.write_text("// rust")
    with pytest.raises(NotImplementedError, match="Rust"):
        loader.load(f)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_strategy_loader.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.strategy_loader'`

- [ ] **Step 3: Implement core/strategy_loader.py**

```python
from pathlib import Path
from core.adapters.python_adapter import PythonAdapter

_LANGUAGE_MAP = {
    ".py":  "python",
    ".mq5": "mt5",
    ".cpp": "cpp",
    ".rs":  "rust",
}


class StrategyLoader:
    def load(self, strategy_path: Path):
        strategy_path = Path(strategy_path)
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
        ext = strategy_path.suffix.lower()
        language = _LANGUAGE_MAP.get(ext)
        if language is None:
            raise ValueError(
                f"Unsupported strategy language: '{ext}'. "
                f"Supported: {list(_LANGUAGE_MAP.keys())}"
            )
        if language == "python":
            return PythonAdapter()
        if language == "mt5":
            raise NotImplementedError("MT5 adapter not yet implemented")
        if language == "cpp":
            raise NotImplementedError("C++ adapter not yet implemented")
        if language == "rust":
            raise NotImplementedError("Rust adapter not yet implemented")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_strategy_loader.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add core/strategy_loader.py tests/test_strategy_loader.py
git commit -m "feat: strategy loader with language detection (Python active, MT5/C++/Rust stubs)"
```

---

## Task 7: Database — BacktestRun Model

**Files:**
- Create: `db/__init__.py`
- Create: `db/models.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_db.py`:

```python
import pytest
from pathlib import Path
from db import get_session, init_db
from db.models import BacktestRun
from sqlalchemy import text


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = init_db(f"sqlite:///{db_path}")
    with get_session(engine) as session:
        yield session


def test_create_backtest_run(db_session):
    run = BacktestRun(
        strategy_name="my_algo",
        strategy_path="strategies/my_algo.py",
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        max_drawdown=-0.12,
        profit_factor=1.5,
        win_rate=0.45,
        expectancy=150.0,
        annual_return=0.22,
        total_trades=100,
        results_path="results/run_001.json",
    )
    db_session.add(run)
    db_session.commit()
    assert run.id is not None


def test_query_backtest_run(db_session):
    run = BacktestRun(
        strategy_name="test_strat",
        strategy_path="strategies/test.py",
        sharpe_ratio=0.8,
        sortino_ratio=1.1,
        max_drawdown=-0.20,
        profit_factor=1.2,
        win_rate=0.40,
        expectancy=80.0,
        annual_return=0.10,
        total_trades=50,
        results_path="results/run_002.json",
    )
    db_session.add(run)
    db_session.commit()
    found = db_session.query(BacktestRun).filter_by(strategy_name="test_strat").first()
    assert found is not None
    assert found.sharpe_ratio == 0.8


def test_backtest_run_created_at_set_automatically(db_session):
    run = BacktestRun(
        strategy_name="ts",
        strategy_path="s.py",
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        expectancy=0.0,
        annual_return=0.0,
        total_trades=0,
        results_path="r.json",
    )
    db_session.add(run)
    db_session.commit()
    assert run.created_at is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```
Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Implement db/models.py**

```python
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name  = Column(String, nullable=False)
    strategy_path  = Column(String, nullable=False)
    sharpe_ratio   = Column(Float, nullable=False)
    sortino_ratio  = Column(Float, nullable=False)
    max_drawdown   = Column(Float, nullable=False)
    profit_factor  = Column(Float, nullable=False)
    win_rate       = Column(Float, nullable=False)
    expectancy     = Column(Float, nullable=False)
    annual_return  = Column(Float, nullable=False)
    total_trades   = Column(Integer, nullable=False)
    results_path   = Column(String, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)
```

- [ ] **Step 4: Implement db/__init__.py**

```python
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from db.models import Base

_DEFAULT_URL = "sqlite:///quantplat.db"


def init_db(database_url: str = _DEFAULT_URL):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(engine=None):
    if engine is None:
        engine = init_db()
    factory = sessionmaker(bind=engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add db/ tests/test_db.py
git commit -m "feat: SQLite BacktestRun model and session factory"
```

---

## Task 8: Analytics — Report Generator

**Files:**
- Create: `analytics/report.py`
- Create: `tests/test_report.py`

The report generator combines LEAN runner output, metrics, and Monte Carlo into a single results dict, saves it as JSON to `results/`, and writes a row to the SQLite database.

- [ ] **Step 1: Write failing tests**

Create `tests/test_report.py`:

```python
import json
import pytest
from pathlib import Path
from analytics.report import ReportGenerator
from db import init_db


@pytest.fixture
def engine(tmp_path):
    return init_db(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def lean_output(sample_equity_curve, sample_pl_list):
    return {
        "pl_list": sample_pl_list,
        "equity_curve": sample_equity_curve,
        "raw_statistics": {"Total Orders": "10", "Net Profit": "10.0%"},
        "runtime_statistics": {"Return": "10.0 %"},
        "results_path": "results/run_001.json",
    }


def test_generate_returns_metrics(lean_output, results_dir, engine):
    gen = ReportGenerator(results_dir=results_dir, engine=engine)
    report = gen.generate("my_algo", "strategies/my_algo.py", lean_output, n_mc=50)
    assert "metrics" in report
    assert "sharpe_ratio" in report["metrics"]
    assert "max_drawdown" in report["metrics"]


def test_generate_includes_monte_carlo(lean_output, results_dir, engine):
    gen = ReportGenerator(results_dir=results_dir, engine=engine)
    report = gen.generate("my_algo", "strategies/my_algo.py", lean_output, n_mc=50)
    assert "monte_carlo" in report
    assert "percentiles" in report["monte_carlo"]


def test_generate_saves_json_to_results_dir(lean_output, results_dir, engine):
    gen = ReportGenerator(results_dir=results_dir, engine=engine)
    report = gen.generate("my_algo", "strategies/my_algo.py", lean_output, n_mc=50)
    saved_file = Path(report["report_path"])
    assert saved_file.exists()
    loaded = json.loads(saved_file.read_text())
    assert "metrics" in loaded


def test_generate_saves_to_db(lean_output, results_dir, engine):
    from db import get_session
    from db.models import BacktestRun
    gen = ReportGenerator(results_dir=results_dir, engine=engine)
    gen.generate("saved_algo", "strategies/saved_algo.py", lean_output, n_mc=50)
    with get_session(engine) as session:
        run = session.query(BacktestRun).filter_by(strategy_name="saved_algo").first()
    assert run is not None
    assert run.total_trades == len(lean_output["pl_list"])


def test_generate_report_path_contains_strategy_name(lean_output, results_dir, engine):
    gen = ReportGenerator(results_dir=results_dir, engine=engine)
    report = gen.generate("named_strategy", "strategies/ns.py", lean_output, n_mc=50)
    assert "named_strategy" in report["report_path"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_report.py -v
```
Expected: `ModuleNotFoundError: No module named 'analytics.report'`

- [ ] **Step 3: Implement analytics/report.py**

```python
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from analytics.metrics import compute_all
from analytics.monte_carlo import run as mc_run
from db import get_session, init_db
from db.models import BacktestRun


class ReportGenerator:
    def __init__(self, results_dir: Optional[Path] = None, engine=None):
        self._results_dir = Path(results_dir) if results_dir else Path("results")
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._engine = engine or init_db()

    def generate(self, strategy_name: str, strategy_path: str,
                 lean_output: Dict[str, Any], n_mc: int = 1000) -> Dict[str, Any]:
        pl_list = lean_output["pl_list"]
        equity_curve = lean_output["equity_curve"]

        starting_equity = equity_curve[0] if equity_curve else 50000.0
        metrics = compute_all(equity_curve, pl_list) if equity_curve and pl_list else {}
        mc = mc_run(pl_list, n_simulations=n_mc, starting_equity=starting_equity) if pl_list else {}

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{strategy_name}.json"
        report_path = self._results_dir / filename

        report = {
            "strategy_name": strategy_name,
            "strategy_path": strategy_path,
            "timestamp": timestamp,
            "metrics": metrics,
            "monte_carlo": mc,
            "raw_statistics": lean_output.get("raw_statistics", {}),
            "runtime_statistics": lean_output.get("runtime_statistics", {}),
            "lean_results_path": lean_output.get("results_path", ""),
            "report_path": str(report_path),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._save_to_db(strategy_name, strategy_path, metrics, str(report_path))
        return report

    def _save_to_db(self, strategy_name: str, strategy_path: str,
                    metrics: Dict[str, Any], report_path: str) -> None:
        run = BacktestRun(
            strategy_name=strategy_name,
            strategy_path=strategy_path,
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            expectancy=metrics.get("expectancy", 0.0),
            annual_return=metrics.get("annual_return", 0.0),
            total_trades=metrics.get("total_trades", 0),
            results_path=report_path,
        )
        with get_session(self._engine) as session:
            session.add(run)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_report.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Run the full test suite to make sure nothing broke**

```bash
pytest -v
```
Expected: all tests pass (37+ tests)

- [ ] **Step 6: Commit**

```bash
git add analytics/report.py tests/test_report.py
git commit -m "feat: report generator — metrics + Monte Carlo + SQLite + JSON output"
```

---

## Task 9: Push to GitHub

- [ ] **Step 1: Verify all tests pass one final time**

```bash
pytest -v
```
Expected: all green

- [ ] **Step 2: Push**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** LEAN runner ✓, Python adapter ✓, analytics/metrics ✓, Monte Carlo ✓, report ✓, SQLite storage ✓, strategy loader ✓. MT5/C++/Rust adapters are stubs — covered by `NotImplementedError` tests, implemented in Plan 4.
- **Placeholders:** None — all steps contain complete code.
- **Type consistency:** `compute_all()` returns a dict with keys used verbatim in `ReportGenerator._save_to_db()` and `BacktestRun` columns. `LeanRunner._parse_lean_output()` returns keys `pl_list`, `equity_curve`, `raw_statistics`, `runtime_statistics`, `results_path` — all consumed correctly in `ReportGenerator.generate()`.
- **Out of scope for this plan:** Streamlit UI (Plan 2), optimizer (Plan 3), AI agent + vault (Plan 4).
