# Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three parameter optimizers (grid search, walk-forward, AI-guided Optuna) with a shared LLM provider, DB storage for trials, and a Streamlit UI page for running and reviewing optimizations.

**Architecture:** Each optimizer takes a strategy path + parameter spec, runs backtests via the existing `core/` stack, stores results in two new SQLite tables (`optimization_runs`, `optimization_trials`), and returns an `OptimizationRun` ORM object. The LLM provider is a thin wrapper around Ollama/Claude/OpenAI configured via `.env`. The Streamlit page wires everything together with live progress reporting.

**Tech Stack:** Optuna >= 3.6, Ollama (local default), existing `core/`, `analytics/`, `db/` modules.

---

## File Map

| File | Responsibility |
|---|---|
| `db/models.py` | Add `OptimizationRun` + `OptimizationTrial` models |
| `ui/data_loader.py` | Add `load_all_optimization_runs()` + `load_optimization_trials()` |
| `tests/test_optimization_models.py` | DB integration tests for new models |
| `tests/test_data_loader.py` | Add tests for new loader functions |
| `ai_agent/__init__.py` | Empty |
| `ai_agent/provider.py` | `LLMProvider.call(prompt) -> str`, reads `AI_PROVIDER` from env |
| `tests/test_llm_provider.py` | Unit tests for LLMProvider (mock network calls) |
| `optimizer/__init__.py` | Empty |
| `optimizer/grid_search.py` | `GridSearchOptimizer.run(strategy_path, param_grid, ...) -> OptimizationRun` |
| `tests/test_grid_search.py` | Unit tests with mocked `LeanRunner` + `ReportGenerator` |
| `optimizer/walk_forward.py` | `WalkForwardOptimizer.run(...)` + `_split_windows()` helper |
| `tests/test_walk_forward.py` | Unit tests for window splitting + mocked backtest runs |
| `optimizer/ai_optimizer.py` | `AIOptimizer.run(...)` — Optuna study + LLM suggestions every N trials |
| `tests/test_ai_optimizer.py` | Unit tests with mocked runner + mocked LLM |
| `ui/chart_builder.py` | Add `optimizer_scatter_chart(param_values, sharpe_values, ...)` |
| `tests/test_chart_builder.py` | Add scatter chart tests |
| `ui/pages/4_Optimizer.py` | Streamlit optimizer UI — mode selector, params, live progress, results |

---

## Task 1: DB Models + Data Loader Extensions

**Files:**
- Modify: `db/models.py`
- Modify: `ui/data_loader.py`
- Create: `tests/test_optimization_models.py`
- Modify: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing tests for new DB models**

Create `C:\Users\noaro\QUANTPLAT\tests\test_optimization_models.py`:

```python
import json
import pytest
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial


@pytest.fixture
def engine(tmp_path):
    return init_db(f"sqlite:///{tmp_path}/test.db")


def test_create_optimization_run(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="my_algo",
            strategy_path="strategies/my_algo.py",
            mode="grid",
            n_trials=4,
        )
        session.add(run)
        session.flush()
        assert run.id is not None
        assert run.best_sharpe is None
        assert run.best_params is None
        assert run.created_at is not None


def test_create_optimization_trial(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="grid",
            n_trials=2,
        )
        session.add(run)
        session.flush()
        trial = OptimizationTrial(
            run_id=run.id,
            trial_number=0,
            params=json.dumps({"sma_fast": 10, "sma_slow": 50}),
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.15,
            win_rate=0.55,
            profit_factor=1.8,
        )
        session.add(trial)
        session.flush()
        assert trial.id is not None
        assert trial.run_id == run.id


def test_optimization_trial_nullable_metrics(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="ai",
            n_trials=1,
        )
        session.add(run)
        session.flush()
        trial = OptimizationTrial(
            run_id=run.id,
            trial_number=0,
            params=json.dumps({"sma_fast": 5}),
        )
        session.add(trial)
        session.flush()
        assert trial.sharpe_ratio is None
        assert trial.win_rate is None


def test_optimization_run_update_best(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="grid",
            n_trials=4,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    with get_session(engine) as session:
        run = session.get(OptimizationRun, run_id)
        run.best_sharpe = 1.5
        run.best_params = json.dumps({"sma_fast": 10, "sma_slow": 50})

    with get_session(engine) as session:
        run = session.get(OptimizationRun, run_id)
        assert run.best_sharpe == 1.5
        assert json.loads(run.best_params)["sma_fast"] == 10


def test_multiple_trials_per_run(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="grid",
            n_trials=3,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        for i in range(3):
            session.add(OptimizationTrial(
                run_id=run_id,
                trial_number=i,
                params=json.dumps({"sma_fast": i * 5 + 5}),
                sharpe_ratio=float(i),
            ))

    with get_session(engine) as session:
        from sqlalchemy import select
        from db.models import OptimizationTrial
        trials = session.execute(
            select(OptimizationTrial).where(OptimizationTrial.run_id == run_id)
        ).scalars().all()
        assert len(trials) == 3


def test_walk_forward_trial_window_label(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="walk_forward",
            n_trials=2,
        )
        session.add(run)
        session.flush()
        trial = OptimizationTrial(
            run_id=run.id,
            trial_number=0,
            params=json.dumps({"sma_fast": 10}),
            sharpe_ratio=0.9,
            window_label="window_0_train",
        )
        session.add(trial)
        session.flush()
        assert trial.window_label == "window_0_train"
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
cd C:\Users\noaro\QUANTPLAT
pytest tests/test_optimization_models.py -v
```
Expected: `ImportError` or `sqlalchemy.exc.NoInspectionAvailable` — `OptimizationRun` does not exist yet.

- [ ] **Step 3: Implement new models in `db/models.py`**

Replace the full content of `C:\Users\noaro\QUANTPLAT\db\models.py`:

```python
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
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


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name  = Column(String, nullable=False)
    strategy_path  = Column(String, nullable=False)
    mode           = Column(String, nullable=False)   # "grid", "walk_forward", "ai"
    n_trials       = Column(Integer, nullable=False, default=0)
    best_sharpe    = Column(Float, nullable=True)
    best_params    = Column(String, nullable=True)    # JSON-encoded dict
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class OptimizationTrial(Base):
    __tablename__ = "optimization_trials"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    trial_number   = Column(Integer, nullable=False)
    params         = Column(String, nullable=False)   # JSON-encoded dict
    sharpe_ratio   = Column(Float, nullable=True)
    sortino_ratio  = Column(Float, nullable=True)
    max_drawdown   = Column(Float, nullable=True)
    win_rate       = Column(Float, nullable=True)
    profit_factor  = Column(Float, nullable=True)
    window_label   = Column(String, nullable=True)    # walk-forward only: "window_N_train/test"
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)
```

- [ ] **Step 4: Run model tests — all must pass**

```bash
pytest tests/test_optimization_models.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Write failing tests for data loader extensions**

Append to `C:\Users\noaro\QUANTPLAT\tests\test_data_loader.py`:

```python
from db.models import OptimizationRun, OptimizationTrial
from ui.data_loader import load_all_optimization_runs, load_optimization_trials


@pytest.fixture
def engine_with_opt_runs(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/opt.db")
    with get_session(engine) as session:
        for i in range(3):
            run = OptimizationRun(
                strategy_name=f"algo_{i}",
                strategy_path=f"strategies/algo_{i}.py",
                mode="grid",
                n_trials=4,
                best_sharpe=float(i),
                best_params='{"sma_fast": 10}',
            )
            session.add(run)
            session.flush()
            for j in range(2):
                session.add(OptimizationTrial(
                    run_id=run.id,
                    trial_number=j,
                    params='{"sma_fast": 10}',
                    sharpe_ratio=float(j),
                ))
    return engine


def test_load_all_optimization_runs_returns_list(engine_with_opt_runs):
    runs = load_all_optimization_runs(engine_with_opt_runs)
    assert isinstance(runs, list)
    assert len(runs) == 3


def test_load_all_optimization_runs_empty(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/empty.db")
    runs = load_all_optimization_runs(engine)
    assert runs == []


def test_load_all_optimization_runs_fields(engine_with_opt_runs):
    runs = load_all_optimization_runs(engine_with_opt_runs)
    for run in runs:
        assert run.strategy_name is not None
        assert run.mode in ("grid", "walk_forward", "ai")


def test_load_optimization_trials_returns_trials(engine_with_opt_runs):
    runs = load_all_optimization_runs(engine_with_opt_runs)
    run_id = runs[0].id
    trials = load_optimization_trials(run_id, engine_with_opt_runs)
    assert isinstance(trials, list)
    assert len(trials) == 2


def test_load_optimization_trials_empty_for_unknown_run(engine_with_opt_runs):
    trials = load_optimization_trials(99999, engine_with_opt_runs)
    assert trials == []
```

- [ ] **Step 6: Run data loader tests — confirm failure**

```bash
pytest tests/test_data_loader.py -v -k "optimization"
```
Expected: `ImportError` — `load_all_optimization_runs` does not exist yet.

- [ ] **Step 7: Add functions to `ui/data_loader.py`**

Append to the end of `C:\Users\noaro\QUANTPLAT\ui\data_loader.py`:

```python
from db.models import OptimizationRun, OptimizationTrial


def load_all_optimization_runs(engine=None) -> List[OptimizationRun]:
    eng = engine or _get_engine()
    with get_session(eng) as session:
        return (
            session.query(OptimizationRun)
            .order_by(OptimizationRun.created_at.desc())
            .all()
        )


def load_optimization_trials(run_id: int, engine=None) -> List[OptimizationTrial]:
    eng = engine or _get_engine()
    with get_session(eng) as session:
        return (
            session.query(OptimizationTrial)
            .filter(OptimizationTrial.run_id == run_id)
            .order_by(OptimizationTrial.trial_number)
            .all()
        )
```

Note: the `List` import is already at the top of `data_loader.py`. The `OptimizationRun` and `OptimizationTrial` imports must be added at the top of the file alongside the existing `BacktestRun` import.

Specifically, change the existing import line:
```python
from db.models import BacktestRun
```
to:
```python
from db.models import BacktestRun, OptimizationRun, OptimizationTrial
```

And remove the duplicate `from db.models import OptimizationRun, OptimizationTrial` line from the appended code.

- [ ] **Step 8: Run all data loader tests — all must pass**

```bash
pytest tests/test_data_loader.py -v
```
Expected: `12 passed` (7 original + 5 new)

- [ ] **Step 9: Run full suite to verify no regressions**

```bash
pytest -q --tb=short
```
Expected: all 68+ tests pass.

- [ ] **Step 10: Commit**

```bash
git add db/models.py ui/data_loader.py tests/test_optimization_models.py tests/test_data_loader.py
git commit -m "feat: OptimizationRun + OptimizationTrial models and data loader extensions"
```

---

## Task 2: LLM Provider

**Files:**
- Create: `ai_agent/__init__.py`
- Create: `ai_agent/provider.py`
- Create: `tests/test_llm_provider.py`

- [ ] **Step 1: Write failing tests**

Create `C:\Users\noaro\QUANTPLAT\tests\test_llm_provider.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from ai_agent.provider import LLMProvider


def test_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    provider = LLMProvider()
    assert provider._provider == "ollama"


def test_provider_reads_env_var(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "claude")
    provider = LLMProvider()
    assert provider._provider == "claude"


def test_call_ollama_returns_string(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    mock_response = {"message": {"content": "test response"}}
    with patch("ollama.chat", return_value=mock_response):
        provider = LLMProvider()
        result = provider.call("hello")
    assert result == "test response"


def test_call_claude_returns_string(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_content = MagicMock()
    mock_content.text = "claude response"
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        provider = LLMProvider()
        result = provider.call("hello")
    assert result == "claude response"


def test_call_openai_returns_string(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_choice = MagicMock()
    mock_choice.message.content = "openai response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = mock_response
        provider = LLMProvider()
        result = provider.call("hello")
    assert result == "openai response"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "unknown_provider")
    provider = LLMProvider()
    with pytest.raises(ValueError, match="Unknown AI provider"):
        provider.call("hello")


def test_ollama_model_defaults_to_llama3(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    provider = LLMProvider()
    assert provider._ollama_model == "llama3"


def test_ollama_model_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    provider = LLMProvider()
    assert provider._ollama_model == "mistral"
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_llm_provider.py -v
```
Expected: `ModuleNotFoundError: No module named 'ai_agent'`

- [ ] **Step 3: Create `ai_agent/__init__.py`** (empty)

```bash
New-Item -ItemType File "C:\Users\noaro\QUANTPLAT\ai_agent\__init__.py" -Force
```

- [ ] **Step 4: Create `ai_agent/provider.py`**

```python
import os


class LLMProvider:
    def __init__(self):
        self._provider = os.getenv("AI_PROVIDER", "ollama")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    def call(self, prompt: str) -> str:
        if self._provider == "ollama":
            return self._call_ollama(prompt)
        elif self._provider == "claude":
            return self._call_claude(prompt)
        elif self._provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown AI provider: {self._provider}")

    def _call_ollama(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self._ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _call_claude(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

- [ ] **Step 5: Run tests — all must pass**

```bash
pytest tests/test_llm_provider.py -v
```
Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add ai_agent/__init__.py ai_agent/provider.py tests/test_llm_provider.py
git commit -m "feat: LLM provider (Ollama/Claude/OpenAI) with env-based config"
```

---

## Task 3: Grid Search Optimizer

**Files:**
- Create: `optimizer/__init__.py`
- Create: `optimizer/grid_search.py`
- Create: `tests/test_grid_search.py`

- [ ] **Step 1: Write failing tests**

Create `C:\Users\noaro\QUANTPLAT\tests\test_grid_search.py`:

```python
import json
import pytest
from unittest.mock import MagicMock, patch
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial
from optimizer.grid_search import GridSearchOptimizer


def _make_fake_lean_output():
    return {
        "pl_list": [100.0, -50.0, 200.0],
        "equity_curve": [50000, 50100, 50050, 50250],
        "raw_statistics": {},
        "runtime_statistics": {},
        "results_path": "/tmp/results.json",
    }


def _make_fake_report(sharpe=1.0):
    return {
        "strategy_name": "test_algo",
        "metrics": {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sharpe * 1.2,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "profit_factor": 1.5,
            "expectancy": 100.0,
            "annual_return": 0.2,
            "total_trades": 50,
        },
        "monte_carlo": {"percentiles": {"p5": [], "p50": [], "p95": []}},
        "equity_curve": [50000, 51000],
        "pl_list": [100.0, -50.0],
        "report_path": "/tmp/report.json",
    }


@pytest.fixture
def mock_deps(tmp_path):
    mock_runner = MagicMock()
    mock_runner.run.return_value = _make_fake_lean_output()

    mock_report_gen = MagicMock()
    mock_report_gen.generate.return_value = _make_fake_report(1.0)

    mock_adapter = MagicMock()
    mock_adapter.prepare.return_value = tmp_path

    mock_loader = MagicMock()
    mock_loader.load.return_value = mock_adapter

    engine = init_db(f"sqlite:///{tmp_path}/test.db")

    return {
        "runner": mock_runner,
        "report_gen": mock_report_gen,
        "loader": mock_loader,
        "adapter": mock_adapter,
        "engine": engine,
    }


def test_grid_search_runs_all_combos(mock_deps, tmp_path):
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    param_grid = {"sma_fast": [5, 10], "sma_slow": [50, 100]}
    result = opt.run("strategies/test.py", param_grid)
    assert mock_deps["runner"].run.call_count == 4


def test_grid_search_returns_optimization_run(mock_deps, tmp_path):
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    result = opt.run("strategies/test.py", {"sma_fast": [5, 10]})
    assert isinstance(result, OptimizationRun)
    assert result.mode == "grid"
    assert result.n_trials == 2


def test_grid_search_stores_trials_in_db(mock_deps):
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    result = opt.run("strategies/test.py", {"sma_fast": [5, 10, 20]})

    with get_session(mock_deps["engine"]) as session:
        trials = (
            session.query(OptimizationTrial)
            .filter(OptimizationTrial.run_id == result.id)
            .all()
        )
    assert len(trials) == 3


def test_grid_search_best_params_set(mock_deps):
    sharpes = [0.5, 1.8, 1.2]
    call_count = [0]

    def fake_generate(name, path, lean_output, n_mc=1000):
        report = _make_fake_report(sharpes[call_count[0]])
        call_count[0] += 1
        return report

    mock_deps["report_gen"].generate.side_effect = fake_generate

    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    result = opt.run("strategies/test.py", {"sma_fast": [5, 10, 20]})
    assert result.best_sharpe == pytest.approx(1.8)
    best = json.loads(result.best_params)
    assert best["sma_fast"] == 10


def test_grid_search_on_trial_callback(mock_deps):
    trial_results = []
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    opt.run("strategies/test.py", {"sma_fast": [5, 10]}, on_trial=trial_results.append)
    assert len(trial_results) == 2
    assert all(isinstance(r, tuple) for r in trial_results)


def test_grid_search_cleanup_called_per_trial(mock_deps):
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    opt.run("strategies/test.py", {"sma_fast": [5, 10, 20]})
    assert mock_deps["adapter"].cleanup.call_count == 3


def test_grid_search_handles_lean_error(mock_deps):
    from core.lean_runner import LeanRunError
    mock_deps["runner"].run.side_effect = LeanRunError("LEAN failed")
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    result = opt.run("strategies/test.py", {"sma_fast": [5, 10]})
    assert isinstance(result, OptimizationRun)
    assert result.best_sharpe is None
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_grid_search.py -v
```
Expected: `ModuleNotFoundError: No module named 'optimizer'`

- [ ] **Step 3: Create `optimizer/__init__.py`** (empty)

```bash
New-Item -ItemType File "C:\Users\noaro\QUANTPLAT\optimizer\__init__.py" -Force
```

- [ ] **Step 4: Create `optimizer/grid_search.py`**

```python
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial


class GridSearchOptimizer:
    def __init__(self, runner=None, report_gen=None, loader=None, engine=None,
                 database_url=None):
        self._runner = runner or LeanRunner()
        self._report_gen = report_gen or ReportGenerator()
        self._loader = loader or StrategyLoader()
        self._engine = engine or init_db(database_url or "sqlite:///quantplat.db")

    def run(
        self,
        strategy_path: str,
        param_grid: Dict[str, List[Any]],
        starting_capital: float = 50_000,
        n_mc: int = 1000,
        on_trial: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationRun:
        strategy_path = Path(strategy_path)
        param_names = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in param_names]))

        with get_session(self._engine) as session:
            opt_run = OptimizationRun(
                strategy_name=strategy_path.stem,
                strategy_path=str(strategy_path),
                mode="grid",
                n_trials=len(combos),
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

        best_sharpe: Optional[float] = None
        best_params: Optional[dict] = None

        for trial_num, combo in enumerate(combos):
            params = dict(zip(param_names, combo))
            metrics = self._run_single(
                strategy_path, params, starting_capital, n_mc, run_id, trial_num
            )
            if on_trial:
                on_trial(trial_num, metrics)
            sharpe = metrics.get("sharpe_ratio") or 0.0
            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        with get_session(self._engine) as session:
            opt_run = session.get(OptimizationRun, run_id)
            opt_run.best_sharpe = best_sharpe
            opt_run.best_params = json.dumps(best_params) if best_params else None

        with get_session(self._engine) as session:
            return session.get(OptimizationRun, run_id)

    def _run_single(
        self,
        strategy_path: Path,
        params: dict,
        starting_capital: float,
        n_mc: int,
        run_id: int,
        trial_num: int,
    ) -> dict:
        adapter = self._loader.load(strategy_path)
        project_dir = adapter.prepare(strategy_path, parameters=params)
        try:
            lean_output = self._runner.run(project_dir)
            report = self._report_gen.generate(
                strategy_path.stem, str(strategy_path), lean_output, n_mc=n_mc
            )
            metrics = report.get("metrics", {})
        except LeanRunError:
            metrics = {}
        finally:
            adapter.cleanup()

        with get_session(self._engine) as session:
            session.add(OptimizationTrial(
                run_id=run_id,
                trial_number=trial_num,
                params=json.dumps(params),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
            ))

        return metrics
```

- [ ] **Step 5: Fix failing test for `on_trial` callback signature**

The `test_grid_search_on_trial_callback` test checks `isinstance(r, tuple)` but `on_trial` is called as `on_trial(trial_num, metrics)` — two separate arguments, not a tuple. Fix the test assertion:

```python
def test_grid_search_on_trial_callback(mock_deps):
    trial_results = []

    def capture(trial_num, metrics):
        trial_results.append((trial_num, metrics))

    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    opt.run("strategies/test.py", {"sma_fast": [5, 10]}, on_trial=capture)
    assert len(trial_results) == 2
    assert trial_results[0][0] == 0
    assert trial_results[1][0] == 1
```

- [ ] **Step 6: Run tests — all must pass**

```bash
pytest tests/test_grid_search.py -v
```
Expected: `7 passed`

- [ ] **Step 7: Commit**

```bash
git add optimizer/__init__.py optimizer/grid_search.py tests/test_grid_search.py
git commit -m "feat: grid search optimizer with Cartesian parameter sweep"
```

---

## Task 4: Walk-Forward Optimizer

**Files:**
- Create: `optimizer/walk_forward.py`
- Create: `tests/test_walk_forward.py`

The walk-forward optimizer injects `__wf_start` and `__wf_end` (ISO-format date strings) into the LEAN parameters dict. Strategies that support walk-forward call `self.GetParameter("__wf_start")` and `self.GetParameter("__wf_end")` in their `Initialize()` to set `SetStartDate` / `SetEndDate`. Strategies that don't use these parameters run with their hardcoded dates.

- [ ] **Step 1: Write failing tests**

Create `C:\Users\noaro\QUANTPLAT\tests\test_walk_forward.py`:

```python
import json
import pytest
from datetime import date
from unittest.mock import MagicMock
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial
from optimizer.walk_forward import WalkForwardOptimizer, _split_windows


# ─── _split_windows unit tests ───────────────────────────────────────────────

def test_split_windows_returns_correct_count():
    windows = _split_windows(date(2020, 1, 1), date(2021, 1, 1), n_windows=4, train_ratio=0.7)
    assert len(windows) == 4


def test_split_windows_train_ends_before_test_starts():
    windows = _split_windows(date(2020, 1, 1), date(2021, 1, 1), n_windows=3, train_ratio=0.7)
    for (ts, te, vs, ve) in windows:
        assert te <= vs


def test_split_windows_test_ends_at_window_boundary():
    windows = _split_windows(date(2020, 1, 1), date(2021, 1, 1), n_windows=2, train_ratio=0.8)
    for (ts, te, vs, ve) in windows:
        assert ts < te < vs < ve


def test_split_windows_no_overlap():
    windows = _split_windows(date(2020, 1, 1), date(2022, 1, 1), n_windows=5, train_ratio=0.7)
    for i in range(len(windows) - 1):
        _, _, _, w1_end = windows[i]
        w2_start, _, _, _ = windows[i + 1]
        assert w1_end <= w2_start


# ─── WalkForwardOptimizer integration tests ──────────────────────────────────

def _make_fake_lean_output():
    return {
        "pl_list": [100.0, -50.0, 200.0],
        "equity_curve": [50000, 51000],
        "raw_statistics": {},
        "runtime_statistics": {},
        "results_path": "/tmp/results.json",
    }


def _make_fake_report(sharpe=1.0):
    return {
        "strategy_name": "test_algo",
        "metrics": {
            "sharpe_ratio": sharpe,
            "sortino_ratio": 1.3,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "profit_factor": 1.5,
            "expectancy": 80.0,
            "annual_return": 0.18,
            "total_trades": 40,
        },
        "monte_carlo": {"percentiles": {"p5": [], "p50": [], "p95": []}},
        "equity_curve": [50000, 51000],
        "pl_list": [100.0, -50.0],
        "report_path": "/tmp/report.json",
    }


@pytest.fixture
def mock_deps(tmp_path):
    mock_runner = MagicMock()
    mock_runner.run.return_value = _make_fake_lean_output()
    mock_report_gen = MagicMock()
    mock_report_gen.generate.return_value = _make_fake_report(1.0)
    mock_adapter = MagicMock()
    mock_adapter.prepare.return_value = tmp_path
    mock_loader = MagicMock()
    mock_loader.load.return_value = mock_adapter
    engine = init_db(f"sqlite:///{tmp_path}/test.db")
    return {
        "runner": mock_runner,
        "report_gen": mock_report_gen,
        "loader": mock_loader,
        "adapter": mock_adapter,
        "engine": engine,
    }


def test_walk_forward_returns_optimization_run(mock_deps):
    opt = WalkForwardOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    result = opt.run(
        "strategies/test.py",
        param_grid={"sma_fast": [5, 10]},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
        n_windows=2,
    )
    assert isinstance(result, OptimizationRun)
    assert result.mode == "walk_forward"


def test_walk_forward_stores_trials_with_window_labels(mock_deps):
    opt = WalkForwardOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    result = opt.run(
        "strategies/test.py",
        param_grid={"sma_fast": [5, 10]},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
        n_windows=2,
    )
    with get_session(mock_deps["engine"]) as session:
        trials = (
            session.query(OptimizationTrial)
            .filter(OptimizationTrial.run_id == result.id)
            .all()
        )
    labels = {t.window_label for t in trials}
    assert any("train" in lbl for lbl in labels)
    assert any("test" in lbl for lbl in labels)


def test_walk_forward_injects_dates_into_params(mock_deps):
    opt = WalkForwardOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    opt.run(
        "strategies/test.py",
        param_grid={"sma_fast": [5]},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
        n_windows=1,
    )
    calls = mock_deps["adapter"].prepare.call_args_list
    for call in calls:
        params = call[1].get("parameters") or call[0][1]
        assert "__wf_start" in params
        assert "__wf_end" in params
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_walk_forward.py -v
```
Expected: `ModuleNotFoundError: No module named 'optimizer.walk_forward'`

- [ ] **Step 3: Create `optimizer/walk_forward.py`**

```python
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial


def _split_windows(
    start_date: date,
    end_date: date,
    n_windows: int,
    train_ratio: float,
) -> List[Tuple[date, date, date, date]]:
    """Returns list of (train_start, train_end, test_start, test_end)."""
    total_days = (end_date - start_date).days
    window_days = total_days // n_windows
    train_days = int(window_days * train_ratio)

    windows = []
    for i in range(n_windows):
        w_start = start_date + timedelta(days=i * window_days)
        w_end = start_date + timedelta(days=(i + 1) * window_days)
        train_end = w_start + timedelta(days=train_days)
        windows.append((w_start, train_end, train_end, w_end))
    return windows


class WalkForwardOptimizer:
    def __init__(self, runner=None, report_gen=None, loader=None, engine=None,
                 database_url=None):
        self._runner = runner or LeanRunner()
        self._report_gen = report_gen or ReportGenerator()
        self._loader = loader or StrategyLoader()
        self._engine = engine or init_db(database_url or "sqlite:///quantplat.db")

    def run(
        self,
        strategy_path: str,
        param_grid: Dict[str, List[Any]],
        start_date: date,
        end_date: date,
        n_windows: int = 5,
        train_ratio: float = 0.7,
        starting_capital: float = 50_000,
        n_mc: int = 1000,
        on_trial: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationRun:
        import itertools

        strategy_path = Path(strategy_path)
        windows = _split_windows(start_date, end_date, n_windows, train_ratio)
        param_names = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in param_names]))

        total_trials = n_windows * (len(combos) + 1)

        with get_session(self._engine) as session:
            opt_run = OptimizationRun(
                strategy_name=strategy_path.stem,
                strategy_path=str(strategy_path),
                mode="walk_forward",
                n_trials=total_trials,
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

        best_sharpe: Optional[float] = None
        best_params: Optional[dict] = None
        trial_num = 0

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Optimize on train period
            window_best_sharpe: Optional[float] = None
            window_best_params: Optional[dict] = None

            for combo in combos:
                params = dict(zip(param_names, combo))
                params["__wf_start"] = train_start.isoformat()
                params["__wf_end"] = train_end.isoformat()

                metrics = self._run_single(
                    strategy_path, params, starting_capital, n_mc, run_id, trial_num,
                    window_label=f"window_{w_idx}_train",
                )
                if on_trial:
                    on_trial(trial_num, metrics)
                trial_num += 1

                sharpe = metrics.get("sharpe_ratio") or 0.0
                if window_best_sharpe is None or sharpe > window_best_sharpe:
                    window_best_sharpe = sharpe
                    window_best_params = {
                        k: v for k, v in params.items()
                        if not k.startswith("__wf_")
                    }

            # Validate best params on test period
            if window_best_params:
                val_params = dict(window_best_params)
                val_params["__wf_start"] = test_start.isoformat()
                val_params["__wf_end"] = test_end.isoformat()

                metrics = self._run_single(
                    strategy_path, val_params, starting_capital, n_mc, run_id, trial_num,
                    window_label=f"window_{w_idx}_test",
                )
                if on_trial:
                    on_trial(trial_num, metrics)
                trial_num += 1

                sharpe = metrics.get("sharpe_ratio") or 0.0
                if best_sharpe is None or sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = window_best_params

        with get_session(self._engine) as session:
            opt_run = session.get(OptimizationRun, run_id)
            opt_run.best_sharpe = best_sharpe
            opt_run.best_params = json.dumps(best_params) if best_params else None

        with get_session(self._engine) as session:
            return session.get(OptimizationRun, run_id)

    def _run_single(
        self,
        strategy_path: Path,
        params: dict,
        starting_capital: float,
        n_mc: int,
        run_id: int,
        trial_num: int,
        window_label: Optional[str] = None,
    ) -> dict:
        adapter = self._loader.load(strategy_path)
        project_dir = adapter.prepare(strategy_path, parameters=params)
        try:
            lean_output = self._runner.run(project_dir)
            report = self._report_gen.generate(
                strategy_path.stem, str(strategy_path), lean_output, n_mc=n_mc
            )
            metrics = report.get("metrics", {})
        except LeanRunError:
            metrics = {}
        finally:
            adapter.cleanup()

        with get_session(self._engine) as session:
            session.add(OptimizationTrial(
                run_id=run_id,
                trial_number=trial_num,
                params=json.dumps(params),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
                window_label=window_label,
            ))

        return metrics
```

- [ ] **Step 4: Run tests — all must pass**

```bash
pytest tests/test_walk_forward.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add optimizer/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: walk-forward optimizer with date-windowed train/test splits"
```

---

## Task 5: AI Optimizer (Optuna + LLM)

**Files:**
- Create: `optimizer/ai_optimizer.py`
- Create: `tests/test_ai_optimizer.py`

- [ ] **Step 1: Write failing tests**

Create `C:\Users\noaro\QUANTPLAT\tests\test_ai_optimizer.py`:

```python
import json
import pytest
from unittest.mock import MagicMock, patch
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial
from optimizer.ai_optimizer import AIOptimizer, _build_prompt, _parse_llm_response


# ─── pure function tests ──────────────────────────────────────────────────────

def test_build_prompt_contains_trial_history():
    top_trials = [{"params": {"sma_fast": 10}, "sharpe_ratio": 1.2}]
    param_ranges = {"sma_fast": (5, 50, "int")}
    prompt = _build_prompt(top_trials, param_ranges)
    assert "1.200" in prompt
    assert "sma_fast" in prompt
    assert "5" in prompt and "50" in prompt


def test_parse_llm_response_valid_json():
    param_ranges = {"sma_fast": (5, 50, "int"), "risk": (0.01, 0.05, "float")}
    response = 'Here is my suggestion: {"sma_fast": 15, "risk": 0.03}'
    result = _parse_llm_response(response, param_ranges)
    assert result is not None
    assert result["sma_fast"] == 15
    assert result["risk"] == pytest.approx(0.03)


def test_parse_llm_response_clips_to_range():
    param_ranges = {"sma_fast": (5, 50, "int")}
    response = '{"sma_fast": 200}'
    result = _parse_llm_response(response, param_ranges)
    assert result["sma_fast"] == 50


def test_parse_llm_response_invalid_json_returns_none():
    param_ranges = {"sma_fast": (5, 50, "int")}
    result = _parse_llm_response("No JSON here at all", param_ranges)
    assert result is None


def test_parse_llm_response_int_type_cast():
    param_ranges = {"sma_fast": (5, 50, "int")}
    response = '{"sma_fast": 12.7}'
    result = _parse_llm_response(response, param_ranges)
    assert isinstance(result["sma_fast"], int)
    assert result["sma_fast"] == 12


# ─── AIOptimizer integration tests ───────────────────────────────────────────

def _make_fake_lean_output():
    return {
        "pl_list": [100.0, -50.0],
        "equity_curve": [50000, 51000],
        "raw_statistics": {},
        "runtime_statistics": {},
        "results_path": "/tmp/results.json",
    }


def _make_fake_report(sharpe=1.0):
    return {
        "strategy_name": "test_algo",
        "metrics": {
            "sharpe_ratio": sharpe,
            "sortino_ratio": 1.3,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "profit_factor": 1.5,
            "expectancy": 80.0,
            "annual_return": 0.18,
            "total_trades": 40,
        },
        "monte_carlo": {"percentiles": {"p5": [], "p50": [], "p95": []}},
        "equity_curve": [50000, 51000],
        "pl_list": [100.0, -50.0],
        "report_path": "/tmp/report.json",
    }


@pytest.fixture
def mock_deps(tmp_path):
    mock_runner = MagicMock()
    mock_runner.run.return_value = _make_fake_lean_output()
    mock_report_gen = MagicMock()
    mock_report_gen.generate.return_value = _make_fake_report(1.0)
    mock_adapter = MagicMock()
    mock_adapter.prepare.return_value = tmp_path
    mock_loader = MagicMock()
    mock_loader.load.return_value = mock_adapter
    mock_llm = MagicMock()
    mock_llm.call.return_value = '{"sma_fast": 15}'
    engine = init_db(f"sqlite:///{tmp_path}/test.db")
    return {
        "runner": mock_runner,
        "report_gen": mock_report_gen,
        "loader": mock_loader,
        "adapter": mock_adapter,
        "llm": mock_llm,
        "engine": engine,
    }


def test_ai_optimizer_returns_optimization_run(mock_deps):
    opt = AIOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        llm=mock_deps["llm"],
        engine=mock_deps["engine"],
    )
    result = opt.run(
        "strategies/test.py",
        param_ranges={"sma_fast": (5, 50, "int")},
        n_trials=5,
    )
    assert isinstance(result, OptimizationRun)
    assert result.mode == "ai"


def test_ai_optimizer_runs_n_trials(mock_deps):
    opt = AIOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        llm=mock_deps["llm"],
        engine=mock_deps["engine"],
    )
    opt.run(
        "strategies/test.py",
        param_ranges={"sma_fast": (5, 50, "int")},
        n_trials=6,
    )
    assert mock_deps["runner"].run.call_count == 6


def test_ai_optimizer_stores_trials(mock_deps):
    opt = AIOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        llm=mock_deps["llm"],
        engine=mock_deps["engine"],
    )
    result = opt.run(
        "strategies/test.py",
        param_ranges={"sma_fast": (5, 50, "int")},
        n_trials=4,
    )
    with get_session(mock_deps["engine"]) as session:
        trials = (
            session.query(OptimizationTrial)
            .filter(OptimizationTrial.run_id == result.id)
            .all()
        )
    assert len(trials) >= 4


def test_ai_optimizer_calls_llm_at_interval(mock_deps):
    opt = AIOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        llm=mock_deps["llm"],
        engine=mock_deps["engine"],
    )
    opt.run(
        "strategies/test.py",
        param_ranges={"sma_fast": (5, 50, "int")},
        n_trials=10,
        llm_interval=5,
    )
    assert mock_deps["llm"].call.call_count == 2


def test_ai_optimizer_llm_failure_does_not_crash(mock_deps):
    mock_deps["llm"].call.side_effect = Exception("LLM unavailable")
    opt = AIOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        llm=mock_deps["llm"],
        engine=mock_deps["engine"],
    )
    result = opt.run(
        "strategies/test.py",
        param_ranges={"sma_fast": (5, 50, "int")},
        n_trials=5,
        llm_interval=3,
    )
    assert isinstance(result, OptimizationRun)
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_ai_optimizer.py -v
```
Expected: `ModuleNotFoundError: No module named 'optimizer.ai_optimizer'`

- [ ] **Step 3: Create `optimizer/ai_optimizer.py`**

```python
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List

import optuna

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from ai_agent.provider import LLMProvider
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_prompt(
    top_trials: List[dict],
    param_ranges: Dict[str, Tuple],
) -> str:
    lines = [
        "You are a quantitative trading strategy optimizer.",
        "Top performing parameter combinations so far:",
    ]
    for t in top_trials:
        lines.append(f"  Sharpe {t['sharpe_ratio']:.3f}: {t['params']}")
    lines.append("\nParameter ranges:")
    for name, (low, high, ptype) in param_ranges.items():
        lines.append(f"  {name}: {low} to {high} ({ptype})")
    lines.append(
        "\nSuggest ONE parameter combination that might improve the Sharpe ratio."
        "\nRespond with ONLY a JSON dict, e.g.: {\"sma_fast\": 15, \"sma_slow\": 80}"
    )
    return "\n".join(lines)


def _parse_llm_response(
    response: str,
    param_ranges: Dict[str, Tuple],
) -> Optional[dict]:
    match = re.search(r'\{[^}]+\}', response)
    if not match:
        return None
    try:
        suggested = json.loads(match.group())
        result = {}
        for name, (low, high, ptype) in param_ranges.items():
            if name in suggested:
                val = float(suggested[name])
                val = max(float(low), min(float(high), val))
                result[name] = int(val) if ptype == "int" else val
        return result if result else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class AIOptimizer:
    def __init__(self, runner=None, report_gen=None, loader=None, llm=None,
                 engine=None, database_url=None):
        self._runner = runner or LeanRunner()
        self._report_gen = report_gen or ReportGenerator()
        self._loader = loader or StrategyLoader()
        self._llm = llm or LLMProvider()
        self._engine = engine or init_db(database_url or "sqlite:///quantplat.db")

    def run(
        self,
        strategy_path: str,
        param_ranges: Dict[str, Tuple],
        n_trials: int = 50,
        starting_capital: float = 50_000,
        n_mc: int = 1000,
        llm_interval: int = 10,
        on_trial: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationRun:
        strategy_path = Path(strategy_path)

        with get_session(self._engine) as session:
            opt_run = OptimizationRun(
                strategy_name=strategy_path.stem,
                strategy_path=str(strategy_path),
                mode="ai",
                n_trials=n_trials,
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

        trial_history: List[dict] = []
        best_sharpe: Optional[float] = None
        best_params: Optional[dict] = None

        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            params: dict = {}
            for name, (low, high, ptype) in param_ranges.items():
                if ptype == "int":
                    params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))

            metrics = self._run_single(
                strategy_path, params, starting_capital, n_mc, run_id, trial.number
            )
            sharpe = metrics.get("sharpe_ratio") or 0.0
            trial_history.append({"params": params, "sharpe_ratio": sharpe})

            if on_trial:
                on_trial(trial.number, {"params": params, **metrics})

            nonlocal best_sharpe, best_params
            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

            if (trial.number + 1) % llm_interval == 0:
                top = sorted(trial_history, key=lambda t: t["sharpe_ratio"], reverse=True)[:5]
                try:
                    response = self._llm.call(_build_prompt(top, param_ranges))
                    suggested = _parse_llm_response(response, param_ranges)
                    if suggested:
                        study.enqueue_trial(suggested)
                except Exception:
                    pass

            return sharpe

        study.optimize(objective, n_trials=n_trials)

        with get_session(self._engine) as session:
            opt_run = session.get(OptimizationRun, run_id)
            opt_run.best_sharpe = best_sharpe
            opt_run.best_params = json.dumps(best_params) if best_params else None

        with get_session(self._engine) as session:
            return session.get(OptimizationRun, run_id)

    def _run_single(
        self,
        strategy_path: Path,
        params: dict,
        starting_capital: float,
        n_mc: int,
        run_id: int,
        trial_num: int,
    ) -> dict:
        adapter = self._loader.load(strategy_path)
        project_dir = adapter.prepare(strategy_path, parameters=params)
        try:
            lean_output = self._runner.run(project_dir)
            report = self._report_gen.generate(
                strategy_path.stem, str(strategy_path), lean_output, n_mc=n_mc
            )
            metrics = report.get("metrics", {})
        except LeanRunError:
            metrics = {}
        finally:
            adapter.cleanup()

        with get_session(self._engine) as session:
            session.add(OptimizationTrial(
                run_id=run_id,
                trial_number=trial_num,
                params=json.dumps(params),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
            ))

        return metrics
```

- [ ] **Step 4: Run tests — all must pass**

```bash
pytest tests/test_ai_optimizer.py -v
```
Expected: `10 passed`

- [ ] **Step 5: Run full suite**

```bash
pytest -q --tb=short
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add optimizer/ai_optimizer.py tests/test_ai_optimizer.py
git commit -m "feat: AI optimizer — Optuna TPE with LLM-guided parameter suggestions"
```

---

## Task 6: Scatter Chart + Optimizer UI Page

**Files:**
- Modify: `ui/chart_builder.py`
- Modify: `tests/test_chart_builder.py`
- Create: `ui/pages/4_Optimizer.py`

- [ ] **Step 1: Write failing tests for scatter chart**

Append to `C:\Users\noaro\QUANTPLAT\tests\test_chart_builder.py`:

```python
from ui.chart_builder import optimizer_scatter_chart


def test_optimizer_scatter_chart_returns_figure():
    fig = optimizer_scatter_chart([5, 10, 20], [0.5, 1.2, 0.9])
    assert isinstance(fig, go.Figure)


def test_optimizer_scatter_chart_has_one_trace():
    fig = optimizer_scatter_chart([5, 10, 20], [0.5, 1.2, 0.9])
    assert len(fig.data) == 1


def test_optimizer_scatter_chart_custom_labels():
    fig = optimizer_scatter_chart(
        [5, 10], [1.0, 1.5],
        x_label="sma_fast",
        title="SMA Fast vs Sharpe",
    )
    assert fig.layout.title.text == "SMA Fast vs Sharpe"
    assert fig.layout.xaxis.title.text == "sma_fast"


def test_optimizer_scatter_chart_handles_empty():
    fig = optimizer_scatter_chart([], [])
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 2: Run new chart tests — confirm failure**

```bash
pytest tests/test_chart_builder.py -v -k "scatter"
```
Expected: `ImportError` — `optimizer_scatter_chart` does not exist yet.

- [ ] **Step 3: Add `optimizer_scatter_chart` to `ui/chart_builder.py`**

Append to the end of `C:\Users\noaro\QUANTPLAT\ui\chart_builder.py`:

```python
def optimizer_scatter_chart(
    param_values: List[float],
    sharpe_values: List[float],
    x_label: str = "Parameter",
    title: str = "Parameter Optimization",
) -> go.Figure:
    colors = [_GREEN if s >= 0 else _RED for s in sharpe_values]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=param_values,
        y=sharpe_values,
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.7),
        name="Trials",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Sharpe Ratio",
        template=_THEME,
        height=400,
    )
    return fig
```

Also add `List` to the imports at the top of `chart_builder.py` if not already present. The current top import is:
```python
from typing import Any, Dict, List
```
`List` is already imported — no change needed.

- [ ] **Step 4: Run all chart builder tests — all must pass**

```bash
pytest tests/test_chart_builder.py -v
```
Expected: `15 passed` (11 original + 4 new)

- [ ] **Step 5: Create `ui/pages/4_Optimizer.py`**

```python
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
    status_placeholder = st.empty()

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
    col3.metric("Best Sharpe", f"{result.best_sharpe:.3f}" if result.best_sharpe else "N/A")

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
```

- [ ] **Step 6: Run full test suite**

```bash
cd C:\Users\noaro\QUANTPLAT
pytest -q --tb=short
```
Expected: all tests pass (95+).

- [ ] **Step 7: Smoke test the optimizer UI**

```bash
streamlit run ui/app.py
```

Navigate to "4 Optimizer". Verify:
- Mode selector shows three options
- Strategy path input renders
- Parameter JSON text area updates based on mode
- Walk-Forward shows date inputs and window slider
- AI mode shows trial count + LLM interval sliders
- Run button is disabled when no path is entered
- Entering a non-existent path and clicking Run shows "File not found" error
- No Python exceptions in the terminal

- [ ] **Step 8: Commit**

```bash
git add ui/chart_builder.py tests/test_chart_builder.py ui/pages/4_Optimizer.py
git commit -m "feat: optimizer scatter chart and Streamlit optimizer UI page"
```

---

## Task 7: Push to GitHub

- [ ] **Step 1: Final test suite**

```bash
pytest -v --tb=short
```
Expected: all tests green.

- [ ] **Step 2: Push**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:**
  - Grid search ✓ (itertools.product, all combos, DB storage)
  - Walk-forward ✓ (date window splitting, train/test per window, date injection via params)
  - AI optimizer ✓ (Optuna TPE, LLM suggestions every N trials, LLM failure handled)
  - LLM provider ✓ (Ollama/Claude/OpenAI, env config)
  - Optimizer UI ✓ (mode selector, parameter input, live progress, scatter chart, trials table)
  - DB models ✓ (OptimizationRun + OptimizationTrial with window_label)
  - Data loader extensions ✓

- **Placeholders:** None — all steps contain complete code.

- **Type consistency:**
  - `param_grid: Dict[str, List[Any]]` in grid search and walk-forward (lists of values)
  - `param_ranges: Dict[str, Tuple]` in AI optimizer where each tuple is `(low, high, type_str)`
  - `on_trial: Callable[[int, dict], None]` — same signature across all three optimizers
  - `OptimizationRun.best_params` stored as JSON string, decoded with `json.loads()` in UI
  - `OptimizationTrial.params` stored as JSON string, decoded with `json.loads()` in UI

- **Walk-forward contract:** Strategies that want true date isolation should call
  `self.GetParameter("__wf_start")` and `self.GetParameter("__wf_end")` in LEAN's
  `Initialize()`. Strategies without these calls still run normally — they just use their
  hardcoded dates for all windows.
