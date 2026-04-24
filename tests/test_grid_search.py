import json
import pytest
from unittest.mock import MagicMock
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


def test_grid_search_runs_all_combos(mock_deps):
    opt = GridSearchOptimizer(
        runner=mock_deps["runner"],
        report_gen=mock_deps["report_gen"],
        loader=mock_deps["loader"],
        engine=mock_deps["engine"],
    )
    param_grid = {"sma_fast": [5, 10], "sma_slow": [50, 100]}
    opt.run("strategies/test.py", param_grid)
    assert mock_deps["runner"].run.call_count == 4


def test_grid_search_returns_optimization_run(mock_deps):
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
