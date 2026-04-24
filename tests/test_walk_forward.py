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
