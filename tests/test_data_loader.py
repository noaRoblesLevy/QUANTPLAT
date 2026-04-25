import json
import pytest
from db import init_db, get_session
from db.models import BacktestRun, OptimizationRun, OptimizationTrial
from ui.data_loader import load_all_runs, load_report, load_all_optimization_runs, load_optimization_trials


@pytest.fixture
def engine_with_runs(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/test.db")
    with get_session(engine) as session:
        for i in range(3):
            run = BacktestRun(
                strategy_name=f"algo_{i}",
                strategy_path=f"strategies/algo_{i}.py",
                sharpe_ratio=float(i),
                sortino_ratio=float(i),
                max_drawdown=-0.1 * i,
                profit_factor=1.0 + i * 0.1,
                win_rate=0.4 + i * 0.05,
                expectancy=100.0 * i,
                annual_return=0.1 * i,
                total_trades=10 * (i + 1),
                results_path=f"results/run_{i}.json",
            )
            session.add(run)
    return engine


def test_load_all_runs_returns_list(engine_with_runs):
    runs = load_all_runs(engine_with_runs)
    assert isinstance(runs, list)
    assert len(runs) == 3


def test_load_all_runs_empty_db(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/empty.db")
    runs = load_all_runs(engine)
    assert runs == []


def test_load_all_runs_returns_backtest_run_objects(engine_with_runs):
    runs = load_all_runs(engine_with_runs)
    assert all(isinstance(r, BacktestRun) for r in runs)


def test_load_all_runs_fields_accessible(engine_with_runs):
    runs = load_all_runs(engine_with_runs)
    for run in runs:
        assert run.strategy_name is not None
        assert run.sharpe_ratio is not None
        assert run.results_path is not None


def test_load_report_returns_dict(tmp_path):
    report = {"metrics": {"sharpe_ratio": 1.2}, "monte_carlo": {"percentiles": {}}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = load_report(str(report_path))
    assert result is not None
    assert result["metrics"]["sharpe_ratio"] == 1.2


def test_load_report_returns_none_for_missing_file():
    result = load_report("/nonexistent/path/report.json")
    assert result is None


def test_load_report_returns_full_structure(tmp_path):
    report = {
        "strategy_name": "my_algo",
        "metrics": {"sharpe_ratio": 0.9, "max_drawdown": -0.12},
        "monte_carlo": {"percentiles": {"p5": [1], "p50": [2], "p95": [3]}},
        "pl_list": [100.0, -50.0],
    }
    path = tmp_path / "full_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = load_report(str(path))
    assert result["strategy_name"] == "my_algo"
    assert "monte_carlo" in result


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
        assert run.mode == "grid"


def test_load_optimization_trials_returns_trials(engine_with_opt_runs):
    runs = load_all_optimization_runs(engine_with_opt_runs)
    run_id = runs[0].id
    trials = load_optimization_trials(run_id, engine_with_opt_runs)
    assert isinstance(trials, list)
    assert len(trials) == 2


def test_load_optimization_trials_empty_for_unknown_run(engine_with_opt_runs):
    trials = load_optimization_trials(99999, engine_with_opt_runs)
    assert trials == []


from ui.data_loader import update_ai_summary


def test_update_ai_summary_sets_field(engine_with_runs):
    runs = load_all_runs(engine_with_runs)
    target = runs[0]
    update_ai_summary(target.results_path, "Great Sharpe, consider reducing drawdown.", engine_with_runs)
    updated = load_all_runs(engine_with_runs)
    match = next(r for r in updated if r.results_path == target.results_path)
    assert match.ai_summary == "Great Sharpe, consider reducing drawdown."


def test_update_ai_summary_no_op_for_unknown_path(engine_with_runs):
    # Should not raise even if the path doesn't match any run
    update_ai_summary("/nonexistent/path.json", "summary", engine_with_runs)


def test_update_ai_summary_overwrites_previous(engine_with_runs):
    runs = load_all_runs(engine_with_runs)
    path = runs[0].results_path
    update_ai_summary(path, "First summary", engine_with_runs)
    update_ai_summary(path, "Updated summary", engine_with_runs)
    updated = load_all_runs(engine_with_runs)
    match = next(r for r in updated if r.results_path == path)
    assert match.ai_summary == "Updated summary"
