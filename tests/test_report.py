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
