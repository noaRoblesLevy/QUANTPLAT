import json
import pytest
from unittest.mock import MagicMock
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
    assert mock_deps["llm"].call.call_count >= 2


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
