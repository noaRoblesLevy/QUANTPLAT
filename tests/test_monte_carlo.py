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
    assert len(result["percentiles"]["p50"]) == len(sample_pl_list) + 1

def test_probability_of_ruin_all_winners():
    pl = [100.0] * 20
    result = run(pl, n_simulations=200)
    assert result["probability_of_ruin"] == 0.0

def test_probability_of_ruin_all_losers():
    pl = [-5000.0] * 20
    result = run(pl, n_simulations=200, starting_equity=50000.0)
    assert result["probability_of_ruin"] == 1.0

def test_run_deterministic_with_seed(sample_pl_list):
    r1 = run(sample_pl_list, n_simulations=100)
    r2 = run(sample_pl_list, n_simulations=100)
    assert r1["final_equity"]["p50"] == r2["final_equity"]["p50"]
