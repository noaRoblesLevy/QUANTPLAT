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
    assert sortino_ratio(returns) == 0.0

def test_sortino_ratio_with_losses():
    returns = [0.01, -0.02, 0.005, -0.015, 0.008]
    result = sortino_ratio(returns)
    assert isinstance(result, float)

def test_max_drawdown_returns_negative():
    equity = [100, 110, 105, 95, 100, 115]
    result = max_drawdown(equity)
    assert result < 0
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
    assert abs(result - 0.4) < 0.001

def test_win_rate_empty():
    assert win_rate([]) == 0.0

def test_expectancy_basic(sample_pl_list):
    result = expectancy(sample_pl_list)
    assert isinstance(result, float)
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
