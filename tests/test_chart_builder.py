import pytest
import plotly.graph_objects as go
from ui.chart_builder import equity_curve_chart, monte_carlo_chart, trade_distribution_chart


@pytest.fixture
def mc_data():
    return {
        "percentiles": {
            "p5":  [50000, 49500, 49000, 48000],
            "p50": [50000, 51000, 52000, 53000],
            "p95": [50000, 53000, 56000, 59000],
        }
    }


def test_equity_curve_chart_returns_figure(sample_equity_curve):
    fig = equity_curve_chart(sample_equity_curve)
    assert isinstance(fig, go.Figure)


def test_equity_curve_chart_has_one_trace(sample_equity_curve):
    fig = equity_curve_chart(sample_equity_curve)
    assert len(fig.data) == 1


def test_equity_curve_chart_custom_title(sample_equity_curve):
    fig = equity_curve_chart(sample_equity_curve, title="NQ Strategy")
    assert fig.layout.title.text == "NQ Strategy"


def test_equity_curve_chart_y_data_matches_input(sample_equity_curve):
    fig = equity_curve_chart(sample_equity_curve)
    assert list(fig.data[0].y) == sample_equity_curve


def test_monte_carlo_chart_returns_figure(mc_data):
    fig = monte_carlo_chart(mc_data)
    assert isinstance(fig, go.Figure)


def test_monte_carlo_chart_has_at_least_three_traces(mc_data):
    fig = monte_carlo_chart(mc_data)
    assert len(fig.data) >= 3


def test_monte_carlo_chart_custom_title(mc_data):
    fig = monte_carlo_chart(mc_data, title="MC Test")
    assert fig.layout.title.text == "MC Test"


def test_trade_distribution_chart_returns_figure(sample_pl_list):
    fig = trade_distribution_chart(sample_pl_list)
    assert isinstance(fig, go.Figure)


def test_trade_distribution_chart_has_two_traces(sample_pl_list):
    fig = trade_distribution_chart(sample_pl_list)
    assert len(fig.data) == 2


def test_trade_distribution_chart_handles_all_wins():
    fig = trade_distribution_chart([100.0, 200.0, 300.0])
    assert isinstance(fig, go.Figure)


def test_trade_distribution_chart_handles_all_losses():
    fig = trade_distribution_chart([-100.0, -200.0])
    assert isinstance(fig, go.Figure)


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
