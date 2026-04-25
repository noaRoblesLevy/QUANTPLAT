import pytest
from unittest.mock import MagicMock
from ai_agent.analyzer import PostBacktestAnalyzer


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.call.return_value = "Sharpe is solid. Max drawdown is high — consider tightening stops. Next step: reduce position size."
    return m


def test_analyze_returns_string(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    result = analyzer.analyze({"sharpe_ratio": 1.2, "max_drawdown": -0.25})
    assert isinstance(result, str)
    assert result == "Sharpe is solid. Max drawdown is high — consider tightening stops. Next step: reduce position size."


def test_analyze_calls_llm_once(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    analyzer.analyze({"sharpe_ratio": 0.8})
    assert mock_llm.call.call_count == 1


def test_analyze_prompt_contains_metrics(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    analyzer.analyze({"sharpe_ratio": 1.5, "win_rate": 0.55})
    prompt = mock_llm.call.call_args[0][0]
    assert "sharpe_ratio" in prompt
    assert "1.5" in prompt


def test_analyze_prompt_contains_pl_stats_when_provided(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    analyzer.analyze({"sharpe_ratio": 1.0}, pl_list=[100.0, -50.0, 200.0, -80.0])
    prompt = mock_llm.call.call_args[0][0]
    assert "Total trades: 4" in prompt


def test_analyze_works_without_pl_list(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    result = analyzer.analyze({"sharpe_ratio": 0.9})
    assert result == "Sharpe is solid. Max drawdown is high — consider tightening stops. Next step: reduce position size."
    prompt = mock_llm.call.call_args[0][0]
    assert "Total trades" not in prompt


def test_analyze_prompt_mentions_performance(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    analyzer.analyze({"sharpe_ratio": 1.0})
    prompt = mock_llm.call.call_args[0][0]
    assert "backtest" in prompt.lower() or "performance" in prompt.lower() or "metric" in prompt.lower()


def test_analyze_avg_win_loss_in_prompt_when_pl_given(mock_llm):
    analyzer = PostBacktestAnalyzer(llm=mock_llm)
    analyzer.analyze({}, pl_list=[100.0, 200.0, -50.0])
    prompt = mock_llm.call.call_args[0][0]
    assert "150" in prompt or "150.0" in prompt or "Avg win" in prompt
