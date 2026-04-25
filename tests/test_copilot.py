import pytest
from unittest.mock import MagicMock
from ai_agent.copilot import StrategyCopilot


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.call.return_value = "Add a 2% stop loss to each position."
    return m


def test_review_returns_string(mock_llm):
    copilot = StrategyCopilot(llm=mock_llm)
    result = copilot.review("def algo(): pass")
    assert isinstance(result, str)
    assert result == "Add a 2% stop loss to each position."


def test_review_calls_llm_once(mock_llm):
    copilot = StrategyCopilot(llm=mock_llm)
    copilot.review("code here")
    assert mock_llm.call.call_count == 1


def test_review_prompt_contains_code(mock_llm):
    copilot = StrategyCopilot(llm=mock_llm)
    copilot.review("my_unique_strategy_code_xyz")
    prompt = mock_llm.call.call_args[0][0]
    assert "my_unique_strategy_code_xyz" in prompt


def test_review_prompt_contains_metrics_when_provided(mock_llm):
    copilot = StrategyCopilot(llm=mock_llm)
    copilot.review("code", metrics={"sharpe_ratio": 1.75})
    prompt = mock_llm.call.call_args[0][0]
    assert "sharpe_ratio" in prompt
    assert "1.75" in prompt


def test_review_works_without_metrics(mock_llm):
    copilot = StrategyCopilot(llm=mock_llm)
    result = copilot.review("def algo(): return 42")
    assert result == "Add a 2% stop loss to each position."


def test_review_prompt_mentions_trading_review(mock_llm):
    copilot = StrategyCopilot(llm=mock_llm)
    copilot.review("code")
    prompt = mock_llm.call.call_args[0][0]
    assert "strategy" in prompt.lower() or "review" in prompt.lower()
