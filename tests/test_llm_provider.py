import pytest
from unittest.mock import MagicMock, patch
import sys
from ai_agent.provider import LLMProvider


def test_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    provider = LLMProvider()
    assert provider._provider == "ollama"


def test_provider_reads_env_var(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "claude")
    provider = LLMProvider()
    assert provider._provider == "claude"


def test_call_ollama_returns_string(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    mock_response = {"message": {"content": "test response"}}
    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = mock_response
    monkeypatch.setitem(sys.modules, "ollama", mock_ollama)
    provider = LLMProvider()
    result = provider.call("hello")
    assert result == "test response"


def test_call_claude_returns_string(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_content = MagicMock()
    mock_content.text = "claude response"
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
    provider = LLMProvider()
    result = provider.call("hello")
    assert result == "claude response"


def test_call_openai_returns_string(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_choice = MagicMock()
    mock_choice.message.content = "openai response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    monkeypatch.setitem(sys.modules, "openai", mock_openai)
    provider = LLMProvider()
    result = provider.call("hello")
    assert result == "openai response"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "unknown_provider")
    provider = LLMProvider()
    with pytest.raises(ValueError, match="Unknown AI provider"):
        provider.call("hello")


def test_ollama_model_defaults_to_llama3(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    provider = LLMProvider()
    assert provider._ollama_model == "llama3"


def test_ollama_model_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    provider = LLMProvider()
    assert provider._ollama_model == "mistral"
