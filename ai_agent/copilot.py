from typing import Optional
from ai_agent.provider import LLMProvider


class StrategyCopilot:
    def __init__(self, llm=None):
        self._llm = llm or LLMProvider()

    def review(self, code: str, metrics: Optional[dict] = None) -> str:
        prompt = self._build_prompt(code, metrics)
        return self._llm.call(prompt)

    def _build_prompt(self, code: str, metrics: Optional[dict]) -> str:
        lines = [
            "You are a quantitative trading strategy code reviewer.",
            "Review the following trading strategy code and provide concrete, actionable suggestions.",
            "Focus on: risk management, entry/exit logic, parameter robustness, common pitfalls.",
            "",
            "Strategy code:",
            "```python",
            code,
            "```",
        ]
        if metrics:
            lines.append("\nBacktest metrics:")
            for key, val in metrics.items():
                lines.append(f"  {key}: {val}")
        lines.extend([
            "",
            "Provide 3-5 specific, actionable suggestions to improve this strategy.",
        ])
        return "\n".join(lines)
