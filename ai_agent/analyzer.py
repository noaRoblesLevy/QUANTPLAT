from typing import Optional, List
from ai_agent.provider import LLMProvider


class PostBacktestAnalyzer:
    def __init__(self, llm=None):
        self._llm = llm or LLMProvider()

    def analyze(
        self,
        metrics: dict,
        pl_list: Optional[List[float]] = None,
    ) -> str:
        prompt = self._build_prompt(metrics, pl_list)
        return self._llm.call(prompt)

    def _build_prompt(
        self,
        metrics: dict,
        pl_list: Optional[List[float]],
    ) -> str:
        lines = [
            "You are a quantitative trading performance analyst.",
            "Analyze the following backtest results and provide a concise summary.",
            "",
            "Backtest metrics:",
        ]
        for key, val in metrics.items():
            lines.append(f"  {key}: {val}")

        if pl_list:
            wins = [p for p in pl_list if p > 0]
            losses = [p for p in pl_list if p < 0]
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            lines.extend([
                "",
                "Trade statistics:",
                f"  Total trades: {len(pl_list)}",
                f"  Winning trades: {len(wins)}",
                f"  Losing trades: {len(losses)}",
                f"  Avg win: ${avg_win:.2f}",
                f"  Avg loss: ${avg_loss:.2f}",
            ])

        lines.extend([
            "",
            "Provide a 2-3 sentence summary covering:",
            "1. What worked well",
            "2. Main weakness or risk",
            "3. One specific next step to improve performance",
        ])
        return "\n".join(lines)
