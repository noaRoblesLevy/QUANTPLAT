import re
from datetime import datetime
from pathlib import Path
from typing import Optional

_DEFAULT_VAULT = Path(__file__).parent.parent / "vault"


class VaultWriter:
    def __init__(self, vault_path=None):
        self._vault = Path(vault_path) if vault_path else _DEFAULT_VAULT

    def write_backtest(
        self,
        strategy_name: str,
        run_date: datetime,
        metrics: dict,
        ai_summary: Optional[str] = None,
        results_path: Optional[str] = None,
    ) -> Path:
        date_str = run_date.strftime("%Y-%m-%d")
        output_path = self._vault / "backtests" / f"{date_str}-{strategy_name}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self._backtest_template(strategy_name, run_date, metrics, ai_summary, results_path),
            encoding="utf-8",
        )
        self._update_hub_table(
            section_header="| Datum | Strategie | Sharpe | Max DD | Profit Factor |",
            section_separator="|---|---|---|---|---|",
            new_row=(
                f"| {date_str} | {strategy_name} "
                f"| {metrics.get('sharpe_ratio', 0):.2f} "
                f"| {metrics.get('max_drawdown', 0)*100:.1f}% "
                f"| {metrics.get('profit_factor', 0):.2f} |"
            ),
        )
        return output_path

    def write_optimization(
        self,
        strategy_name: str,
        run_date: datetime,
        mode: str,
        best_params: Optional[dict] = None,
        best_sharpe: Optional[float] = None,
        n_trials: int = 0,
    ) -> Path:
        date_str = run_date.strftime("%Y-%m-%d")
        output_path = self._vault / "optimizations" / f"{date_str}-{strategy_name}-{mode}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self._optimization_template(strategy_name, run_date, mode, best_params, best_sharpe, n_trials),
            encoding="utf-8",
        )
        sharpe_str = f"{best_sharpe:.2f}" if best_sharpe is not None else "N/A"
        self._update_hub_table(
            section_header="| Datum | Strategie | Mode | Best Sharpe |",
            section_separator="|---|---|---|---|",
            new_row=f"| {date_str} | {strategy_name} | {mode} | {sharpe_str} |",
        )
        return output_path

    def _backtest_template(
        self,
        name: str,
        date: datetime,
        metrics: dict,
        ai_summary: Optional[str],
        results_path: Optional[str],
    ) -> str:
        sharpe = metrics.get("sharpe_ratio", 0)
        dd = metrics.get("max_drawdown", 0)
        wr = metrics.get("win_rate", 0)
        pf = metrics.get("profit_factor", 0)
        trades = metrics.get("total_trades", 0)

        lines = [
            f"# Backtest: {name}",
            f"**Datum:** {date.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Metrics",
            "",
            "| Metric | Waarde |",
            "|---|---|",
            f"| Sharpe Ratio | {sharpe:.3f} |",
            f"| Max Drawdown | {dd*100:.1f}% |",
            f"| Win Rate | {wr*100:.1f}% |",
            f"| Profit Factor | {pf:.3f} |",
            f"| Trades | {trades} |",
            "",
        ]
        if ai_summary:
            lines += ["## AI Analyse", "", ai_summary, ""]
        if results_path:
            lines += ["## Resultaten", "", f"`{results_path}`", ""]
        lines += ["---", "*Gegenereerd door QUANTPLAT*"]
        return "\n".join(lines)

    def _optimization_template(
        self,
        name: str,
        date: datetime,
        mode: str,
        best_params: Optional[dict],
        best_sharpe: Optional[float],
        n_trials: int,
    ) -> str:
        sharpe_str = f"{best_sharpe:.3f}" if best_sharpe is not None else "N/A"
        lines = [
            f"# Optimalisatie: {name} ({mode})",
            f"**Datum:** {date.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Samenvatting",
            "",
            f"- **Mode:** {mode}",
            f"- **Trials:** {n_trials}",
            f"- **Best Sharpe:** {sharpe_str}",
            "",
        ]
        if best_params:
            lines += ["## Beste Parameters", ""]
            for k, v in best_params.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
        lines += ["---", "*Gegenereerd door QUANTPLAT*"]
        return "\n".join(lines)

    def _update_hub_table(
        self,
        section_header: str,
        section_separator: str,
        new_row: str,
    ) -> None:
        hub_path = self._vault / "00-Hub.md"
        if not hub_path.exists():
            return
        content = hub_path.read_text(encoding="utf-8")
        # Remove placeholder row if present
        content = re.sub(r'\| _\(leeg\)_ \|[^\n]*\n?', '', content)
        # Find the header+separator and insert new row after separator
        sep_pos = content.find(section_separator, content.find(section_header))
        if sep_pos != -1:
            insert_at = content.index("\n", sep_pos) + 1
            content = content[:insert_at] + new_row + "\n" + content[insert_at:]
        hub_path.write_text(content, encoding="utf-8")
