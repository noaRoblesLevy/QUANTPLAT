import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from analytics.metrics import compute_all
from analytics.monte_carlo import run as mc_run
from db import get_session, init_db
from db.models import BacktestRun


class ReportGenerator:
    def __init__(self, results_dir: Optional[Path] = None, engine=None):
        self._results_dir = Path(results_dir) if results_dir else Path("results")
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._engine = engine or init_db()

    def generate(self, strategy_name: str, strategy_path: str,
                 lean_output: Dict[str, Any], n_mc: int = 1000) -> Dict[str, Any]:
        pl_list = lean_output["pl_list"]
        equity_curve = lean_output["equity_curve"]

        starting_equity = equity_curve[0] if equity_curve else 50000.0
        metrics = compute_all(equity_curve, pl_list) if equity_curve and pl_list else {}
        mc = mc_run(pl_list, n_simulations=n_mc, starting_equity=starting_equity) if pl_list else {}

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{strategy_name}.json"
        report_path = self._results_dir / filename

        report = {
            "strategy_name": strategy_name,
            "strategy_path": strategy_path,
            "timestamp": timestamp,
            "metrics": metrics,
            "monte_carlo": mc,
            "raw_statistics": lean_output.get("raw_statistics", {}),
            "runtime_statistics": lean_output.get("runtime_statistics", {}),
            "lean_results_path": lean_output.get("results_path", ""),
            "report_path": str(report_path),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._save_to_db(strategy_name, strategy_path, metrics, str(report_path))
        return report

    def _save_to_db(self, strategy_name: str, strategy_path: str,
                    metrics: Dict[str, Any], report_path: str) -> None:
        run = BacktestRun(
            strategy_name=strategy_name,
            strategy_path=strategy_path,
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            expectancy=metrics.get("expectancy", 0.0),
            annual_return=metrics.get("annual_return", 0.0),
            total_trades=metrics.get("total_trades", 0),
            results_path=report_path,
        )
        with get_session(self._engine) as session:
            session.add(run)
