import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List

import optuna

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from ai_agent.provider import LLMProvider
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_prompt(
    top_trials: List[dict],
    param_ranges: Dict[str, Tuple],
) -> str:
    lines = [
        "You are a quantitative trading strategy optimizer.",
        "Top performing parameter combinations so far:",
    ]
    for t in top_trials:
        lines.append(f"  Sharpe {t['sharpe_ratio']:.3f}: {t['params']}")
    lines.append("\nParameter ranges:")
    for name, (low, high, ptype) in param_ranges.items():
        lines.append(f"  {name}: {low} to {high} ({ptype})")
    lines.append(
        "\nSuggest ONE parameter combination that might improve the Sharpe ratio."
        "\nRespond with ONLY a JSON dict, e.g.: {\"sma_fast\": 15, \"sma_slow\": 80}"
    )
    return "\n".join(lines)


def _parse_llm_response(
    response: str,
    param_ranges: Dict[str, Tuple],
) -> Optional[dict]:
    match = re.search(r'\{[^}]+\}', response)
    if not match:
        return None
    try:
        suggested = json.loads(match.group())
        result = {}
        for name, (low, high, ptype) in param_ranges.items():
            if name in suggested:
                val = float(suggested[name])
                val = max(float(low), min(float(high), val))
                result[name] = int(val) if ptype == "int" else val
        return result if result else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class AIOptimizer:
    def __init__(self, runner=None, report_gen=None, loader=None, llm=None,
                 engine=None, database_url=None):
        self._runner = runner or LeanRunner()
        self._report_gen = report_gen or ReportGenerator()
        self._loader = loader or StrategyLoader()
        self._llm = llm or LLMProvider()
        self._engine = engine or init_db(database_url or "sqlite:///quantplat.db")

    def run(
        self,
        strategy_path: str,
        param_ranges: Dict[str, Tuple],
        n_trials: int = 50,
        starting_capital: float = 50_000,
        n_mc: int = 1000,
        llm_interval: int = 10,
        on_trial: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationRun:
        strategy_path = Path(strategy_path)

        with get_session(self._engine) as session:
            opt_run = OptimizationRun(
                strategy_name=strategy_path.stem,
                strategy_path=str(strategy_path),
                mode="ai",
                n_trials=n_trials,
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

        trial_history: List[dict] = []
        best_sharpe: Optional[float] = None
        best_params: Optional[dict] = None

        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            params: dict = {}
            for name, (low, high, ptype) in param_ranges.items():
                if ptype == "int":
                    params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))

            metrics = self._run_single(
                strategy_path, params, starting_capital, n_mc, run_id, trial.number
            )
            sharpe = metrics.get("sharpe_ratio") or 0.0
            trial_history.append({"params": params, "sharpe_ratio": sharpe})

            if on_trial:
                on_trial(trial.number, {"params": params, **metrics})

            nonlocal best_sharpe, best_params
            sharpe_real = metrics.get("sharpe_ratio")
            if sharpe_real is not None and (best_sharpe is None or sharpe_real > best_sharpe):
                best_sharpe = sharpe_real
                best_params = params

            if (trial.number + 1) % llm_interval == 0:
                top = sorted(trial_history, key=lambda t: t["sharpe_ratio"], reverse=True)[:5]
                try:
                    response = self._llm.call(_build_prompt(top, param_ranges))
                    suggested = _parse_llm_response(response, param_ranges)
                    if suggested:
                        study.enqueue_trial(suggested)
                except Exception:
                    pass

            return sharpe

        study.optimize(objective, n_trials=n_trials)

        with get_session(self._engine) as session:
            opt_run = session.get(OptimizationRun, run_id)
            opt_run.best_sharpe = best_sharpe
            opt_run.best_params = json.dumps(best_params) if best_params else None
            session.add(opt_run)

        with get_session(self._engine) as session:
            return session.get(OptimizationRun, run_id)

    def _run_single(
        self,
        strategy_path: Path,
        params: dict,
        starting_capital: float,
        n_mc: int,
        run_id: int,
        trial_num: int,
    ) -> dict:
        adapter = self._loader.load(strategy_path)
        try:
            project_dir = adapter.prepare(strategy_path, parameters=params)
            lean_output = self._runner.run(project_dir)
            report = self._report_gen.generate(
                strategy_path.stem, str(strategy_path), lean_output, n_mc=n_mc
            )
            metrics = report.get("metrics", {})
        except LeanRunError:
            metrics = {}
        finally:
            adapter.cleanup()

        with get_session(self._engine) as session:
            session.add(OptimizationTrial(
                run_id=run_id,
                trial_number=trial_num,
                params=json.dumps(params),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
            ))

        return metrics
