import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial


class GridSearchOptimizer:
    def __init__(self, runner=None, report_gen=None, loader=None, engine=None,
                 database_url=None):
        self._runner = runner or LeanRunner()
        self._report_gen = report_gen or ReportGenerator()
        self._loader = loader or StrategyLoader()
        self._engine = engine or init_db(database_url or "sqlite:///quantplat.db")

    def run(
        self,
        strategy_path: str,
        param_grid: Dict[str, List[Any]],
        starting_capital: float = 50_000,
        n_mc: int = 1000,
        on_trial: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationRun:
        strategy_path = Path(strategy_path)
        param_names = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in param_names]))

        with get_session(self._engine) as session:
            opt_run = OptimizationRun(
                strategy_name=strategy_path.stem,
                strategy_path=str(strategy_path),
                mode="grid",
                n_trials=len(combos),
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

        best_sharpe: Optional[float] = None
        best_params: Optional[dict] = None

        for trial_num, combo in enumerate(combos):
            params = dict(zip(param_names, combo))
            metrics = self._run_single(
                strategy_path, params, starting_capital, n_mc, run_id, trial_num
            )
            if on_trial:
                on_trial(trial_num, metrics)
            sharpe = metrics.get("sharpe_ratio")
            if sharpe is not None and (best_sharpe is None or sharpe > best_sharpe):
                best_sharpe = sharpe
                best_params = params

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
