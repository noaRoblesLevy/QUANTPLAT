import json
import itertools
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

from core.strategy_loader import StrategyLoader
from core.lean_runner import LeanRunner, LeanRunError
from analytics.report import ReportGenerator
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial


def _split_windows(
    start_date: date,
    end_date: date,
    n_windows: int,
    train_ratio: float,
) -> List[Tuple[date, date, date, date]]:
    """Returns list of (train_start, train_end, test_start, test_end)."""
    total_days = (end_date - start_date).days
    window_days = total_days // n_windows
    train_days = int(window_days * train_ratio)

    windows = []
    for i in range(n_windows):
        w_start = start_date + timedelta(days=i * window_days)
        w_end = start_date + timedelta(days=(i + 1) * window_days)
        train_end = w_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        windows.append((w_start, train_end, test_start, w_end))
    return windows


class WalkForwardOptimizer:
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
        start_date: date,
        end_date: date,
        n_windows: int = 5,
        train_ratio: float = 0.7,
        starting_capital: float = 50_000,
        n_mc: int = 1000,
        on_trial: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationRun:
        strategy_path = Path(strategy_path)
        windows = _split_windows(start_date, end_date, n_windows, train_ratio)
        param_names = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in param_names]))

        total_trials = n_windows * (len(combos) + 1)

        with get_session(self._engine) as session:
            opt_run = OptimizationRun(
                strategy_name=strategy_path.stem,
                strategy_path=str(strategy_path),
                mode="walk_forward",
                n_trials=total_trials,
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

        best_sharpe: Optional[float] = None
        best_params: Optional[dict] = None
        trial_num = 0

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            window_best_sharpe: Optional[float] = None
            window_best_params: Optional[dict] = None

            for combo in combos:
                params = dict(zip(param_names, combo))
                params["__wf_start"] = train_start.isoformat()
                params["__wf_end"] = train_end.isoformat()

                metrics = self._run_single(
                    strategy_path, params, starting_capital, n_mc, run_id, trial_num,
                    window_label=f"window_{w_idx}_train",
                )
                if on_trial:
                    on_trial(trial_num, metrics)
                trial_num += 1

                sharpe = metrics.get("sharpe_ratio")
                if sharpe is not None and (window_best_sharpe is None or sharpe > window_best_sharpe):
                    window_best_sharpe = sharpe
                    window_best_params = {
                        k: v for k, v in params.items()
                        if not k.startswith("__wf_")
                    }

            if window_best_params:
                val_params = dict(window_best_params)
                val_params["__wf_start"] = test_start.isoformat()
                val_params["__wf_end"] = test_end.isoformat()

                metrics = self._run_single(
                    strategy_path, val_params, starting_capital, n_mc, run_id, trial_num,
                    window_label=f"window_{w_idx}_test",
                )
                if on_trial:
                    on_trial(trial_num, metrics)
                trial_num += 1

                sharpe = metrics.get("sharpe_ratio")
                if sharpe is not None and (best_sharpe is None or sharpe > best_sharpe):
                    best_sharpe = sharpe
                    best_params = window_best_params

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
        window_label: Optional[str] = None,
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
                params=json.dumps({k: v for k, v in params.items() if not k.startswith("__wf_")}),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
                window_label=window_label,
            ))

        return metrics
