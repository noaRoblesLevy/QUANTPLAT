import json
from pathlib import Path
from typing import List, Optional

from db import get_session, init_db
from db.models import BacktestRun, OptimizationRun, OptimizationTrial

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = init_db()
    return _engine


def load_all_runs(engine=None) -> List[BacktestRun]:
    eng = engine or _get_engine()
    with get_session(eng) as session:
        runs = (
            session.query(BacktestRun)
            .order_by(BacktestRun.created_at.desc())
            .all()
        )
        return runs


def load_report(report_path: str) -> Optional[dict]:
    p = Path(report_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_all_optimization_runs(engine=None) -> List[OptimizationRun]:
    eng = engine or _get_engine()
    with get_session(eng) as session:
        return (
            session.query(OptimizationRun)
            .order_by(OptimizationRun.created_at.desc())
            .all()
        )


def load_optimization_trials(run_id: int, engine=None) -> List[OptimizationTrial]:
    eng = engine or _get_engine()
    with get_session(eng) as session:
        return (
            session.query(OptimizationTrial)
            .filter(OptimizationTrial.run_id == run_id)
            .order_by(OptimizationTrial.trial_number)
            .all()
        )
