import json
from pathlib import Path
from typing import List, Optional

from db import get_session, init_db
from db.models import BacktestRun

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
