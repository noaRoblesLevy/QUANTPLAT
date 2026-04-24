import pytest
from pathlib import Path
from db import get_session, init_db
from db.models import BacktestRun


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = init_db(f"sqlite:///{db_path}")
    with get_session(engine) as session:
        yield session


def test_create_backtest_run(db_session):
    run = BacktestRun(
        strategy_name="my_algo",
        strategy_path="strategies/my_algo.py",
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        max_drawdown=-0.12,
        profit_factor=1.5,
        win_rate=0.45,
        expectancy=150.0,
        annual_return=0.22,
        total_trades=100,
        results_path="results/run_001.json",
    )
    db_session.add(run)
    db_session.commit()
    assert run.id is not None


def test_query_backtest_run(db_session):
    run = BacktestRun(
        strategy_name="test_strat",
        strategy_path="strategies/test.py",
        sharpe_ratio=0.8,
        sortino_ratio=1.1,
        max_drawdown=-0.20,
        profit_factor=1.2,
        win_rate=0.40,
        expectancy=80.0,
        annual_return=0.10,
        total_trades=50,
        results_path="results/run_002.json",
    )
    db_session.add(run)
    db_session.commit()
    found = db_session.query(BacktestRun).filter_by(strategy_name="test_strat").first()
    assert found is not None
    assert found.sharpe_ratio == 0.8


def test_backtest_run_created_at_set_automatically(db_session):
    run = BacktestRun(
        strategy_name="ts",
        strategy_path="s.py",
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        expectancy=0.0,
        annual_return=0.0,
        total_trades=0,
        results_path="r.json",
    )
    db_session.add(run)
    db_session.commit()
    assert run.created_at is not None
