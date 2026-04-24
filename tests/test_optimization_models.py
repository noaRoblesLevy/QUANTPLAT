import json
import pytest
from db import init_db, get_session
from db.models import OptimizationRun, OptimizationTrial


@pytest.fixture
def engine(tmp_path):
    return init_db(f"sqlite:///{tmp_path}/test.db")


def test_create_optimization_run(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="my_algo",
            strategy_path="strategies/my_algo.py",
            mode="grid",
            n_trials=4,
        )
        session.add(run)
        session.flush()
        assert run.id is not None
        assert run.best_sharpe is None
        assert run.best_params is None
        assert run.created_at is not None


def test_create_optimization_trial(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="grid",
            n_trials=2,
        )
        session.add(run)
        session.flush()
        trial = OptimizationTrial(
            run_id=run.id,
            trial_number=0,
            params=json.dumps({"sma_fast": 10, "sma_slow": 50}),
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.15,
            win_rate=0.55,
            profit_factor=1.8,
        )
        session.add(trial)
        session.flush()
        assert trial.id is not None
        assert trial.run_id == run.id


def test_optimization_trial_nullable_metrics(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="ai",
            n_trials=1,
        )
        session.add(run)
        session.flush()
        trial = OptimizationTrial(
            run_id=run.id,
            trial_number=0,
            params=json.dumps({"sma_fast": 5}),
        )
        session.add(trial)
        session.flush()
        assert trial.sharpe_ratio is None
        assert trial.win_rate is None


def test_optimization_run_update_best(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="grid",
            n_trials=4,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    with get_session(engine) as session:
        run = session.get(OptimizationRun, run_id)
        run.best_sharpe = 1.5
        run.best_params = json.dumps({"sma_fast": 10, "sma_slow": 50})

    with get_session(engine) as session:
        run = session.get(OptimizationRun, run_id)
        assert run.best_sharpe == 1.5
        assert json.loads(run.best_params)["sma_fast"] == 10


def test_multiple_trials_per_run(engine):
    from sqlalchemy import select
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="grid",
            n_trials=3,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        for i in range(3):
            session.add(OptimizationTrial(
                run_id=run_id,
                trial_number=i,
                params=json.dumps({"sma_fast": i * 5 + 5}),
                sharpe_ratio=float(i),
            ))

    with get_session(engine) as session:
        trials = session.execute(
            select(OptimizationTrial).where(OptimizationTrial.run_id == run_id)
        ).scalars().all()
        assert len(trials) == 3


def test_walk_forward_trial_window_label(engine):
    with get_session(engine) as session:
        run = OptimizationRun(
            strategy_name="algo",
            strategy_path="strategies/algo.py",
            mode="walk_forward",
            n_trials=2,
        )
        session.add(run)
        session.flush()
        trial = OptimizationTrial(
            run_id=run.id,
            trial_number=0,
            params=json.dumps({"sma_fast": 10}),
            sharpe_ratio=0.9,
            window_label="window_0_train",
        )
        session.add(trial)
        session.flush()
        assert trial.window_label == "window_0_train"
