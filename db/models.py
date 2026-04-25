from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, CheckConstraint, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name  = Column(String, nullable=False)
    strategy_path  = Column(String, nullable=False)
    sharpe_ratio   = Column(Float, nullable=False)
    sortino_ratio  = Column(Float, nullable=False)
    max_drawdown   = Column(Float, nullable=False)
    profit_factor  = Column(Float, nullable=False)
    win_rate       = Column(Float, nullable=False)
    expectancy     = Column(Float, nullable=False)
    annual_return  = Column(Float, nullable=False)
    total_trades   = Column(Integer, nullable=False)
    results_path   = Column(String, nullable=False)
    ai_summary     = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    __table_args__ = (
        CheckConstraint("mode IN ('grid', 'walk_forward', 'ai')", name="ck_opt_run_mode"),
    )

    id             = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name  = Column(String, nullable=False)
    strategy_path  = Column(String, nullable=False)
    mode           = Column(String, nullable=False)
    n_trials       = Column(Integer, nullable=False, default=0)
    best_sharpe    = Column(Float, nullable=True)
    best_params    = Column(String, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class OptimizationTrial(Base):
    __tablename__ = "optimization_trials"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    trial_number   = Column(Integer, nullable=False)
    params         = Column(String, nullable=False)
    sharpe_ratio   = Column(Float, nullable=True)
    sortino_ratio  = Column(Float, nullable=True)
    max_drawdown   = Column(Float, nullable=True)
    win_rate       = Column(Float, nullable=True)
    profit_factor  = Column(Float, nullable=True)
    window_label   = Column(String, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)
