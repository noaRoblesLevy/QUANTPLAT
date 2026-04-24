from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
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
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)
