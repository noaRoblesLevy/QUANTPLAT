import numpy as np
from typing import List


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    arr = np.array(returns, dtype=float)
    if arr.std() == 0:
        return 0.0
    excess = arr - risk_free_rate / 252
    return float(np.sqrt(252) * excess.mean() / arr.std())


def sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate / 252
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / downside_std)


def max_drawdown(equity_curve: List[float]) -> float:
    arr = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    with np.errstate(invalid='ignore'):
        dd = np.where(peak > 0, (arr - peak) / peak, 0.0)
    return float(dd.min())


def calmar_ratio(equity_curve: List[float], annual_return: float) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(annual_return / mdd)


def profit_factor(pl_list: List[float]) -> float:
    wins = sum(p for p in pl_list if p > 0)
    losses = abs(sum(p for p in pl_list if p < 0))
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(pl_list: List[float]) -> float:
    if not pl_list:
        return 0.0
    return float(sum(1 for p in pl_list if p > 0) / len(pl_list))


def expectancy(pl_list: List[float]) -> float:
    if not pl_list:
        return 0.0
    wr = win_rate(pl_list)
    wins = [p for p in pl_list if p > 0]
    losses = [p for p in pl_list if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(abs(np.mean(losses))) if losses else 0.0
    return float(wr * avg_win - (1 - wr) * avg_loss)


def compute_all(equity_curve: List[float], pl_list: List[float],
                risk_free_rate: float = 0.0) -> dict:
    arr = np.array(equity_curve, dtype=float)
    returns = (np.diff(arr) / arr[:-1]).tolist()
    n_days = len(returns)
    annual_return = float((arr[-1] / arr[0]) ** (252 / max(n_days, 1)) - 1)
    wins = [p for p in pl_list if p > 0]
    losses = [p for p in pl_list if p < 0]
    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve, annual_return),
        "profit_factor": profit_factor(pl_list),
        "win_rate": win_rate(pl_list),
        "expectancy": expectancy(pl_list),
        "annual_return": annual_return,
        "total_trades": len(pl_list),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }
