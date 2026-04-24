import numpy as np
from typing import List, Dict


def run(pl_list: List[float], n_simulations: int = 1000,
        starting_equity: float = 50000.0) -> Dict:
    rng = np.random.default_rng(42)
    pl_arr = np.array(pl_list, dtype=float)
    n_trades = len(pl_arr)

    sim_equity = np.empty((n_simulations, n_trades + 1))
    sim_equity[:, 0] = starting_equity
    for i in range(n_simulations):
        shuffled = rng.permutation(pl_arr)
        sim_equity[i, 1:] = starting_equity + np.cumsum(shuffled)

    final_equities = sim_equity[:, -1]
    sorted_idx = np.argsort(final_equities)

    max_drawdowns = np.array([_max_drawdown(sim_equity[i]) for i in range(n_simulations)])
    probability_of_ruin = float(np.mean(final_equities < starting_equity * 0.5))

    return {
        "percentiles": {
            "p5":  sim_equity[sorted_idx[int(0.05 * n_simulations)]].tolist(),
            "p50": sim_equity[sorted_idx[int(0.50 * n_simulations)]].tolist(),
            "p95": sim_equity[sorted_idx[int(0.95 * n_simulations)]].tolist(),
        },
        "final_equity": {
            "mean": float(final_equities.mean()),
            "std":  float(final_equities.std()),
            "p5":   float(np.percentile(final_equities, 5)),
            "p50":  float(np.percentile(final_equities, 50)),
            "p95":  float(np.percentile(final_equities, 95)),
        },
        "max_drawdown": {
            "mean": float(max_drawdowns.mean()),
            "p95":  float(np.percentile(max_drawdowns, 95)),
        },
        "probability_of_ruin": probability_of_ruin,
        "n_simulations": n_simulations,
        "n_trades": n_trades,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    with np.errstate(invalid='ignore'):
        dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    return float(dd.min())
