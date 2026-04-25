import pytest
from datetime import datetime
from pathlib import Path
from vault_sync.writer import VaultWriter

_METRICS = {
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.15,
    "win_rate": 0.55,
    "profit_factor": 1.8,
    "total_trades": 42,
}


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "backtests").mkdir()
    (tmp_path / "optimizations").mkdir()
    hub = tmp_path / "00-Hub.md"
    hub.write_text(
        "# Hub\n\n"
        "## Recente Backtests\n\n"
        "| Datum | Strategie | Sharpe | Max DD | Profit Factor |\n"
        "|---|---|---|---|---|\n"
        "| _(leeg)_ | | | | |\n\n"
        "## Recente Optimalisaties\n\n"
        "| Datum | Strategie | Mode | Best Sharpe |\n"
        "|---|---|---|---|\n"
        "| _(leeg)_ | | | |\n",
        encoding="utf-8",
    )
    return VaultWriter(vault_path=tmp_path)


def test_write_backtest_creates_file(vault):
    path = vault.write_backtest("my_algo", datetime(2026, 4, 25, 10, 0), _METRICS)
    assert path.exists()


def test_write_backtest_filename_contains_date_and_name(vault):
    path = vault.write_backtest("test_algo", datetime(2026, 4, 25, 10, 0), _METRICS)
    assert "2026-04-25" in path.name
    assert "test_algo" in path.name


def test_write_backtest_content_has_metrics(vault):
    path = vault.write_backtest("algo", datetime(2026, 4, 25), _METRICS)
    content = path.read_text(encoding="utf-8")
    assert "1.2" in content or "1.20" in content  # sharpe
    assert "42" in content  # total trades


def test_write_backtest_content_has_ai_summary(vault):
    path = vault.write_backtest(
        "algo", datetime(2026, 4, 25), _METRICS,
        ai_summary="Strong momentum strategy, reduce leverage."
    )
    content = path.read_text(encoding="utf-8")
    assert "Strong momentum strategy" in content


def test_write_backtest_updates_hub(vault, tmp_path):
    vault.write_backtest("algo", datetime(2026, 4, 25), _METRICS)
    hub_content = (tmp_path / "00-Hub.md").read_text(encoding="utf-8")
    assert "2026-04-25" in hub_content
    assert "algo" in hub_content


def test_write_optimization_creates_file(vault):
    path = vault.write_optimization(
        "algo", datetime(2026, 4, 25), "grid",
        best_params={"sma_fast": 10}, best_sharpe=1.5, n_trials=4,
    )
    assert path.exists()


def test_write_optimization_filename_contains_mode(vault):
    path = vault.write_optimization("algo", datetime(2026, 4, 25), "walk_forward",
                                     best_sharpe=1.2, n_trials=10)
    assert "walk_forward" in path.name


def test_write_optimization_content_has_sharpe(vault):
    path = vault.write_optimization("algo", datetime(2026, 4, 25), "ai",
                                     best_sharpe=1.8, n_trials=50)
    content = path.read_text(encoding="utf-8")
    assert "1.8" in content or "1.80" in content


def test_write_optimization_updates_hub(vault, tmp_path):
    vault.write_optimization("algo", datetime(2026, 4, 25), "grid", best_sharpe=1.3, n_trials=4)
    hub_content = (tmp_path / "00-Hub.md").read_text(encoding="utf-8")
    assert "grid" in hub_content
    assert "algo" in hub_content


def test_write_backtest_creates_backtests_dir_if_missing(tmp_path):
    writer = VaultWriter(vault_path=tmp_path)
    path = writer.write_backtest("algo", datetime(2026, 4, 25), _METRICS)
    assert path.exists()
