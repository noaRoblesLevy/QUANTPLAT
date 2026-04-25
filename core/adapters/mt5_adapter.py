import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any, Dict


class MT5Adapter:
    def __init__(self):
        self._project_dir: Optional[Path] = None

    def prepare(self, strategy_path: Path, parameters: Optional[Dict[str, Any]] = None) -> Path:
        strategy_path = Path(strategy_path)
        self._project_dir = Path(tempfile.mkdtemp(prefix="quantplat_mt5_"))
        shutil.copy2(strategy_path, self._project_dir / "main.mq5")
        return self._project_dir

    def run(self, project_dir: Path, on_output=None) -> dict:
        raise NotImplementedError(
            "MT5 Strategy Tester cannot be triggered programmatically via the Python API. "
            "Run the backtest manually in the MT5 GUI (Strategy Tester), export the results, "
            "and import them via the History page."
        )

    def cleanup(self) -> None:
        if self._project_dir and self._project_dir.exists():
            shutil.rmtree(self._project_dir)
            self._project_dir = None
