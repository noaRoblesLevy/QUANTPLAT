import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, Any, Optional


class LeanRunError(Exception):
    pass


class LeanRunner:
    def __init__(self, lean_workspace: Optional[Path] = None):
        self._workspace = Path(lean_workspace) if lean_workspace else Path.cwd()

    def run(self, project_dir: Path,
            on_output: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        project_dir = Path(project_dir)
        cmd = ["lean", "backtest", str(project_dir)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self._workspace),
        )
        output_lines = []
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            output_lines.append(line)
            if on_output:
                on_output(line)
        proc.wait()
        if proc.returncode != 0:
            raise LeanRunError(
                f"LEAN exited with code {proc.returncode}.\n" + "".join(output_lines)
            )
        results_file = self._find_results_file()
        raw = json.loads(results_file.read_text(encoding="utf-8"))
        return self._parse_lean_output(raw, results_path=results_file)

    def _find_results_file(self) -> Path:
        backtest_dirs = sorted(
            (self._workspace / "backtests").glob("*/backtestResults.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not backtest_dirs:
            raise LeanRunError("No backtestResults.json found in workspace/backtests/")
        return backtest_dirs[0]

    def _parse_lean_output(self, raw: Dict, results_path: Path) -> Dict[str, Any]:
        pl_list = list(raw.get("profitLoss", {}).values())
        equity_points = (
            raw.get("charts", {})
            .get("Strategy Equity", {})
            .get("series", {})
            .get("Equity", {})
            .get("values", [])
        )
        equity_curve = [pt["y"] for pt in equity_points] if equity_points else []
        return {
            "pl_list": pl_list,
            "equity_curve": equity_curve,
            "raw_statistics": raw.get("statistics", {}),
            "runtime_statistics": raw.get("runtimeStatistics", {}),
            "results_path": str(results_path),
        }
