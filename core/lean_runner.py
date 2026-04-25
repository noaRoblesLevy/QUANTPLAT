import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, Any, Optional

LEAN_IMAGE = "quantconnect/lean:latest"


class LeanRunError(Exception):
    pass


class LeanRunner:
    def __init__(self, data_dir: Optional[Path] = None, results_dir: Optional[Path] = None):
        root = Path(__file__).parent.parent
        self._data_dir = Path(data_dir) if data_dir else root / "data"
        self._results_dir = Path(results_dir) if results_dir else root / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, project_dir: Path,
            on_output: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        project_dir = Path(project_dir).resolve()
        algo_name = self._detect_algorithm_name(project_dir / "main.py")

        config_path = project_dir / "_lean_config.json"
        config_path.write_text(
            json.dumps(self._build_lean_config(algo_name), indent=2), encoding="utf-8"
        )
        try:
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{self._data_dir.resolve()}:/Data:ro",
                "-v", f"{project_dir}:/Algorithm",
                "-v", f"{self._results_dir.resolve()}:/Results",
                "-v", f"{config_path}:/Lean/Launcher/bin/Debug/config.json:ro",
                LEAN_IMAGE,
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        finally:
            config_path.unlink(missing_ok=True)

        results_file = self._find_results_file(algo_name)
        raw = json.loads(results_file.read_text(encoding="utf-8"))
        return self._parse_lean_output(raw, results_path=results_file)

    def _detect_algorithm_name(self, main_py: Path) -> str:
        source = main_py.read_text(encoding="utf-8")
        match = re.search(r"class\s+(\w+)\s*\(", source)
        if match:
            return match.group(1)
        return main_py.parent.name

    def _build_lean_config(self, algo_name: str) -> dict:
        return {
            "environment": "backtesting",
            "algorithm-type-name": algo_name,
            "algorithm-language": "Python",
            "algorithm-location": "/Algorithm/main.py",
            "data-folder": "/Data/",
            "results-destination-folder": "/Results/",
            "debugging": False,
            "log-handler": "QuantConnect.Logging.CompositeLogHandler",
            "messaging-handler": "QuantConnect.Messaging.Messaging",
            "job-queue-handler": "QuantConnect.Queues.JobQueue",
            "api-handler": "QuantConnect.Api.Api",
            "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
            "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
            "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
            "object-store": "QuantConnect.Lean.Engine.Storage.LocalObjectStore",
            "show-missing-data-logs": False,
        }

    def _find_results_file(self, algo_name: str) -> Path:
        # LEAN writes {AlgoName}.json to the results folder
        direct = self._results_dir / f"{algo_name}.json"
        if direct.exists():
            return direct
        # Fallback: pick most recently modified JSON that looks like results
        candidates = sorted(
            [p for p in self._results_dir.glob("*.json")
             if "order-events" not in p.name and "summary" not in p.name
             and "monitor" not in p.name and "requests" not in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise LeanRunError("No results JSON found in results directory")
        return candidates[0]

    def _parse_lean_output(self, raw: Dict, results_path: Path) -> Dict[str, Any]:
        pl_list = list(raw.get("profitLoss", {}).values())
        equity_points = (
            raw.get("charts", {})
            .get("Strategy Equity", {})
            .get("series", {})
            .get("Equity", {})
            .get("values", [])
        )
        equity_curve = self._extract_equity_curve(equity_points)
        return {
            "pl_list": pl_list,
            "equity_curve": equity_curve,
            "raw_statistics": raw.get("statistics", {}),
            "runtime_statistics": raw.get("runtimeStatistics", {}),
            "results_path": str(results_path),
        }

    @staticmethod
    def _extract_equity_curve(values: list) -> list:
        if not values:
            return []
        # LEAN v2.5+: flat array [timestamp, open, high, low, close, timestamp, ...]
        if isinstance(values[0], (int, float)):
            return [values[i + 4] for i in range(0, len(values) - 4, 5)]
        # Older LEAN format: list of {x: timestamp, y: value} dicts
        return [pt["y"] for pt in values]
