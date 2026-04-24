import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any


class PythonAdapter:
    def __init__(self):
        self._project_dir: Optional[Path] = None

    def prepare(self, strategy_path: Path,
                parameters: Optional[Dict[str, Any]] = None) -> Path:
        strategy_path = Path(strategy_path)
        self._project_dir = Path(tempfile.mkdtemp(prefix="quantplat_"))
        main_file = self._project_dir / "main.py"
        shutil.copy2(strategy_path, main_file)
        config = {
            "algorithm-language": "Python",
            "algorithm-name": strategy_path.stem,
            "description": "",
            "parameters": parameters or {},
            "cloud-id": 0,
            "local-id": 0,
        }
        (self._project_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        return self._project_dir

    def cleanup(self) -> None:
        if self._project_dir and self._project_dir.exists():
            shutil.rmtree(self._project_dir)
            self._project_dir = None
