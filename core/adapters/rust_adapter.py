import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Any, Dict

from core.lean_runner import LeanRunError


class RustAdapter:
    def __init__(self):
        self._project_dir: Optional[Path] = None

    def prepare(self, strategy_path: Path, parameters: Optional[Dict[str, Any]] = None) -> Path:
        strategy_path = Path(strategy_path)
        self._project_dir = Path(tempfile.mkdtemp(prefix="quantplat_rust_"))
        try:
            shutil.copy2(strategy_path, self._project_dir / "main.rs")
            if parameters:
                (self._project_dir / "params.json").write_text(
                    json.dumps(parameters), encoding="utf-8"
                )
        except Exception:
            self.cleanup()
            raise
        return self._project_dir

    def run(self, project_dir: Path, on_output=None) -> dict:
        project_dir = Path(project_dir)
        ext = ".exe" if sys.platform == "win32" else ""
        binary = project_dir / f"strategy{ext}"
        src = project_dir / "main.rs"
        compile_result = subprocess.run(
            ["rustc", "-O", "-o", str(binary), str(src)],
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise LeanRunError(f"Rust compilation failed:\n{compile_result.stderr}")
        params_file = project_dir / "params.json"
        cmd = [str(binary)]
        if params_file.exists():
            cmd += ["--params", str(params_file)]
        run_result = subprocess.run(cmd, capture_output=True, text=True)
        if run_result.returncode != 0:
            raise LeanRunError(f"Rust strategy failed:\n{run_result.stderr}")
        try:
            return json.loads(run_result.stdout)
        except json.JSONDecodeError as e:
            raise LeanRunError(f"Rust strategy output is not valid JSON: {e}")

    def cleanup(self) -> None:
        if self._project_dir and self._project_dir.exists():
            shutil.rmtree(self._project_dir)
            self._project_dir = None
