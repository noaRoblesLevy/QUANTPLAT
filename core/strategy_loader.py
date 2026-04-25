from pathlib import Path
from core.adapters.python_adapter import PythonAdapter
from core.adapters.mt5_adapter import MT5Adapter
from core.adapters.cpp_adapter import CppAdapter
from core.adapters.rust_adapter import RustAdapter

_LANGUAGE_MAP = {
    ".py":  "python",
    ".mq5": "mt5",
    ".cpp": "cpp",
    ".rs":  "rust",
}


class StrategyLoader:
    def load(self, strategy_path: Path):
        strategy_path = Path(strategy_path)
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
        if not strategy_path.is_file():
            raise ValueError(f"Strategy path is not a file: {strategy_path}")
        ext = strategy_path.suffix.lower()
        language = _LANGUAGE_MAP.get(ext)
        if language is None:
            raise ValueError(
                f"Unsupported strategy language: '{ext}'. "
                f"Supported: {list(_LANGUAGE_MAP.keys())}"
            )
        if language == "python":
            return PythonAdapter()
        if language == "mt5":
            return MT5Adapter()
        if language == "cpp":
            return CppAdapter()
        return RustAdapter()
