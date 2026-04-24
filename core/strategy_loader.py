from pathlib import Path
from core.adapters.python_adapter import PythonAdapter

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
            raise NotImplementedError("MT5 adapter not yet implemented")
        if language == "cpp":
            raise NotImplementedError("C++ adapter not yet implemented")
        if language == "rust":
            raise NotImplementedError("Rust adapter not yet implemented")
