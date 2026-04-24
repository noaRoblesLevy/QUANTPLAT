import pytest
from pathlib import Path
from core.strategy_loader import StrategyLoader
from core.adapters.python_adapter import PythonAdapter


@pytest.fixture
def loader():
    return StrategyLoader()


def test_load_python_strategy_returns_python_adapter(tmp_path, loader):
    f = tmp_path / "my_algo.py"
    f.write_text("# algo")
    adapter = loader.load(f)
    assert isinstance(adapter, PythonAdapter)


def test_load_detects_by_extension(tmp_path, loader):
    f = tmp_path / "algo.py"
    f.write_text("# algo")
    adapter = loader.load(f)
    assert adapter is not None


def test_load_raises_for_unsupported_extension(tmp_path, loader):
    f = tmp_path / "algo.java"
    f.write_text("// java")
    with pytest.raises(ValueError, match="Unsupported strategy language"):
        loader.load(f)


def test_load_raises_if_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load(Path("/nonexistent/strategy.py"))


def test_load_mt5_not_yet_implemented(tmp_path, loader):
    f = tmp_path / "algo.mq5"
    f.write_text("// mt5")
    with pytest.raises(NotImplementedError, match="MT5"):
        loader.load(f)


def test_load_cpp_not_yet_implemented(tmp_path, loader):
    f = tmp_path / "algo.cpp"
    f.write_text("// cpp")
    with pytest.raises(NotImplementedError, match="C++"):
        loader.load(f)


def test_load_rust_not_yet_implemented(tmp_path, loader):
    f = tmp_path / "algo.rs"
    f.write_text("// rust")
    with pytest.raises(NotImplementedError, match="Rust"):
        loader.load(f)
