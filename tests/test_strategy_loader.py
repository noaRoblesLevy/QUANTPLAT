import pytest
from pathlib import Path
from core.strategy_loader import StrategyLoader
from core.adapters.python_adapter import PythonAdapter
from core.adapters.mt5_adapter import MT5Adapter
from core.adapters.cpp_adapter import CppAdapter
from core.adapters.rust_adapter import RustAdapter


@pytest.fixture
def py_file(tmp_path):
    f = tmp_path / "algo.py"
    f.write_text("class MyAlgo: pass", encoding="utf-8")
    return f


@pytest.fixture
def mq5_file(tmp_path):
    f = tmp_path / "algo.mq5"
    f.write_text("// MT5 strategy", encoding="utf-8")
    return f


@pytest.fixture
def cpp_file(tmp_path):
    f = tmp_path / "algo.cpp"
    f.write_text("int main() { return 0; }", encoding="utf-8")
    return f


@pytest.fixture
def rs_file(tmp_path):
    f = tmp_path / "algo.rs"
    f.write_text("fn main() {}", encoding="utf-8")
    return f


def test_loader_returns_python_adapter(py_file):
    adapter = StrategyLoader().load(py_file)
    assert isinstance(adapter, PythonAdapter)


def test_loader_returns_mt5_adapter(mq5_file):
    adapter = StrategyLoader().load(mq5_file)
    assert isinstance(adapter, MT5Adapter)


def test_loader_returns_cpp_adapter(cpp_file):
    adapter = StrategyLoader().load(cpp_file)
    assert isinstance(adapter, CppAdapter)


def test_loader_returns_rust_adapter(rs_file):
    adapter = StrategyLoader().load(rs_file)
    assert isinstance(adapter, RustAdapter)


def test_loader_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        StrategyLoader().load(Path("nonexistent.py"))


def test_loader_raises_for_unsupported_extension(tmp_path):
    f = tmp_path / "algo.java"
    f.write_text("class Algo {}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        StrategyLoader().load(f)


def test_mt5_adapter_prepare_returns_path(mq5_file):
    adapter = MT5Adapter()
    project_dir = adapter.prepare(mq5_file)
    assert project_dir.exists()
    adapter.cleanup()
    assert not project_dir.exists()


def test_mt5_adapter_main_file_copied(mq5_file):
    adapter = MT5Adapter()
    project_dir = adapter.prepare(mq5_file)
    assert (project_dir / "main.mq5").exists()
    adapter.cleanup()


def test_cpp_adapter_prepare_copies_file(cpp_file):
    adapter = CppAdapter()
    project_dir = adapter.prepare(cpp_file)
    assert (project_dir / "main.cpp").exists()
    adapter.cleanup()


def test_rust_adapter_prepare_copies_file(rs_file):
    adapter = RustAdapter()
    project_dir = adapter.prepare(rs_file)
    assert (project_dir / "main.rs").exists()
    adapter.cleanup()


def test_mt5_adapter_run_raises_not_implemented(mq5_file):
    adapter = MT5Adapter()
    project_dir = adapter.prepare(mq5_file)
    with pytest.raises(NotImplementedError, match="Strategy Tester"):
        adapter.run(project_dir)
    adapter.cleanup()


def test_cpp_adapter_run_raises_on_missing_compiler(cpp_file, monkeypatch):
    from core.lean_runner import LeanRunError
    adapter = CppAdapter()
    project_dir = adapter.prepare(cpp_file)
    monkeypatch.setattr(
        "core.adapters.cpp_adapter.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("g++ not found")),
    )
    with pytest.raises((FileNotFoundError, LeanRunError)):
        adapter.run(project_dir)
    adapter.cleanup()


def test_rust_adapter_run_raises_on_missing_compiler(rs_file, monkeypatch):
    from core.lean_runner import LeanRunError
    adapter = RustAdapter()
    project_dir = adapter.prepare(rs_file)
    monkeypatch.setattr(
        "core.adapters.rust_adapter.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("rustc not found")),
    )
    with pytest.raises((FileNotFoundError, LeanRunError)):
        adapter.run(project_dir)
    adapter.cleanup()
