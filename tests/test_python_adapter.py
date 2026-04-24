import json
import pytest
from pathlib import Path
import tempfile
from core.adapters.python_adapter import PythonAdapter


@pytest.fixture
def sample_strategy_file(tmp_path):
    f = tmp_path / "my_strategy.py"
    f.write_text("# dummy strategy\nclass MyAlgo: pass\n")
    return f


def test_prepare_creates_project_dir(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    assert project_dir.is_dir()


def test_prepare_copies_strategy_as_main(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    main_file = project_dir / "main.py"
    assert main_file.exists()
    assert "dummy strategy" in main_file.read_text()


def test_prepare_creates_config_json(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    config_file = project_dir / "config.json"
    assert config_file.exists()
    config = json.loads(config_file.read_text())
    assert config["algorithm-language"] == "Python"
    assert "parameters" in config


def test_prepare_config_includes_strategy_name(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    config = json.loads((project_dir / "config.json").read_text())
    assert config["algorithm-name"] == "my_strategy"


def test_cleanup_removes_project_dir(sample_strategy_file):
    adapter = PythonAdapter()
    project_dir = adapter.prepare(sample_strategy_file)
    assert project_dir.exists()
    adapter.cleanup()
    assert not project_dir.exists()


def test_prepare_with_parameters(sample_strategy_file):
    adapter = PythonAdapter()
    params = {"fast_ema": 30, "slow_ema": 200}
    project_dir = adapter.prepare(sample_strategy_file, parameters=params)
    config = json.loads((project_dir / "config.json").read_text())
    assert config["parameters"]["fast_ema"] == 30
    assert config["parameters"]["slow_ema"] == 200
