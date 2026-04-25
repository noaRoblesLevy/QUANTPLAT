import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.lean_runner import LeanRunner, LeanRunError


@pytest.fixture
def mock_project_dir(tmp_path):
    project = tmp_path / "my_strategy"
    project.mkdir()
    (project / "main.py").write_text(
        "from AlgorithmImports import *\nclass MyStrategy(QCAlgorithm): pass",
        encoding="utf-8",
    )
    (project / "config.json").write_text('{"algorithm-language": "Python"}')
    return project


@pytest.fixture
def mock_backtest_output(tmp_path, sample_lean_output):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result_file = results_dir / "backtestResults.json"
    result_file.write_text(json.dumps(sample_lean_output))
    return results_dir


def test_run_calls_docker(mock_project_dir, mock_backtest_output):
    runner = LeanRunner(results_dir=mock_backtest_output)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(
            return_value=iter([b"Launching LEAN...\n", b"Backtest complete.\n"])
        )
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        runner.run(mock_project_dir)
    mock_popen.assert_called_once()
    args = mock_popen.call_args[0][0]
    assert args[0] == "docker"
    assert args[1] == "run"


def test_run_returns_parsed_results(mock_project_dir, mock_backtest_output):
    runner = LeanRunner(results_dir=mock_backtest_output)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter([]))
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        result = runner.run(mock_project_dir)
    assert "pl_list" in result
    assert "equity_curve" in result
    assert "raw_statistics" in result
    assert "results_path" in result


def test_run_raises_on_nonzero_exit(mock_project_dir, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    runner = LeanRunner(results_dir=results_dir)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(
            return_value=iter([b"Error: strategy failed\n"])
        )
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        with pytest.raises(LeanRunError):
            runner.run(mock_project_dir)


def test_run_streams_output_to_callback(mock_project_dir, mock_backtest_output):
    runner = LeanRunner(results_dir=mock_backtest_output)
    received_lines = []
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(
            return_value=iter([b"line one\n", b"line two\n"])
        )
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        runner.run(mock_project_dir, on_output=received_lines.append)
    assert "line one\n" in received_lines
    assert "line two\n" in received_lines


def test_detect_algorithm_name_from_class(tmp_path):
    main_py = tmp_path / "main.py"
    main_py.write_text("class SmaCrossover(QCAlgorithm): pass", encoding="utf-8")
    runner = LeanRunner()
    assert runner._detect_algorithm_name(main_py) == "SmaCrossover"


def test_detect_algorithm_name_falls_back_to_dirname(tmp_path):
    project = tmp_path / "my_algo"
    project.mkdir()
    main_py = project / "main.py"
    main_py.write_text("# no class here", encoding="utf-8")
    runner = LeanRunner()
    assert runner._detect_algorithm_name(main_py) == "my_algo"


def test_build_lean_config_contains_required_keys():
    runner = LeanRunner()
    config = runner._build_lean_config("MyAlgo")
    assert config["algorithm-type-name"] == "MyAlgo"
    assert config["algorithm-language"] == "Python"
    assert config["environment"] == "backtesting"
    assert config["data-folder"] == "/Data/"
    assert config["results-destination-folder"] == "/Results/"


def test_parse_lean_output_extracts_pl_list(sample_lean_output):
    runner = LeanRunner()
    result = runner._parse_lean_output(sample_lean_output, results_path=Path("."))
    assert isinstance(result["pl_list"], list)
    assert len(result["pl_list"]) == len(sample_lean_output["profitLoss"])


def test_parse_lean_output_extracts_equity_curve(sample_lean_output):
    runner = LeanRunner()
    result = runner._parse_lean_output(sample_lean_output, results_path=Path("."))
    assert isinstance(result["equity_curve"], list)
    assert len(result["equity_curve"]) > 0
    assert result["equity_curve"][0] == 50000.0
