import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.lean_runner import LeanRunner, LeanRunError


@pytest.fixture
def mock_project_dir(tmp_path):
    project = tmp_path / "my_strategy"
    project.mkdir()
    (project / "main.py").write_text("# algo")
    (project / "config.json").write_text('{"algorithm-name": "my_strategy"}')
    return project


@pytest.fixture
def mock_backtest_output(tmp_path, sample_lean_output):
    output_dir = tmp_path / "backtests" / "20250101_120000"
    output_dir.mkdir(parents=True)
    result_file = output_dir / "backtestResults.json"
    result_file.write_text(json.dumps(sample_lean_output))
    return tmp_path


def test_run_calls_lean_subprocess(mock_project_dir, mock_backtest_output, sample_lean_output):
    runner = LeanRunner(lean_workspace=mock_backtest_output)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter([b"Launching LEAN...\n", b"Backtest complete.\n"]))
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        result = runner.run(mock_project_dir)
    mock_popen.assert_called_once()
    args = mock_popen.call_args[0][0]
    assert args[0] == "lean"
    assert args[1] == "backtest"


def test_run_returns_parsed_results(mock_project_dir, mock_backtest_output, sample_lean_output):
    runner = LeanRunner(lean_workspace=mock_backtest_output)
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
    runner = LeanRunner(lean_workspace=tmp_path)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter([b"Error: strategy failed\n"]))
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        with pytest.raises(LeanRunError):
            runner.run(mock_project_dir)


def test_run_streams_output_to_callback(mock_project_dir, mock_backtest_output):
    runner = LeanRunner(lean_workspace=mock_backtest_output)
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
