"""The command surface must remain useful before installing simulator dependencies."""

import json
import os
from pathlib import Path
import subprocess
import sys


def run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "praxis", *args], capture_output=True, text=True, env=env, timeout=20
    )


def test_each_command_explains_itself():
    for args in [
        (),
        ("--help",),
        ("train", "--help"),
        ("evaluate", "--help"),
        ("doctor", "--help"),
    ]:
        result = run_cli(*args)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_doctor_reports_missing_assets_without_loading_weights(tmp_path):
    env = dict(os.environ, PRAXIS_CHECKPOINT_DIR=str(tmp_path))
    result = run_cli("doctor", "--json", env=env)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert Path(report["checkpoint_directory"]) == tmp_path
    assert not any(report["assets"].values())
    assert "not tested" in report["note"]


def test_unknown_command_is_rejected():
    result = run_cli("does-not-exist")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
