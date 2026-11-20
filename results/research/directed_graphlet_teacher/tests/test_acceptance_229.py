"""Issue #229: the single-command acceptance must stay green."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEACHER = Path(__file__).resolve().parent.parent


def test_acceptance_script_exits_zero():
    result = subprocess.run([sys.executable, str(TEACHER / "accept_teacher_229.py")],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verdict: DONE_WITH_DECLARATIONS" in result.stdout
    assert "FAIL" not in result.stdout
