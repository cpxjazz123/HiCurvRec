"""The formal Stage 2 launch must reach a real process.

`tools/genrec-mcp/server.py` ends every launch with

    sys.executable, "-m", "tools.research.stage2.bootstrap", ...

The module was deleted by the `tools/research` cleanup (9ba91c7b) while that call stayed, so
every candidate with a passing smoke died with `No module named
tools.research.stage2.bootstrap` and started no training at all.  The smoke passes first, so
the failure was invisible unless the run log was read.  These tests are the reproduction.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools.research.stage2 import bootstrap  # noqa: E402


def test_the_module_the_server_invokes_imports_and_parses_its_flags():
    done = subprocess.run([sys.executable, "-m", "tools.research.stage2.bootstrap", "--help"],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    for flag in ("--entry", "--result-dir", "--python", "--visible-devices",
                 "--world-size", "--port"):
        assert flag in done.stdout, flag


def test_four_rank_layout_forks_torchrun_on_the_given_port():
    command = bootstrap.build_command(Path("/tree/train_rqvae.py"),
                                      "/env/bin/python3.9", 4, 50202)
    assert command[0] == "/env/bin/python3.9"
    assert "--standalone" in command
    assert "--nproc_per_node=4" in command
    assert "--master_port=50202" in command
    assert command[-1] == "/tree/train_rqvae.py"


def test_one_card_layout_is_the_bare_entrypoint_with_the_batch_kept_by_the_trainer():
    command = bootstrap.build_command(Path("/tree/train_rqvae.py"),
                                      "/env/bin/python3.9", 1, 50202)
    assert command == ["/env/bin/python3.9", "/tree/train_rqvae.py"]


def test_the_allocated_card_travels_through_the_child_environment():
    env = bootstrap.build_env({"PATH": "/bin"}, "2")
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["PATH"] == "/bin"
    # An empty allocation must not clear a pin the operator set.
    assert "CUDA_VISIBLE_DEVICES" not in bootstrap.build_env({"PATH": "/bin"}, "")


def test_a_missing_entrypoint_is_refused_instead_of_starting_nothing():
    code = bootstrap.main(["--entry", str(ROOT / "no-such-trainer.py"),
                           "--result-dir", str(ROOT / "results" / "research" / "issue238"),
                           "--python", sys.executable, "--world-size", "1", "--port", "50202"])
    assert code == 2
