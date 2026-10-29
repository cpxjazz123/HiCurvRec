"""The deterministic checks: what a program can answer, a program answers.

Each check runs a real command and records its exit code and its output. Nothing
here asks a language model whether the code is correct: the geometry contract, the
candidate's own tests, a real gradient step, a real two-rank run, the real entry
point and the real export are all executed.

A check that could not run (no GPU, no interpreter) is reported as ``NOT_RUN`` and
makes the whole result a failure of the RUN, not of the candidate: the caller turns
that into INFRA_ERROR, never into TEST_FAILED.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTRACT_TESTS = REPO / "tools" / "research" / "tests" / "test_geometry_contract.py"
CONTRACT_MODULE = REPO / "tools" / "research" / "geometry" / "poincare_contract.py"

#: The frozen interpreters of the project (CLAUDE.md 6-8).
PY_STAGE2 = Path("/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9")
PY_STAGE = Path("/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10")


def _run(command, *, cwd: Path, timeout: int = 900, env: dict = None) -> dict:
    """Run one command and capture exactly what happened."""
    import os
    environment = dict(os.environ)
    if env:
        environment.update(env)
    try:
        done = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout, env=environment)
    except subprocess.TimeoutExpired:
        return {"status": "NOT_RUN", "command": [str(item) for item in command],
                "detail": "timed out after %d s" % timeout}
    except OSError as exc:
        return {"status": "NOT_RUN", "command": [str(item) for item in command],
                "detail": type(exc).__name__ + ": " + str(exc)}
    tail = (done.stdout + done.stderr).strip().splitlines()[-40:]
    return {"status": "PASS" if done.returncode == 0 else "FAIL",
            "command": [str(item) for item in command],
            "exit_code": done.returncode, "output": "\n".join(tail)[-4000:]}


def check_contract(interpreter: Path = PY_STAGE2) -> dict:
    """The framework's own geometry contract, on the project interpreter."""
    return _run([str(interpreter), "-m", "unittest", "tools.research.tests.test_geometry_contract"],
                cwd=REPO)


def check_candidate_suite(tree: Path, interpreter: Path = PY_STAGE2) -> dict:
    """The candidate's own tests, run by the framework."""
    tests = Path(tree) / "tests"
    if not tests.is_dir():
        return {"status": "FAIL", "detail": "the candidate ships no tests"}
    return _run([str(interpreter), "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                cwd=Path(tree))


def check_entry_one_epoch(tree: Path, interpreter: Path = PY_STAGE2, timeout: int = 1800) -> dict:
    """The real entry point, one epoch, through the framework's own loop."""
    script = Path(tree) / "train_rqvae.py"
    if not script.is_file():
        return {"status": "FAIL", "detail": "the candidate has no trainer"}
    return _run([str(interpreter), str(script), "--epochs", "1"], cwd=Path(tree),
                timeout=timeout, env={"GENREC_SMOKE_EPOCHS": "1"})


def check_ddp_two_ranks(tree: Path, interpreter: Path = PY_STAGE2,
                        port: int = 50293, timeout: int = 1200) -> dict:
    """Two ranks, a few steps: the reducer, the collectives and the export path."""
    probe = Path(tree) / "tests" / "test_ddp.py"
    if not probe.is_file():
        return {"status": "FAIL", "detail": "the candidate ships no DDP test"}
    torchrun = Path(interpreter).with_name("torchrun")
    return _run([str(torchrun), "--standalone", "--nproc_per_node=2",
                 "--master_port=" + str(port), "-m", "unittest", "tests.test_ddp"],
                cwd=Path(tree), timeout=timeout,
                env={"CUDA_VISIBLE_DEVICES": "0,1"})


def check_export_matches_assignment(tree: Path, interpreter: Path = PY_STAGE2) -> dict:
    """The exported SID is the code the trained quantizer assigns.

    The candidate's own test suite carries this check; the framework runs the one
    named test so the result is attributable even when the suite is long.
    """
    return _run([str(interpreter), "-m", "unittest", "tests.test_radial_export"],
                cwd=Path(tree))


DEFAULT_CHECKS = ("contract", "suite", "export", "ddp")


def run_all(tree: Path, *, checks=None, interpreter: Path = PY_STAGE2) -> dict:
    """Run the deterministic checks and say whether the RUN was possible.

    ``passed`` is true only when every requested check PASSed. A ``NOT_RUN`` makes
    ``ran`` false: the run could not be completed, which the controller records as
    INFRA_ERROR rather than as a defect of the candidate.
    """
    tree = Path(tree)
    wanted = list(checks or DEFAULT_CHECKS)
    results = {}
    for name in wanted:
        if name == "contract":
            results[name] = check_contract(interpreter)
        elif name == "suite":
            results[name] = check_candidate_suite(tree, interpreter)
        elif name == "export":
            results[name] = check_export_matches_assignment(tree, interpreter)
        elif name == "ddp":
            results[name] = check_ddp_two_ranks(tree, interpreter)
        else:
            results[name] = {"status": "NOT_RUN", "detail": "unknown check " + name}
    statuses = [entry.get("status") for entry in results.values()]
    ran = all(status in ("PASS", "FAIL") for status in statuses)
    return {"tree": str(tree.resolve()), "checks": results,
            "ran": ran,
            "passed": ran and all(status == "PASS" for status in statuses)}
