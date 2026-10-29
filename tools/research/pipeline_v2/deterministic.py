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


def check_entry_one_epoch(tree: Path) -> dict:
    """The real entry point for one epoch, through the framework's own smoke.

    The candidate's trainer has a frozen EPOCHS constant and takes no command-line
    argument; a second entry invented here would be a second training path. The
    framework's ``stage2.checks.check_entry`` already runs the real trainer for one
    epoch in a scratch directory with the single-card contract, so that is what V2
    calls.
    """
    from ..stage2 import checks as stage2_checks
    try:
        result = stage2_checks.check_entry(Path(tree))
    except Exception as exc:                               # noqa: BLE001
        return {"status": "NOT_RUN", "detail": type(exc).__name__ + ": " + str(exc)}
    return {"status": result.get("status"), "detail": result.get("detail"),
            "name": "entry"}


def check_gradient(tree: Path) -> dict:
    """A real batch moves the parameters the mechanism claims to move.

    The candidate's own gradient test does this; the framework runs it by name so
    the result is attributable even when the suite is long.
    """
    tests = Path(tree) / "tests"
    names = []
    for candidate in ("test_radial_wiring", "test_mechanism"):
        if (tests / (candidate + ".py")).is_file():
            names.append("tests." + candidate)
    if not names:
        return {"status": "FAIL", "detail": "the candidate has no gradient test"}
    return _run([str(PY_STAGE2), "-m", "unittest"] + names, cwd=Path(tree))


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


def check_mechanism_contract(tree: Path) -> dict:
    """Run the mechanism contract's own tests, if the candidate has one.

    The contract is written at ideation, before the code: it names, from a fixed
    template library, what would make THIS mechanism inert. It is data, so a new
    mechanism fills parameters instead of writing a new verifier.
    """
    from . import contracts as contract_library

    tree = Path(tree)
    name = tree.name[6:] if tree.name.startswith("TIGER_") else tree.name
    path = (REPO / "results" / "research" / "pipeline_v2" / "contracts" /
            (name + ".json"))
    if not path.is_file():
        return {"status": "NOT_RUN",
                "detail": "the candidate has no mechanism contract at " + str(path)}
    try:
        contract = contract_library.load_contract(path)
        outcome = contract_library.run_tests(tree, contract)
    except Exception as exc:                               # noqa: BLE001
        return {"status": "NOT_RUN",
                "detail": "the mechanism contract could not run: " +
                          type(exc).__name__ + ": " + str(exc)}
    detail = "; ".join("%s=%s" % (key, entry.get("status"))
                       for key, entry in outcome["tests"].items())
    return {"status": "PASS" if outcome["passed"] else "FAIL", "detail": detail,
            "tests": outcome["tests"], "contract": str(path)}


DEFAULT_CHECKS = ("contract", "suite", "gradient", "export", "ddp", "mechanism")


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
        elif name == "gradient":
            results[name] = check_gradient(tree)
        elif name == "export":
            results[name] = check_export_matches_assignment(tree, interpreter)
        elif name == "ddp":
            results[name] = check_ddp_two_ranks(tree, interpreter)
        elif name == "mechanism":
            results[name] = check_mechanism_contract(tree)
        else:
            results[name] = {"status": "NOT_RUN", "detail": "unknown check " + name}
    statuses = [entry.get("status") for entry in results.values()]
    ran = all(status in ("PASS", "FAIL") for status in statuses)
    return {"tree": str(tree.resolve()), "checks": results,
            "ran": ran,
            "passed": ran and all(status == "PASS" for status in statuses)}
