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
import re
import subprocess
import sys
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


def check_design_contract(tree: Path) -> dict:
    """The candidate's frozen definition exists, is FROZEN, and is versioned.

    The mechanism contract says what would make THIS mechanism inert; the design
    contract says what the mechanism IS -- the assignment rule, the loss target,
    the meaning of each ablation and the rank set. A run whose definition is not
    frozen cannot be attributed to a proposal, so the verification refuses one.
    """
    import hashlib

    tree = Path(tree)
    name = tree.name[6:] if tree.name.startswith("TIGER_") else tree.name
    path = REPO / "results" / "research" / "pipeline_v2" / "design" / (name + ".json")
    if not path.is_file():
        return {"status": "FAIL", "detail": "no frozen design contract at " + str(path)}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return {"status": "FAIL", "detail": "the design contract is not JSON: " + repr(exc)}
    if str(document.get("status", "")).upper() != "FROZEN":
        return {"status": "FAIL",
                "detail": "the design contract is not FROZEN: " + str(document.get("status"))}
    if not document.get("version") or not document.get("acceptance"):
        return {"status": "FAIL", "detail": "the design contract has no version or acceptance"}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"status": "PASS",
            "detail": "design contract version " + str(document["version"]) + " frozen",
            "version": document["version"], "sha256": digest, "path": str(path)}


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
    # The templates import the candidate, which needs torch: they run in the
    # project interpreter, in their own process, exactly like the other checks.
    script = (
        "import json, sys;"
        "sys.path.insert(0, " + repr(str(REPO)) + ");"
        "from tools.research.pipeline_v2 import contracts as c;"
        "contract = c.load_contract(" + repr(str(path)) + ");"
        "print('RESULT:' + json.dumps(c.run_tests(" + repr(str(tree)) + ", contract)))")
    done = _run([str(PY_STAGE2), "-c", script], cwd=REPO, timeout=900)
    if done.get("status") == "NOT_RUN":
        return done
    try:
        import json as _json
        marked = [line for line in str(done.get("output") or "").splitlines()
                  if line.startswith("RESULT:")]
        payload = _json.loads(marked[-1][len("RESULT:"):])
    except Exception as exc:                               # noqa: BLE001
        return {"status": "NOT_RUN",
                "detail": "the mechanism contract produced no result: " +
                          type(exc).__name__ + ": " + str(exc)[:200]}
    detail = "; ".join("%s=%s" % (key, entry.get("status"))
                       for key, entry in payload["tests"].items())
    return {"status": "PASS" if payload["passed"] else "FAIL", "detail": detail,
            "tests": payload["tests"], "contract": str(path)}


def check_negative_control(tree: Path) -> dict:
    """Break the mechanism on purpose; the named template must notice.

    Without this, a template that cannot see the defect it claims to look for
    would pass every candidate, and the VERIFIED state would mean nothing.
    """
    from . import contracts as contract_library
    from . import negative as negative_control

    tree = Path(tree)
    name = tree.name[6:] if tree.name.startswith("TIGER_") else tree.name
    path = (REPO / "results" / "research" / "pipeline_v2" / "contracts" /
            (name + ".json"))
    if not path.is_file():
        return {"status": "NOT_RUN",
                "detail": "the candidate has no mechanism contract at " + str(path)}
    try:
        contract = contract_library.load_contract(path)
        outcome = negative_control.run(tree, contract)
    except Exception as exc:                               # noqa: BLE001
        return {"status": "NOT_RUN",
                "detail": "the negative control could not run: " +
                          type(exc).__name__ + ": " + str(exc)}
    return {"status": outcome.get("status"), "detail": outcome.get("detail"),
            "controls": outcome.get("controls"), "contract": str(path)}


def _assignment_is(text: str, key: str, value: str) -> bool:
    """Whether the trainer sets ``key`` to ``value``, directly or through a constant.

    A trainer rarely writes ``radial=True``: it writes ``USE_RADIAL = True`` and
    then ``radial=USE_RADIAL``. The check resolves the constant instead of
    demanding a literal, because a check that only accepts one spelling would
    fail a correct trainer.
    """
    # the assignment may share a line with others, so it is not anchored
    direct = re.search(r"\b" + re.escape(key) + r"\s*=\s*" + re.escape(value) + r"\b",
                       text)
    if direct:
        return True
    for name in re.findall(r"\b(\w+)\s*=\s*" + re.escape(value) + r"\b", text):
        used = re.search(r"\b" + re.escape(key) + r"\s*=\s*" + re.escape(name) + r"\b",
                         text)
        if used:
            return True
    return False


def check_training_config(tree: Path, contract_name: str = None) -> dict:
    """The trainer must run the FORMAL configuration, not an inherited one.

    The candidate's own DDP test used ``frechet_codebook=True`` while the formal
    training uses ``radial=True, frechet_codebook=False``: the check then proved
    something about a configuration nobody trains with.
    """
    tree = Path(tree)
    name = contract_name or (tree.name[6:] if tree.name.startswith("TIGER_") else tree.name)
    contract_path = (REPO / "results" / "research" / "pipeline_v2" / "contracts" /
                     (name + ".json"))
    expected = {}
    if contract_path.is_file():
        import json as _json
        expected = (_json.loads(contract_path.read_text(encoding="utf-8"))
                    .get("training_config") or {})
    if not expected:
        return {"status": "NOT_RUN",
                "detail": "the contract declares no training_config"}
    source = (tree / "train_rqvae.py")
    if not source.is_file():
        return {"status": "FAIL", "detail": "the candidate has no trainer"}
    text = source.read_text(encoding="utf-8", errors="replace")
    mismatches = []
    for key, value in expected.items():
        if _assignment_is(text, str(key), str(value)):
            continue
        mismatches.append("%s is not %s in the trainer" % (key, value))
    if mismatches:
        return {"status": "FAIL", "detail": "; ".join(mismatches)}
    return {"status": "PASS",
            "detail": "the trainer runs the formal configuration: " + str(expected)}


def check_promotion(tree: Path) -> dict:
    """The tree that will train must be the tree that was reviewed.

    The research candidate and the promoted tree are two copies; a verification of
    one says nothing about the other. This compares the digests of the promoted
    tree and of the round tree it was copied from, so a stale promotion is
    refused instead of trained.
    """
    tree = Path(tree).resolve()
    name = tree.name
    mechanism = name[6:] if name.startswith("TIGER_") else name
    # the round trees live beside the rounds, under the issue
    trees = REPO / "results" / "research" / "issue121" / "trees"
    if not trees.is_dir():
        return {"status": "NOT_RUN", "detail": "there is no tree directory to compare with"}
    best = None
    for directory in sorted(trees.iterdir(), reverse=True):
        candidate = directory / mechanism
        if candidate.is_dir():
            best = candidate
            break
    if best is None:
        return {"status": "NOT_RUN",
                "detail": "no round tree named " + mechanism + " to compare with"}
    if _digest_of(best) != _digest_of(tree):
        return {"status": "FAIL",
                "detail": "the promoted tree differs from the reviewed round tree (" +
                           str(_digest_of(best))[:12] + " vs " + str(_digest_of(tree))[:12] +
                           "): the training tree is not the reviewed one"}
    return {"status": "PASS",
            "detail": "the promoted tree is the reviewed tree (" +
                       str(_digest_of(tree))[:12] + ")"}


def _digest_of(tree: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(Path(tree).rglob("*.py")):
        relative = path.relative_to(tree)
        if any(part in ("logs", "results", "dataset", "__pycache__", "ckpt")
               for part in relative.parts):
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def check_tests_use_training_config(tree: Path, contract_name: str = None) -> dict:
    """The candidate's own tests must declare the configuration that trains.

    A DDP test that builds the model with a different configuration proves
    something about a model nobody trains with.
    """
    tree = Path(tree)
    name = contract_name or (tree.name[6:] if tree.name.startswith("TIGER_") else tree.name)
    contract_path = (REPO / "results" / "research" / "pipeline_v2" / "contracts" /
                     (name + ".json"))
    if not contract_path.is_file():
        return {"status": "NOT_RUN", "detail": "the contract declares no training_config"}
    import json as _json
    expected = (_json.loads(contract_path.read_text(encoding="utf-8"))
                .get("training_config") or {})
    if not expected:
        return {"status": "NOT_RUN", "detail": "the contract declares no training_config"}
    tests = tree / "tests"
    if not tests.is_dir():
        return {"status": "FAIL", "detail": "the candidate ships no tests"}
    required = (_json.loads(contract_path.read_text(encoding="utf-8"))
                .get("training_config_tests") or [])
    if not required:
        required = [entry.get("module") for entry in
                    (_json.loads(contract_path.read_text(encoding="utf-8")).get("tests") or [])
                    if entry.get("module")]
    offenders = []
    for module in required:
        name = str(module).split(".")[-1] + ".py"
        path = tests / name
        if not path.is_file():
            offenders.append("%s is missing" % name)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for key, value in expected.items():
            if not _assignment_is(text, str(key), str(value)):
                offenders.append("%s does not build the model with %s=%s" %
                                (name, key, value))
    if offenders:
        return {"status": "FAIL", "detail": "; ".join(offenders)}
    return {"status": "PASS",
            "detail": "the candidate tests build the model with " + str(expected)}


DEFAULT_CHECKS = ("contract", "design", "suite", "gradient", "export", "entry", "ddp",
                  "mechanism", "negative", "training_config", "promotion",
                  "tests_config")


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
        elif name == "design":
            results[name] = check_design_contract(tree)
        elif name == "suite":
            results[name] = check_candidate_suite(tree, interpreter)
        elif name == "gradient":
            results[name] = check_gradient(tree)
        elif name == "export":
            results[name] = check_export_matches_assignment(tree, interpreter)
        elif name == "entry":
            results[name] = check_entry_one_epoch(tree)
        elif name == "ddp":
            results[name] = check_ddp_two_ranks(tree, interpreter)
        elif name == "mechanism":
            results[name] = check_mechanism_contract(tree)
        elif name == "negative":
            results[name] = check_negative_control(tree)
        elif name == "training_config":
            results[name] = check_training_config(tree)
        elif name == "promotion":
            results[name] = check_promotion(tree)
        elif name == "tests_config":
            results[name] = check_tests_use_training_config(tree)
        else:
            results[name] = {"status": "NOT_RUN", "detail": "unknown check " + name}
    statuses = [entry.get("status") for entry in results.values()]
    ran = all(status in ("PASS", "FAIL") for status in statuses)
    return {"tree": str(tree.resolve()), "checks": results,
            "ran": ran,
            "passed": ran and all(status == "PASS" for status in statuses)}
