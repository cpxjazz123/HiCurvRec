"""The checks every candidate must pass, whatever its mechanism is.

A candidate may fail the experiment. It must not fail because its entry point
cannot run, because the reconstruction gradient never reaches the encoder,
because the four ranks deadlock, or because the SID it exports is not a valid
identifier. Those four things are the same question for every mechanism, so they
are asked by the framework and not by the candidate's own smoke test.

Each check returns a dict with a ``status`` of ``PASS``, ``FAIL`` or
``NOT_RUN`` and a ``detail`` that a reviewer can read. ``NOT_RUN`` is not a
pass: the admission treats it as missing evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import contract


def _scratch_dir(tree: Path) -> Path:
    """Where a conformance run writes: inside results, never inside the tree.

    A check that writes into the candidate tree pollutes the very snapshot the
    review hashes, and a directory inside a tree can break the marker that
    resolves the repository root.
    """
    root = contract.resolve_root()
    scratch = root.joinpath("results", "stage2_RQ-VAE", "_conformance", Path(tree).name)
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def _python_for() -> str:
    """The Stage 2 interpreter, as the frozen contract sets it."""
    return "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


def check_constants(tree: Path | str) -> dict:
    """The candidate's frozen constants match the table, and its launcher is ours.

    A candidate that changes the epochs, the batch, the learning rate or the seed
    is not the same experiment as the published arms, and a candidate that forks
    its own four-rank job reintroduces the failures the launcher exists to
    prevent.
    """
    tree = Path(tree).resolve()
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        return {"status": "FAIL", "detail": "the entry point is missing", "name": "constants"}
    text = entry.read_text(encoding="utf-8")
    problems: list = []
    for name, value in (("EPOCHS", contract.FROZEN["epochs"]),
                        ("BATCH_SIZE_PER_RANK", contract.FROZEN["batch_size_per_rank"]),
                        ("LR", contract.FROZEN["learning_rate"]),
                        ("WEIGHT_DECAY", contract.FROZEN["weight_decay"]),
                        ("CODEBOOK_NUM", contract.FROZEN["codebook_num"]),
                        ("CODEBOOK_DIM", contract.FROZEN["codebook_dim"]),
                        ("GRADIENT_CLIP_NORM", contract.FROZEN["gradient_clip_norm"]),
                        ("SEED", contract.FROZEN["seed"])):
        marker = name + " "
        line = next((row for row in text.splitlines()
                     if row.startswith(marker) and "=" in row), "")
        if not line:
            problems.append("the frozen constant " + name + " is not declared at the top level")
            continue
        _, _, raw = line.partition("=")
        try:
            found = eval(raw.split("#")[0].strip())  # noqa: S307 - a declared constant, not user input
        except Exception:                                     # noqa: BLE001
            problems.append("the constant " + name + " is not a literal")
            continue
        if found != value:
            problems.append(name + " is " + repr(found) + ", the frozen value is " + repr(value))
    # The candidate may name the interpreter, but it may not build the argv of a
    # four-rank job: those two flags are the launcher's.
    if "--nproc_per_node=" in text or "--master_port=" in text:
        problems.append("the candidate builds its own four-rank argv instead of "
                        "calling the framework launcher")
    if problems:
        return {"status": "FAIL", "name": "constants", "detail": "; ".join(problems)}
    return {"status": "PASS", "name": "constants",
            "detail": "the frozen constants match and the launcher is the framework's"}


def check_entry(tree: Path | str) -> dict:
    """The real entry point runs one epoch in this process and exports a SID."""
    tree = Path(tree).resolve()
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        return {"status": "FAIL", "name": "entry", "detail": "the entry point is missing"}
    scratch = _scratch_dir(tree)
    command = [_python_for(), "-c",
               "import os, sys;"
               "os.environ['" + contract.SINGLE_GPU_ENV + "'] = '1';"
               "sys.path.insert(0, '" + str(tree) + "');"
               "import train_rqvae as t;"
               "t.EPOCHS = 1;"
               "t.RESULT_DIR = __import__('pathlib').Path('" + str(scratch) + "');"
               "t.OUTPUT_SID = t.RESULT_DIR / 'sids_for_hgrec.npy';"
               "t.OUTPUT_JSON = t.RESULT_DIR / 'item_sids.json';"
               "t.CHECKPOINT = t.RESULT_DIR / 'rqvae_best.pth';"
               "t.LOG_DIR = t.RESULT_DIR / 'logs';"
               "t.main()"]
    try:
        done = subprocess.run(command, cwd=str(tree), capture_output=True, text=True,
                              timeout=3600,
                              env={**contract.launch_env(), contract.SINGLE_GPU_ENV: "1"})
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "name": "entry", "detail": "one epoch did not finish in an hour"}
    if done.returncode != 0:
        tail = (done.stdout + done.stderr).strip().splitlines()[-12:]
        return {"status": "FAIL", "name": "entry",
                "detail": "the entry point exited " + str(done.returncode) + ": " + " | ".join(tail)}
    sid = scratch / "item_sids.json"
    if not sid.is_file():
        return {"status": "FAIL", "name": "entry", "detail": "one epoch wrote no item_sids.json"}
    return {"status": "PASS", "name": "entry", "detail": "one epoch runs and exports the identifier"}


def check_identifier(tree: Path | str) -> dict:
    """The exported identifier is technically valid: unique, in range, four tokens."""
    tree = Path(tree).resolve()
    scratch = _scratch_dir(tree)
    sid = scratch / "item_sids.json"
    if not sid.is_file():
        return {"status": "NOT_RUN", "name": "identifier",
                "detail": "the entry check did not produce an identifier"}
    try:
        rows = json.loads(sid.read_text())
    except (OSError, ValueError) as exc:
        return {"status": "FAIL", "name": "identifier", "detail": "unreadable: " + str(exc)}
    values = list(rows.values()) if isinstance(rows, dict) else list(rows)
    if not values:
        return {"status": "FAIL", "name": "identifier", "detail": "the identifier is empty"}
    length = len(values[0])
    if any(len(row) != length for row in values):
        return {"status": "FAIL", "name": "identifier", "detail": "the tokens are ragged"}
    unique = len({tuple(row) for row in values})
    if unique != len(values):
        return {"status": "FAIL", "name": "identifier",
                "detail": str(len(values) - unique) + " identifiers repeat"}
    return {"status": "PASS", "name": "identifier",
            "detail": str(unique) + " unique identifiers of " + str(length) + " tokens"}


def check_four_rank(tree: Path | str) -> dict:
    """The four-rank step finishes and a rank-0 evaluation does not wait."""
    tree = Path(tree).resolve()
    script = (tree / "ddp_check.py")
    if not script.is_file():
        return {"status": "NOT_RUN", "name": "four_rank",
                "detail": "the candidate ships no four-rank script"}
    torchrun = Path(_python_for()).with_name("torchrun")
    attempts = []
    for attempt in range(2):
        try:
            done = subprocess.run(
                [str(torchrun), "--standalone", "--nproc_per_node=" + str(contract.WORLD_SIZE),
                 "--master_port=" + str(contract.STAGE2_PORT + 10 + attempt), str(script)],
                cwd=str(tree), capture_output=True, text=True, timeout=1200,
                env={**contract.launch_env(), "CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "name": "four_rank",
                    "detail": "the four-rank step did not finish: a collective mismatch"}
        if done.returncode == 0 and "DDP_PASS" in done.stdout:
            detail = "the four ranks finish and the rank-0 evaluation does not wait"
            if attempts:
                detail += " (a first attempt failed transiently: " + attempts[0][:200] + ")"
            return {"status": "PASS", "name": "four_rank", "detail": detail}
        tail = (done.stdout + done.stderr).strip().splitlines()[-40:]
        attempts.append("exit " + str(done.returncode) + ": " + " | ".join(tail))
        # A deadlock does not fix itself, and a second attempt would hang the
        # admission for another twenty minutes: only a crash is retried.
        if "Traceback" in " ".join(tail) and "NCCL" not in " ".join(tail):
            break
    return {"status": "FAIL", "name": "four_rank", "detail": " | ".join(attempts[-1:])}


def run_all(tree: Path | str) -> dict:
    """The conformance report of one candidate tree."""
    tree = Path(tree).resolve()
    checks = [check_constants(tree), check_entry(tree), check_identifier(tree),
              check_four_rank(tree)]
    passed = all(check["status"] == "PASS" for check in checks)
    return {"tree": str(tree), "passed": passed, "checks": checks}
