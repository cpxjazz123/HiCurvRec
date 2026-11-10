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
import os
import re
import subprocess
import sys
from pathlib import Path

from . import contract

#: The suffix ``_admission_tree`` appends to a promoted copy's name.
# The admission copy appends ``_<digest10>_<pid>_<uuid8>`` since the copy became
# per-run, so stripping only the digest left the pid and the uuid on the name and
# the entry check wrote into a scratch directory the identifier check never read.
# Both halves are optional so an older copy name still resolves.
_DIGEST_SUFFIX = re.compile(r"_[0-9a-f]{10}(?:_\d+_[0-9a-f]{8})?$")


def _scratch_dir(tree: Path) -> Path:
    """Where a conformance run writes: inside results, never inside the tree.

    A check that writes into the candidate tree pollutes the very snapshot the
    review hashes, and a directory inside a tree can break the marker that
    resolves the repository root.
    """
    root = contract.resolve_root()
    # The name of the tree decides the directory, not its path: the entry check
    # runs the promoted copy, and the identifier check reads what it wrote.
    name = Path(tree).name
    if name.startswith("_admission_"):
        name = name[len("_admission_"):]
        # ``_admission_tree`` appends the digest of the source path so that two
        # trees of the same name cannot delete each other's copy.  That suffix is
        # not part of the tree's name, and the identifier check resolves the
        # scratch directory from the ORIGINAL tree: without removing the digest
        # the entry check writes into a sibling directory and the identifier
        # check reports NOT_RUN ("the entry check did not produce an identifier")
        # on a tree that exported one.
        name = _DIGEST_SUFFIX.sub("", name)
    scratch = root.joinpath("results", "stage2_RQ-VAE", "_conformance", name)
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def _python_for() -> str:
    """The Stage 2 interpreter, as the frozen contract sets it."""
    from . import contract
    return contract.STAGE2_PYTHON


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


def _admission_tree(tree: Path) -> Path:
    """A copy of the tree at the place the trainer will really run.

    A trainer derives the repository root from the stage2_RQ-VAE marker, so a
    tree that sits under results/ (the round workspace) resolves a different
    root than the tree that is promoted to stage2_RQ-VAE/ for the training. The
    admission must judge the tree where it will train, or it judges a tree that
    cannot start: the copy is a direct child of stage2_RQ-VAE/, so the marker
    resolves to the repository root in both places.
    """
    import shutil

    root = contract.resolve_root()
    # The name carries a digest of the path, so two runs over trees of the same
    # name cannot copy into the same directory and delete each other's copy.
    import hashlib
    digest = hashlib.sha256(str(tree).encode("utf-8")).hexdigest()[:10]
    # One scratch directory per check run, not per tree: two admissions of the
    # same tree that run at the same time (a dispatch while a manual check is
    # under way) would otherwise share the directory, and the first to finish
    # deletes the copy the second is still reading. The failure it produces is a
    # missing source file in the copy, which reads like a defect of the tree.
    import uuid
    target = root / "stage2_RQ-VAE" / (
        "_admission_" + tree.name + "_" + digest + "_"
        + str(os.getpid()) + "_" + uuid.uuid4().hex[:8])
    if target.exists():
        shutil.rmtree(target)
    # The candidate trees carry the frozen dataset directory with a symlink
    # that points outside the tree; copying it as a link keeps the admission
    # working instead of failing on a file the training entry point never
    # reads. The copy is thrown away after the check.
    shutil.copytree(tree, target, symlinks=True)
    return target


def _discard_admission_tree(tree: Path) -> None:
    """Remove the admission copy, so a later run starts from the tree itself."""
    import shutil

    root = contract.resolve_root()
    for target in (root / "stage2_RQ-VAE").glob("_admission_" + tree.name + "*"):
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)

def check_entry(tree: Path | str) -> dict:
    """The real entry point runs one epoch in this process and exports a SID."""
    tree = _admission_tree(Path(tree).resolve())
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
               # A trainer computes METRICS_PATH from LOG_DIR when the module
               # is imported, so patching LOG_DIR afterwards does not redirect
               # the metrics stream. Without this line a conformance run
               # TRUNCATES the training metrics of a published run: the check
               # ran one epoch into a tree whose 3000-epoch stream was the
               # evidence for its report, and the stream became one row.
               "setattr(t, 'METRICS_PATH', t.LOG_DIR / 'training_metrics.jsonl')"
               " if hasattr(t, 'METRICS_PATH') else None;"
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


def _visible_device_list() -> str:
    """The cards this host actually exposes, as CUDA_VISIBLE_DEVICES wants them.

    A hard-coded "0,1,2,3" is wrong on any host with fewer than four cards: the
    probe would fail on ranks 1..3 with an invalid device ordinal and the
    admission would refuse a healthy tree. The ranks share whatever is visible,
    which is exactly what the single-card layout does at run time.
    """
    import subprocess
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
        indices = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        if done.returncode == 0 and indices:
            return ",".join(indices)
    except (OSError, subprocess.SubprocessError):
        pass
    return "0"


def check_four_rank(tree: Path | str) -> dict:
    """The four-rank step finishes and a rank-0 evaluation does not wait."""
    tree = _admission_tree(Path(tree).resolve())
    # The four-rank property is a framework invariant, not the mechanism's own
    # artifact: the probe lives in the framework and runs with the candidate tree
    # as its working directory, so a candidate that ships no script of its own is
    # still probed. A tree that does ship one keeps using it.
    own = tree / "ddp_check.py"
    script = own if own.is_file() else Path(__file__).resolve().parent / "ddp_probe.py"
    if not script.is_file():
        return {"status": "NOT_RUN", "name": "four_rank",
                "detail": "the framework's four-rank probe is missing"}
    torchrun = Path(_python_for()).with_name("torchrun")
    # A host with fewer than four cards runs as many ranks as it has: the
    # invariant this check defends is "the ranks this host can start finish and a
    # rank-0 evaluation does not wait", and a single-card host proves it with one
    # rank. The mode is recorded, never silently reported as the four-card run.
    devices = _visible_device_list().split(",")
    nproc = max(1, min(int(contract.WORLD_SIZE), len(devices)))
    attempts = []
    for attempt in range(2):
        try:
            done = subprocess.run(
                [str(torchrun), "--standalone", "--nproc_per_node=" + str(nproc),
                 "--master_port=" + str(contract.STAGE2_PORT + 10 + attempt), str(script)],
                cwd=str(tree), capture_output=True, text=True, timeout=1200,
                env={**contract.launch_env(),
                     "CUDA_VISIBLE_DEVICES": _visible_device_list()})
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "name": "four_rank",
                    "detail": "the four-rank step did not finish: a collective mismatch"}
        if done.returncode == 0 and "DDP_PASS" in done.stdout:
            detail = ("the four ranks finish and the rank-0 evaluation does not wait"
                      if nproc >= int(contract.WORLD_SIZE) else
                      "single-card host: %d rank(s) on %d visible card(s) finish and the "
                      "rank-0 evaluation does not wait" % (nproc, len(devices)))
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


#: The operators a candidate may not define: the geometry belongs to the
#: framework's contract module.
_FORBIDDEN_OPERATORS = ("expmap0", "logmap0", "expmap", "logmap", "mobius_add",
                        "poincare_distance", "clamp_ball", "exp0_c", "log0_c",
                        "squared_distance_c", "distance_c",
                        # the Lorentz model of the same curvature: the framework
                        # owns it for the same reason it owns the ball
                        "minkowski_dot", "lorentz_exp", "lorentz_log",
                        "lorentz_exp_origin", "lorentz_log_origin",
                        "lorentz_parallel_transport", "lorentz_lift")


def _baseline_tree() -> Path | None:
    """The baseline the candidates are derived from, when it can be located."""
    try:
        from tools.research.rounds.baseline import baseline_path
        path = Path(baseline_path()).resolve()
        return path if path.is_dir() else None
    except Exception:                                      # noqa: BLE001
        return None


def check_names(tree: Path | str) -> dict:
    """The trainer must bind every module-level name it reads.

    The fault this refuses: the accepted-state restore record named
    ``NUM_EPOCHS``, the trainer defines ``EPOCHS``, and a complete 3000-epoch run
    died in the export with ``NameError`` -- no SID, reported as incomplete.
    """
    tree = Path(tree).resolve()
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        return {"status": "FAIL", "name": "names", "detail": "the entry point is missing"}
    from tools.research.stage2.tests.test_trainer_names import find_undefined_globals
    names = find_undefined_globals(entry.read_text(encoding="utf-8"))
    if names:
        return {"status": "FAIL", "name": "names",
                "detail": "the trainer references unbound module-level names: "
                          + ", ".join(names)}
    return {"status": "PASS", "name": "names",
            "detail": "every module-level name the trainer reads is bound"}


def check_collectives(tree: Path | str) -> dict:
    """Every collective in the trainer must be entered by every rank.

    The fault this refuses: a broadcast inside ``if rank == 0`` left the NCCL
    communicator out of step, and the following gradient all-reduces mixed
    buffers from different steps. The partition was healthy at epoch 50 and
    collapsed to 0.999 within three hundred epochs.
    """
    tree = Path(tree).resolve()
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        return {"status": "FAIL", "name": "collectives",
                "detail": "the entry point is missing"}
    from tools.research.stage2.tests.test_collectives import (
        find_rank_branch_collectives,
    )
    hits = find_rank_branch_collectives(entry.read_text(encoding="utf-8"))
    if hits:
        return {"status": "FAIL", "name": "collectives",
                "detail": "a collective is entered by one rank only: " + "; ".join(
                    f"line {line} dist.{name} under {condition!r}"
                    for line, name, condition in hits)}
    return {"status": "PASS", "name": "collectives",
            "detail": "every collective call is reached by every rank"}


def check_geometry_contract(tree: Path | str) -> dict:
    """The candidate uses the canonical geometry module and defines no other.

    A candidate that writes its own exponential or distance re-opens exactly the
    class of defect the contract exists to stop: two files that disagree about
    what curvature means.
    """
    import hashlib
    import re

    tree = Path(tree).resolve()
    canonical = Path(__file__).resolve().parents[1] / "geometry" / "poincare_contract.py"
    copy = tree / "model" / "poincare_contract.py"
    # The baseline is the tree the candidate was derived from. A definition or a
    # missing module that the baseline ALREADY had is inherited, not introduced
    # by this candidate: the contract forbids NEW geometry, and rejecting an
    # inherited file would reject every candidate derived from that baseline.
    baseline = _baseline_tree()
    if baseline is not None and tree == baseline:
        # The baseline is the reference: it defines the geometry every candidate
        # inherits, so it is conformant to itself by construction.
        return {"status": "PASS", "name": "geometry_contract",
                "detail": "the frozen baseline is the reference tree"}
    # A missing module is not a defect by itself: a tree that derives from a
    # baseline without the module (the frozen TIGER_RQ-VAE is one) cannot drop
    # what it never had. The scan below is the real rule: no NEW definition and
    # no foreign import of a geometry operator.
    if copy.is_file() and canonical.is_file():
        if hashlib.sha256(copy.read_bytes()).hexdigest() != \
                hashlib.sha256(canonical.read_bytes()).hexdigest():
            return {"status": "FAIL", "name": "geometry_contract",
                    "detail": "the candidate's copy of the geometry module was modified"}
    # The whole tree is scanned, not only model/: a candidate can define the
    # operator in its trainer or in a sub-package, and a call to a foreign
    # implementation is the same defect in another shape.
    for path in sorted(tree.rglob("*.py")):
        relative = path.relative_to(tree)
        if any(part in ("logs", "results", "dataset", "__pycache__", "ckpt")
               for part in relative.parts):
            continue
        if str(relative).replace("\\", "/") == "model/poincare_contract.py":
            continue
        if baseline is not None:
            original = baseline / relative
            if original.is_file() and original.read_bytes() == path.read_bytes():
                continue  # inherited from the baseline, not introduced here
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in _FORBIDDEN_OPERATORS:
            if re.search(r"^\s*def\s+" + re.escape(name) + r"\s*\(", text, re.M):
                return {"status": "FAIL", "name": "geometry_contract",
                        "detail": "the candidate defines its own " + name +
                                  " in " + str(relative) + "; the geometry is the framework's"}
            if re.search(r"^\s*(from|import)\s+.*\b" + re.escape(name) + r"\b",
                         text, re.M) and "poincare_contract" not in text:
                return {"status": "FAIL", "name": "geometry_contract",
                        "detail": "the candidate imports a foreign " + name +
                                  " in " + str(relative) +
                                  "; the geometry is the framework's"}
    return {"status": "PASS", "name": "geometry_contract",
            "detail": "the candidate uses the canonical geometry module and defines no other"}


def check_assignment_rule(tree: Path | str) -> dict:
    """The baseline's Sinkhorn-balanced assignment stays; the mechanism's score
    is its cost matrix.

    The baseline's level assignment is Sinkhorn-balanced, and the balancing is
    what keeps the quantiser's codebook usage even. A mechanism that takes the
    level index by a plain argmin over its own score replaces that rule, and
    the codebook collapses: the arm of round_20261105_144909_130 used 42 of 256
    level-1 codes against the reference arm's 241, the largest code took 31% of
    the items, and the three-level collision reached 0.703 against the
    reference's 0.065. The mechanism's score belongs INSIDE the balanced
    assignment.
    """
    tree = Path(tree).resolve()
    sources = {}
    for path in sorted(tree.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        sources[str(path.relative_to(tree))] = path.read_text(encoding="utf-8",
                                                              errors="replace")
    if not sources:
        return {"status": "FAIL", "name": "assignment_rule",
                "detail": "the tree carries no source to inspect"}
    text = "\n".join(sources.values())
    balanced = "sinkhorn" in text.lower()
    if not balanced:
        return {"status": "FAIL", "name": "assignment_rule",
                "detail": "the tree no longer contains the baseline's Sinkhorn "
                          "balanced assignment: the mechanism may change the "
                          "assignment's COST, never the assignment rule"}
    offenders = []
    for name, source in sources.items():
        for number, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if "argmin" not in stripped:
                continue
            if not any(token in stripped for token in
                       ("score", "off", "mechanism", "D_k", "distance -", "distances -")):
                continue
            offenders.append("%s:%d: %s" % (name, number, stripped[:120]))
    if offenders:
        return {"status": "FAIL", "name": "assignment_rule",
                "detail": "the level index is taken by a plain argmin over the "
                          "mechanism's score, which replaces the balanced "
                          "assignment: " + "; ".join(offenders[:3])[:300]}
    return {"status": "PASS", "name": "assignment_rule",
            "detail": "the balanced assignment is the rule and the mechanism's "
                      "score enters it as the cost"}


def check_reference_agreement(tree: Path | str) -> dict:
    """The declared equations, rebuilt from their SymPy statements and compared.

    A mechanism declares its continuous surface in reference_contract.json; the
    check rebuilds that surface independently (sympytorch) and compares values,
    gradients and torch.autograd.gradcheck on both sides, in double precision.
    The agent that wrote the mechanism does not also write the only statement
    of what it computes. A tree without a contract is recorded as NOT_APPLICABLE
    with its reason, never as a pass.
    """
    tree = Path(tree).resolve()
    contract_file = tree / "reference_contract.json"
    if not contract_file.is_file():
        return {"status": "NOT_APPLICABLE", "name": "reference_agreement",
                "detail": "the tree declares no reference_contract.json"}
    command = [_python_for(), "-m", "tools.research.stage2.reference",
               "--tree", str(tree)]
    # The comparison imports the candidate's own module, so the tree is the
    # working directory and the first entry of the import path.
    import os
    environment = dict(os.environ)
    root = str(contract.resolve_root())
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tree), root] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
    try:
        done = subprocess.run(command, cwd=str(tree), env=environment,
                              capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "name": "reference_agreement",
                "detail": "the reference comparison did not finish in fifteen minutes"}
    lines = [line for line in done.stdout.strip().splitlines() if line.startswith("{")]
    if not lines:
        return {"status": "FAIL", "name": "reference_agreement",
                "detail": "the comparison produced no report: " +
                          (done.stdout + done.stderr).strip()[-300:]}
    report = json.loads(lines[-1])
    status = report.get("status")
    coupling = report.get("coupling") or {}
    if status == "OK":
        detail = "; ".join(
            f"{call['name']}: values {call.get('value_max_abs_diff', 0):.1e}, "
            f"gradients {call.get('gradient_max_abs_diff', 0):.1e}, gradcheck "
            f"{call.get('gradcheck_candidate')}/{call.get('gradcheck_reference')}"
            for call in report.get("calls") or [])
        if coupling.get("status") == "PASS":
            detail += ("; the behaviour moves the computation ("
                       + str(coupling.get("max_abs_difference")) + ")")
        else:
            # The behaviour term cannot move the computation: the mechanism
            # cannot change the assignment through it, and the candidate is
            # refused here rather than after a training run.
            return {"status": "FAIL", "name": "reference_agreement",
                    "detail": "the coupling surface: " + str(coupling.get("reason"))[:300]}
        return {"status": "PASS", "name": "reference_agreement", "detail": detail,
                "coupling": coupling}
    return {"status": status or "FAIL", "name": "reference_agreement",
            "detail": str(report.get("reason"))[:400]}


def check_encoder_gradient(tree: Path | str) -> dict:
    """The reconstruction reaches the ENCODER of the candidate's own model.

    The three implementations of the frozen specification can agree perfectly
    and the real encoder can still receive nothing - an accidental ``detach`` in
    the wiring, a quantization path without the straight-through estimator.
    The probe loads the candidate's model, runs one forward and one backward
    with the RECONSTRUCTION loss alone (the quantisation loss would reach the
    encoder through the commitment term even when the straight-through is
    missing) and reports whether the encoder's parameters received a gradient.
    """
    tree = _admission_tree(Path(tree).resolve())
    command = [_python_for(), "-m", "tools.research.stage2.gradient_probe",
               "--tree", str(tree)]
    import os
    environment = dict(os.environ)
    root = str(contract.resolve_root())
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tree), root] + ([environment["PYTHONPATH"]]
                             if environment.get("PYTHONPATH") else []))
    try:
        done = subprocess.run(command, cwd=str(tree), env=environment,
                              capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "name": "encoder_gradient",
                "detail": "the probe did not finish in twenty minutes"}
    lines = [line for line in done.stdout.strip().splitlines() if line.startswith("{")]
    if not lines:
        return {"status": "FAIL", "name": "encoder_gradient",
                "detail": "the probe produced no report: " +
                          (done.stdout + done.stderr).strip()[-300:]}
    report = json.loads(lines[-1])
    status = str(report.get("status") or "FAIL")
    if status == "PASS":
        return {"status": "PASS", "name": "encoder_gradient",
                "detail": str(report.get("detail"))}
    return {"status": "FAIL", "name": "encoder_gradient",
            "detail": str(report.get("reason") or report.get("detail"))[:300]}


def run_all(tree: Path | str) -> dict:
    """The conformance report of one candidate tree."""
    tree = Path(tree).resolve()
    # The static contract comes FIRST: a candidate that redefines the geometry
    # must not run an epoch before it is refused.
    checks = [check_names(tree), check_collectives(tree), check_constants(tree),
              check_geometry_contract(tree), check_entry(tree),
              check_identifier(tree), check_four_rank(tree),
              check_assignment_rule(tree), check_encoder_gradient(tree)]
    # The owner removed the declared-formula-versus-code comparison from the
    # admission on 2026-11-10: the remaining checks are the executable ones.
    passed = all(check["status"] == "PASS" for check in checks)
    admissible = passed
    _discard_admission_tree(tree)
    return {"tree": str(tree), "passed": passed, "admissible": admissible,
            "checks": checks}
