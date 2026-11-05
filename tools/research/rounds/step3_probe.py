"""Step 3 - the pre-score evidence probe. No language model runs
here: the probe measures the training-only behaviour behind each
proposal.

Moved out of worker.py by the split; the bodies are unchanged.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from .. import settings
from ..agents.runner import (READ_ONLY_ROLES, boundary_fingerprint, run_agent)
from ..agents import policy, progress_server, runner
from ..core.evidence import atomic_json, digest_file, read_json
from . import agents as agent_contracts
from . import baseline as baseline_module
from . import evals as evals_module
from . import evidence as evidence_module
from . import ideas as ideas_module
from . import literature as literature_module
from . import math_checks as math_checks_module
from . import module_differential
from . import pipeline as pipeline_module
from . import plaintext
from . import probes as probes_module
from . import scoring as scoring_module
from . import spec_tests as spec_tests_module
from . import specification as specification_module


from .agent_calls import (  # noqa: F401  (the shared call)
    _run_agent, _run_agents_concurrently, _round_roots,
    _workspace, geometry_contract, role_of_task,
)



def _run_probe_subprocess(idea: dict, *, repo: Path, budget: float,
                          max_users: int, seed: int) -> dict:
    """The data probe, in the project's interpreter and inside its budget.

    The MCP server's interpreter carries neither pandas nor torch, so the probe
    runs as a bounded subprocess of the research interpreter. The budget is
    enforced twice: the probe measures its own time and the caller kills the
    process when it overruns.
    """
    import subprocess
    import tempfile

    from .. import settings

    python = getattr(settings, "RESEARCH_PYTHON", sys.executable)
    argv = [python, "-m", "tools.research.rounds.probes",
            "--idea", json.dumps(idea, ensure_ascii=False),
            "--repo", str(repo), "--budget", str(budget),
            "--max-users", str(max_users), "--seed", str(seed)]
    # The output goes to files, not to pipes: a pipe is inherited by the
    # probe's own children, and a child that outlives the probe keeps the pipe
    # open, so the wait for the probe would never end.
    with tempfile.TemporaryDirectory() as work:
        out_path = Path(work) / "out.txt"
        err_path = Path(work) / "err.txt"
        with out_path.open("w", encoding="utf-8") as out, \
                err_path.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(argv, cwd=str(repo), stdout=out, stderr=err,
                                       text=True, start_new_session=True)
            try:
                process.wait(timeout=budget + 30.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {"status": "UNKNOWN", "idea_id": idea.get("idea_id"),
                        "reason": "the probe overran its budget and was stopped",
                        "metrics": {}}
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")
    if process.returncode != 0:
        return {"status": "UNKNOWN", "idea_id": idea.get("idea_id"),
                "reason": "the probe process failed: " + stderr[-300:],
                "metrics": {}}
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "UNKNOWN", "idea_id": idea.get("idea_id"),
                "reason": "the probe produced no readable report", "metrics": {}}


#: The math check imports torch in the project's interpreter, and a cold import
#: on this filesystem costs about seventy seconds on its own: a two-minute
#: budget spends more than half of it before the check starts.
MATH_BUDGET_SECONDS = 240.0


def _run_math_batch(ideas: list, *, seed: int, budget: float = MATH_BUDGET_SECONDS) -> dict:
    """The math checks of every proposal behind ONE torch import.

    A cold torch import costs around seventy seconds on this filesystem, so one
    process per proposal spent the whole budget before the checks started. The
    batch pays that cost once and keeps the per-proposal reports comparable.
    """
    import subprocess
    import tempfile

    from .. import settings

    python = getattr(settings, "RESEARCH_PYTHON", sys.executable)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.write(json.dumps(list(ideas)))
    handle.close()
    argv = [python, "-m", "tools.research.rounds.math_checks",
            "--ideas", handle.name, "--seed", str(seed)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=float(budget))
    except subprocess.TimeoutExpired:
        return {}
    lines = [line for line in done.stdout.strip().splitlines() if line.startswith("{")]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {}


def _run_math_subprocess(idea: dict, *, seed: int,
                         budget: float = MATH_BUDGET_SECONDS) -> dict:
    """The math checks, in the project's interpreter (the only one with torch)."""
    import subprocess
    import tempfile

    from .. import settings

    python = getattr(settings, "RESEARCH_PYTHON", sys.executable)
    argv = [python, "-m", "tools.research.rounds.math_checks",
            "--idea", json.dumps(idea, ensure_ascii=False), "--seed", str(seed)]
    with tempfile.TemporaryDirectory() as work:
        out_path = Path(work) / "out.txt"
        err_path = Path(work) / "err.txt"
        with out_path.open("w", encoding="utf-8") as out, \
                err_path.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(argv, cwd=str(Path.cwd()), stdout=out, stderr=err,
                                       text=True, start_new_session=True)
            try:
                process.wait(timeout=float(budget))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {"status": "NOT_RUN", "idea_id": idea.get("idea_id"), "checks": [],
                        "reason": "the checks overran their budget"}
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")
    if process.returncode != 0:
        return {"status": "NOT_RUN", "idea_id": idea.get("idea_id"), "checks": [],
                "reason": "the checks could not run: " + stderr[-300:]}
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "NOT_RUN", "idea_id": idea.get("idea_id"), "checks": [],
                "reason": "the checks produced no readable report"}


def run_probes(root: Path, ideas: list, *, repo: Path, cache_dir: Path = None,
               deadline: float | None = None) -> dict:
    """The pre-score evidence probe of every proposal.

    Three bounded measurements per idea: the train-only behaviour statistics of
    its hypothesis, the sanity of its curved operation on small tensors, and the
    related work. Nothing here trains, edits a tree or touches validation or
    test data.
    """
    repo = Path(repo)
    cache_dir = Path(cache_dir) if cache_dir is not None else (repo / "results" /
                                                            "research" / "issue121" /
                                                            "literature_cache")
    from concurrent.futures import ThreadPoolExecutor

    # Every proposal's math checks in one process: the import is the cost, not
    # the check. The batch runs CONCURRENTLY with the per-idea measurements:
    # awaited first it spent the whole phase budget before the data probes even
    # started, and the phase then reported every measurement as spent.
    budget_pool = ThreadPoolExecutor(max_workers=min(len(ideas), 6) + 1)
    remaining = None if deadline is None else deadline - time.monotonic()
    math_future = None
    if remaining is None or remaining > 0:
        math_future = budget_pool.submit(
            _run_math_batch, ideas, seed=math_checks_module.SEED,
            budget=(min(MATH_BUDGET_SECONDS, remaining)
                    if remaining is not None else MATH_BUDGET_SECONDS))

    def measure(idea: dict) -> tuple:
        """The three measurements of one proposal, at the same time.

        The data probe and the math checks are separate processes and the
        literature pass is the network: running them one after another made the
        phase cost the sum of their budgets for every proposal.
        """
        identifier = idea["idea_id"]

        def _remaining() -> float | None:
            if deadline is None:
                return None
            return deadline - time.monotonic()

        def _spent() -> dict:
            return {"status": "NOT_RUN", "idea_id": identifier, "checks": [],
                    "reason": "the phase budget was spent before this measurement ran"}

        def data():
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                return _spent()
            report = _run_probe_subprocess(
                idea, repo=repo,
                budget=(min(probes_module.DEFAULT_BUDGET_SECONDS, remaining)
                        if remaining is not None else probes_module.DEFAULT_BUDGET_SECONDS),
                max_users=probes_module.DEFAULT_MAX_USERS,
                seed=probes_module.DEFAULT_SEED)
            report["sha256"] = _probe_digest(report)
            return report

        def math():
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                return _spent()
            reports = {}
            if math_future is not None:
                try:
                    reports = math_future.result() or {}
                except Exception:  # noqa: BLE001
                    reports = {}
            report = reports.get(identifier)
            if not report:
                report = {"status": "NOT_RUN", "idea_id": identifier, "checks": [],
                          "reason": "the batched math checks produced no report for it"}
            report = dict(report)
            report["sha256"] = _probe_digest(report)
            return report

        def literature():
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                return _spent()
            report = literature_module.run_literature(
                idea, cache_dir=cache_dir,
                budget_seconds=(min(60.0, remaining) if remaining is not None else 60.0))
            report["sha256"] = _probe_digest(report)
            return report

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(data), pool.submit(math), pool.submit(literature)]
            probe, math_report, lit = [future.result() for future in futures]
        return identifier, {"probe": probe, "literature": lit, "math": math_report}

    # The proposals are independent, so they are measured at the same time, and
    # the math batch is one more worker in the same pool.
    outcomes = {}
    try:
        for identifier, outcome in budget_pool.map(measure, ideas):
            outcomes[identifier] = outcome
    finally:
        budget_pool.shutdown(wait=False)
    return outcomes


def _probe_digest(payload: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def _write_probe_artifacts(directory: Path, identifier: str, outcome: dict,
                           bundle: dict) -> None:
    """Store the probe, the literature, the math checks and the bundle."""
    from ..core.evidence import atomic_json

    target = Path(directory) / "probes" / identifier
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "probe.json", outcome.get("probe") or {})
    atomic_json(target / "literature.json", outcome.get("literature") or {})
    atomic_json(target / "math_checks.json", outcome.get("math") or {})
    atomic_json(target / "evidence_bundle.json", bundle or {})


def _probe_gaps(outcome: dict) -> list:
    """What the probe CONTRADICTED, as the proposal author must answer it.

    A status that means "the probe did not measure this" is not a gap in the
    proposal: the probe module states that such a status is neither a
    measurement nor evidence against the idea. Only a measured FAIL and an
    inconclusive measurement are answered with a revision, because those are the
    cases where the author can change something.
    """
    gaps = []
    probe = outcome.get("probe") or {}
    for name, value in (probe.get("metrics") or {}).items():
        if isinstance(value, dict) and value.get("status") in ("FAIL", "INCONCLUSIVE"):
            gaps.append(name + ": " + str(value.get("reason") or value.get("status")))
    math_report = outcome.get("math") or {}
    for check in math_report.get("checks", []):
        if check.get("status") in ("FAIL", "INCONCLUSIVE"):
            gaps.append("math " + str(check.get("check")) + ": " + str(check.get("detail"))[:160])
    literature = outcome.get("literature") or {}
    if literature.get("status") != "OK":
        gaps.append("literature: " + str(literature.get("status")))
    return gaps


def phase_probe(root: Path, round_id: str, *, log=print) -> dict:
    """Run the evidence probe of every proposal and freeze the bundles."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    if state["step"] not in ("probe", "scoring"):
        return {"phase": "probe", "ran": False, "step": state["step"],
                "reason": "the round is not in the probe phase"}
    if state.get("evidence_bundles") and state.get("probe_results"):
        # The measurement is done: the bundles are frozen and the results are
        # recorded. Re-probing would replace evidence the scorers have already
        # seen, and it would move the round back a step for no reason.
        if state["step"] != "scoring":
            state["step"] = "scoring"
            pipeline_module.save_state(directory, state)
        return {"phase": "probe", "ran": False, "step": state["step"],
                "reason": "the evidence bundles are already frozen"}
    ideas = state["ideas"]
    state["step"] = "probe"
    pipeline_module.save_state(directory, state)
    outcomes = run_probes(directory, ideas, repo=Path(root),
                          deadline=time.monotonic() + settings.AGENT_TIMEOUT_SECONDS)
    bundles = {}
    for idea in ideas:
        identifier = idea["idea_id"]
        outcome = outcomes.get(identifier, {})
        base = evidence_module.build_bundle(idea, root=Path(root))
        bundles[identifier] = evidence_module.merge_probe_evidence(
            base, probe=outcome.get("probe"), literature=outcome.get("literature"),
            math_report=outcome.get("math"))
    state["probe_results"] = {identifier: {
        "probe": outcome.get("probe"), "literature": outcome.get("literature"),
        "math": outcome.get("math")}
        for identifier, outcome in outcomes.items()}
    state["evidence_bundles"] = bundles
    for identifier, bundle in bundles.items():
        _write_probe_artifacts(directory, identifier, outcomes.get(identifier, {}), bundle)
    # The probe does NOT revise the proposals. It measures them, and the
    # evidence is handed to the scorers unchanged. A revision here would let a
    # proposal be rewritten until it satisfies the measurement, and the
    # rewritten proposal would be a different research mechanism with its own
    # novelty, its own evidence and its own hypothesis. A proposal whose claim
    # is contradicted by a measurement keeps the counterexample and goes to the
    # scorers; the ideation agents of the NEXT round may propose a new version,
    # which is scored as a new proposal.
    state["evidence_bundles"] = bundles
    state["probe_results"] = {identifier: {
        "probe": outcome.get("probe"), "literature": outcome.get("literature"),
        "math": outcome.get("math")}
        for identifier, outcome in outcomes.items()}
    state["evidence_revisions"] = []
    state["step"] = "scoring"
    pipeline_module.record_event(state, "probe_complete", ideas=len(ideas))
    pipeline_module.save_state(directory, state)
    log("probe: %d bundles frozen" % len(bundles))
    return {"phase": "probe", "ran": True, "step": state["step"],
            "ideas": len(ideas),
            "statuses": {identifier: (outcome.get("probe") or {}).get("status")
                         for identifier, outcome in outcomes.items()},
            "literature": {identifier: (outcome.get("literature") or {}).get("status")
                           for identifier, outcome in outcomes.items()},
            "math": {identifier: (outcome.get("math") or {}).get("status")
                     for identifier, outcome in outcomes.items()}}


def phase_probe_step(root: Path, round_id: str, *, log=print) -> dict:
    """The pre-score probe: measure, then freeze the evidence bundle."""
    return phase_probe(root, round_id, log=log)
