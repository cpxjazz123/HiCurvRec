"""The shared call into an agent: its workspace, its boundary, the
runner and the concurrent driver.

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




#: One idea per agent. A round takes five agents, so it collects five proposals
#: (the contract wants three to nine), and one idea block fits in the five
#: minutes an agent has.
IDEA_MIN = 1
IDEA_MAX = 1
SCORERS = 3

#: The phases the worker can run, in the order the flow runs them.
PHASES = ("ideation", "probe", "scoring", "implementation")


def _repository_root(start: Path) -> Path:
    """The repository root above a round directory, found by the marker."""
    here = Path(start).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() and (parent / "stage2_RQ-VAE").is_dir():
            return parent
    return here

def _workspace(root: Path, task_id: str) -> Path:
    return Path(root) / "agents" / task_id


class RoundIncomplete(RuntimeError):
    """A round phase is missing agents; the work that exists is kept.

    The main agent decides what to do next: retry the missing agents when the
    class is an execution fault, pause when the class is an authentication or a
    quota fault, and never read the missing answers as a negative verdict on the
    mechanism.
    """

    def __init__(self, message: str, *, results: list, failures: list, classes: set):
        super().__init__(message)
        self.results = results
        self.failures = failures
        self.classes = classes
        self.waiting = bool(classes & {"auth", "quota"})


class AgentPhaseIncomplete(RuntimeError):
    """Some agents of a phase produced no answer; the others did.

    ``results`` holds the answers that exist, so a phase can persist them and ask
    only for the missing ones next time instead of paying for all of them again.
    ``classes`` says why the others failed: an ``auth`` or ``quota`` class means
    the service, not the candidate, is the problem.
    """

    def __init__(self, summary: str, *, results: list, failures: list, classes: set,
                 delivered: list = None):
        super().__init__("the phase is incomplete: " + summary)
        self.results = results
        self.failures = failures
        #: the answers that exist, each with the task it belongs to
        self.delivered = list(delivered or [])
        self.classes = classes


#: The framework's mathematics. Every research agent reads the contract, and the
#: roles that inspect implementations also read the module that implements it.
GEOMETRY_CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "HYPERBOLIC_GEOMETRY_CONTRACT.md"
GEOMETRY_MODULE = Path(__file__).resolve().parents[1] / "geometry" / "poincare_contract.py"


def geometry_contract() -> dict:
    """The shared mathematical input: its path, version and digest."""
    import hashlib
    text = GEOMETRY_CONTRACT.read_text(encoding="utf-8")
    version = "poincare-contract/v1"
    for line in text.splitlines():
        if line.startswith("Status:") and "`" in line:
            version = line.split("`")[1]
            break
    return {"path": str(GEOMETRY_CONTRACT),
            "sha256": hashlib.sha256(GEOMETRY_CONTRACT.read_bytes()).hexdigest(),
            "version": version,
            "module": str(GEOMETRY_MODULE)}


def _round_roots(root: Path) -> list:
    """The directories a task of this round may read from.

    The round's own directory and the round's trees. The trees live beside the
    rounds, under the issue: ``results/research/<issue>/trees/<round_id>``, and
    the candidate tree of an idea is one of them. Anything else - another
    round, the arms' results, the Stage 3 outputs, the earlier sessions - is
    outside the boundary, and the policy refuses it.
    """
    root = Path(root).resolve()
    issue_dir = root.parent.parent
    return [root, issue_dir / "trees" / root.name]


def role_of_task(task_id: str) -> str:
    """The role a task id names.

    The cached-answer path records the role of the task that produced the
    answer, so a reused answer says which role wrote it. The helper was missing
    and the path raised a NameError the moment an answer was reused - which is
    exactly when a round is interrupted and restarted.
    """
    prefix = str(task_id).rsplit("_", 1)[0]
    return {"ideation": "ideate", "scoring": "score", "derivation": "derive",
            "formulation": "formulate", "math_review": "review_math",
            "coding": "code", "probe": "probe"}.get(prefix, "verify")


def _run_agent(root: Path, task_id: str, prompt: str, output_name: str, parser,
               role: str = "verify", read_paths=None, sandbox: bool = True,
               allowed_roots=None, candidate_tree=None,
               allow_shell: bool = False, allow_write: bool = False,
               deadline_seconds: int | None = None,
               source_digest: str = "") -> dict:
    """Run one agent in its own workspace and parse its plain-text answer.

    The agent writes a labelled text file, not JSON: a language model that
    drops one comma must not lose the whole answer. The parser turns the text
    into the record the round keeps.
    """
    # The runner gives the agent a workspace but not a destination; without this
    # the agent answers in prose and writes no file.
    destination = Path(root) / "agents" / task_id / output_name
    workspace = Path(root) / "agents" / task_id
    if str(role) in READ_ONLY_ROLES:
        # A read-only role has no write tool, and the permission object now
        # enforces that. The answer is its reply: the runner reads the session
        # stream and writes the file itself.
        prompt = (prompt + "\n\nYour session has no write tool and no shell: your reply IS "
                  "the answer. Print the complete answer in exactly the format above, "
                  "and print nothing else.")
    else:
        prompt = (prompt + "\n\nWrite the answer to " + str(destination) +
                  " (this absolute path). Write nowhere else.")
    prompt = (prompt + "\n\nPROGRESS: record every meaningful step with the "
              "append_progress tool. It is your only write: it appends one stage "
              "record to your own progress file, and nothing else can write there "
              "or read your file back as a verdict. Each record says the goal of "
              "the check, what you read, what you observed, where the evidence "
              "is, your short reasoning, and what is still open. Append a "
              "correction when you overturn an earlier conclusion. Append each "
              "result you produce as its own record as soon as you have it (one "
              "answer block per record), and append the complete answer as the "
              "last record: a task that ends early leaves the framework reading "
              "these records, so each one has to stand on its own.")
    # The boundary this task would run under, computed from the FINAL prompt:
    # the fingerprint has to be the one the executor will write, or no answer is
    # ever reusable and every restart pays for every agent again.
    # The identity of the run: the boundary, the task and the model. A role that
    # reads a tree also carries the digest of that tree, so an answer about an
    # older tree is never reused for a newer one.
    expected_boundary = boundary_fingerprint(
        workspace, role, runner.effective_read_paths(read_paths),
        allow_shell, allow_write, prompt, runner.effective_model(),
        source_digest=source_digest)
    # Resumability: an answer that is already on disk is reused, so a round that
    # was interrupted does not pay for the same agents twice. The reuse requires
    # the receipt to say the run was valid: a partial answer left by a timed-out
    # or cancelled run is not an answer, and reading it as one is how a cancelled
    # reviewer became a verdict. The receipt also has to name the same boundary,
    # the same prompt and the same model: a task that is asked again with a
    # different question is not answered by the old answer.
    receipt_path = workspace / "execution.json"
    if destination.is_file():
        verified = False
        boundary = None
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text())
                verified = bool(receipt.get("verified"))
                boundary = receipt.get("boundary_sha256")
            except (OSError, ValueError):
                verified = False
        if verified and boundary == expected_boundary:
            report = parser(destination.read_text(encoding="utf-8", errors="replace"))
            report["boundary_version"] = policy.BOUNDARY_VERSION
            report["boundary_sha256"] = boundary
            report["progress_file"] = str(workspace / runner.PROGRESS_FILE_NAME)
            report["progress_records"] = progress_server.read_records(
                report["progress_file"])
            report["progress_count"] = len(report["progress_records"])
            report["cached"] = True
            report.setdefault("role", role_of_task(task_id))
            return report
        # Keep the unverified answer as evidence and let the agent run again.
        stale = workspace / "unverified"
        stale.mkdir(parents=True, exist_ok=True)
        destination.replace(stale / (output_name + ".unverified"))
        if receipt_path.is_file():
            receipt_path.replace(stale / "execution.json.unverified")
    if deadline_seconds is None:
        deadline_seconds = settings.AGENT_DEADLINE_SECONDS
    if allowed_roots is None:
        allowed_roots = _round_roots(Path(root))
    progress_file = workspace / runner.PROGRESS_FILE_NAME
    partial = False
    failure_class = ""
    try:
        receipt = run_agent(task_id, prompt, output_name, Path(root),
                            model=runner.effective_model(),
                            role=role, read_paths=read_paths,
                            allowed_roots=allowed_roots, candidate_tree=candidate_tree,
                            allow_shell=allow_shell, allow_write=allow_write,
                            deadline_seconds=deadline_seconds,
                            source_digest=source_digest)
    except runner.AgentFailure as failure:
        # The task did not finish inside its budget. What it saved is what it
        # delivered: the records are handed to the main agent, marked as a
        # partial delivery, and never as a completed answer.
        partial = True
        failure_class = failure.failure_class
        receipt = Path(failure.receipt)
    # run_agent returns its execution receipt, not the answer: locate the file
    # the agent wrote, accepting the same bounded relocation the runner allows.
    answer = workspace / output_name
    if not answer.is_file():
        matches = [match for match in sorted(workspace.rglob(output_name))
                   if match.is_file()
                   and not any(part.startswith("attempt-")
                               for part in match.relative_to(workspace).parts)]
        if matches:
            answer = matches[0]
    records = progress_server.read_records(progress_file)
    text = ""
    answer_from_progress = False
    if answer.is_file():
        text = answer.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        # The task ended before it replied. The records it saved are what it
        # delivered: the last one that carries a complete answer is taken. A
        # scorer that delivered one block per record is assembled; a verdict is
        # not, because a stage record is never the final judgement.
        text = runner.saved_answer(workspace, parser,
                                   assemble=(parser is plaintext.parse_ranks))
        answer_from_progress = bool(text)
    if not text:
        if partial:
            raise pipeline_module.RoundError(
                "agent " + task_id + " did not finish (" + failure_class +
                ") and saved no complete answer; its " + str(len(records)) +
                " stage records are kept as evidence")
        raise pipeline_module.RoundError("agent " + task_id + " wrote no " + output_name)
    report = parser(text)
    if isinstance(report, dict) and isinstance(report.get("scores"), list):
        # The records of several attempts are joined in order, so an idea can
        # appear more than once; the latest judgement is the one that stands.
        latest = {}
        for entry in report["scores"]:
            latest[str(entry.get("idea_id"))] = entry
        report["scores"] = list(latest.values())
    report["boundary_version"] = policy.BOUNDARY_VERSION
    report["boundary_sha256"] = expected_boundary
    report["receipt"] = str(receipt)
    # The stage records the agent delivered before the deadline: the main agent
    # judges from what was actually saved, not from what the model meant.
    report["progress_file"] = str(progress_file)
    report["progress_records"] = records
    report["progress_count"] = len(records)
    report["answer_from_progress"] = answer_from_progress
    report["deadline_seconds"] = deadline_seconds
    # A delivery that did not finish says so. The main agent may use a partial
    # proposal; it may never read a partial review as a verdict.
    report["partial"] = partial
    report["failure_class"] = failure_class
    return report


def _run_agents_concurrently(tasks: list, worker, *, max_workers: int = 0) -> list:
    """Run the agents of one phase concurrently.

    Every agent of a phase has its own workspace, its own permission file and
    its own subprocess, so the only shared thing is the machine. Running them
    one after another made a review round cost the sum of five agents; running
    them together costs the slowest one.

    A failure is returned, not raised: one agent that crashed must not throw
    away the answers of the others, and the phase decides whether it has enough
    to continue. The failures carry the class of the fault, so an expired
    credential pauses the service instead of being retried five times.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not tasks:
        return []
    workers = max_workers or len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(task, pool.submit(worker, *task)) for task in tasks]
        results: list = []
        failures: list = []
        for task, future in futures:
            try:
                results.append(future.result())
            except Exception as exc:                   # noqa: BLE001
                failures.append({"task": str(task[0]) if task else "",
                                 "error": str(exc)[:400],
                                 "failure_class": getattr(exc, "failure_class", "unknown")})
    if failures:
        classes = {entry["failure_class"] for entry in failures}
        summary = "; ".join(str(entry["task"]) + " (" + entry["failure_class"] + ")"
                            for entry in failures)
        # The results keep the task they belong to: a caller that compares two
        # answers must not attribute the second answer to the first task when
        # the first one failed.
        delivered = [{"task": str(task[0]) if task else "",
                      "result": result}
                     for (task, _future), result in zip(futures, results)]
        raise AgentPhaseIncomplete(summary, results=results, failures=failures,
                                   classes=classes, delivered=delivered)
    return results
