"""The round worker: the agent half of an Issue #120 round.

It runs the five ideation agents, the probe and the three scoring agents. There
is no implementation agent and no review agent: the main agent writes each
selected mechanism and registers it, and verification is the framework's MACHINE
admission (``tools/research/stage2/checks.py``) that runs at launch. The worker
never trains anything: the Stage 2 and Stage 3 jobs are executed by the MCP,
which owns the GPU boundary.
"""
from __future__ import annotations

import itertools
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path

from .. import settings
from ..agents.runner import (READ_ONLY_ROLES, boundary_fingerprint, run_agent)
from ..agents import policy, progress_server, runner
from ..core.evidence import atomic_json, digest_file, read_json
from . import agents as agent_contracts
from . import baseline as baseline_module
from . import evidence as evidence_module
from . import ideas as ideas_module
from . import literature as literature_module
from . import math_checks as math_checks_module
from . import probes as probes_module
from . import plaintext
from . import scoring as scoring_module
from . import pipeline as pipeline_module

#: One idea per agent. A round takes five agents, so it collects five
#: proposals (the contract wants three to nine), and one idea block fits in the
#: five minutes an agent has: a proposal that takes longer than the deadline is
#: a proposal that never arrives.
IDEA_MIN = 1
IDEA_MAX = 1
SCORERS = 3


def _workspace(root: Path, task_id: str) -> Path:
    return Path(root) / "agents" / task_id


class RoundIncomplete(RoundError if "RoundError" in globals() else RuntimeError):
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

    def __init__(self, summary: str, *, results: list, failures: list, classes: set):
        super().__init__("the phase is incomplete: " + summary)
        self.results = results
        self.failures = failures
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
        raise AgentPhaseIncomplete(summary, results=results, failures=failures,
                                   classes=classes)
    return results


def round_history(repo: Path, limit: int = 3) -> str:
    """What the earlier rounds measured, for the ideation agents to read.

    The proposals of a round are not revised inside the round. What the earlier
    rounds measured, what their reviews objected to and what their runs scored is
    therefore the material the next round starts from.
    """
    rounds_root = _repository_root(Path(repo)) / "results" / "research"
    if not rounds_root.is_dir():
        return ""
    blocks: list = []
    for issue in sorted(rounds_root.iterdir()):
        if not issue.is_dir():
            continue
        for directory in sorted((issue / "rounds").glob("round_*")):
            state_path = directory / "state.json"
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                continue
            lines = [directory.name]
            selected = {entry.get("idea_id")
                        for entry in (state.get("selection") or {}).get("selected", [])}
            for idea in state.get("ideas") or []:
                if idea.get("idea_id") in selected:
                    lines.append("  selected: " + str(idea.get("idea_id")) + " | " +
                                 str(idea.get("title"))[:90])
            for identifier, outcome in (state.get("probe_results") or {}).items():
                statuses = []
                probe = (outcome.get("probe") or {}).get("status")
                math_report = (outcome.get("math") or {}).get("status")
                literature = (outcome.get("literature") or {}).get("status")
                if probe and probe != "OK":
                    statuses.append("probe=" + str(probe))
                if math_report and math_report != "OK":
                    statuses.append("math=" + str(math_report))
                if literature and literature != "OK":
                    statuses.append("literature=" + str(literature))
                if statuses:
                    lines.append("  probe of " + identifier + ": " + ", ".join(statuses))
            for job in (state.get("stage2_jobs") or []):
                lines.append("  run: " + str(job.get("idea_id")) + " -> " +
                             str(job.get("status")) + " " + str(job.get("error"))[:80])
            for job in (state.get("stage3_jobs") or []):
                result = job.get("result") or {}
                if result:
                    lines.append("  stage3 of " + str(job.get("idea_id")) + ": recall@10=" +
                                 str(result.get("test_recall@10")))
            if len(lines) > 1:
                blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return "\n".join(blocks[-limit:])


def ideate_one(root: Path, objective: str, extra: bool, index: int) -> tuple:
    """One ideation agent: its own task id, prompt, workspace and answer file.

    The function is the unit the round runs concurrently and the unit the
    flow dispatches it concurrently; every caller uses it, so the workflows
    cannot drift apart.
    """
    prompt_base = agent_contracts.IDEA_PROMPT.format(
        minimum=IDEA_MIN, maximum=IDEA_MAX,
        agents=pipeline_module.IDEA_AGENTS,
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY)
    task_id = "ideation_" + str(index)
    prompt = (prompt_base + "\n\nThe round objective: " + (objective or "improve the "
                 "item semantic IDs for a generative recommender") +
                 "\nYour answer must be a different set of ideas from the other " +
                 str(pipeline_module.IDEA_AGENTS - 1) +
                 " agents, which you cannot see.")
    history = round_history(Path(root))
    if history:
        prompt += ("\n\nWHAT THE EARLIER ROUNDS MEASURED, AND WHAT THEY FAILED ON:\n"
                   + history +
                   "\nUse it to aim at something the earlier rounds did not measure. "
                   "Do not restate a proposal that was already made and rejected.")
    if extra:
        prompt += ("\n\nThe first set of ideas you gave has been collected. Give one "
                   "MORE idea, and it must be different from the one you gave before "
                   "and from what the other agents propose.")
    return (task_id, _run_agent(root, task_id, prompt,
                                "ideas_extra.txt" if extra else "ideas.txt",
                                plaintext.parse_ideas,
                                role="ideate", read_paths=[],
                                allowed_roots=_round_roots(root)))


def run_ideation(root: Path, objective: str, extra: bool = False) -> list:
    """Five independent ideation agents; each returns one to three ideas.

    With ``extra`` the agents are asked for a second, different set, written to
    another file. That pass exists because agents that agree produce one
    proposal between them, and a round needs three distinct ones to be
    compared; agreement is a result, not a reason to end the round.
    """
    prompt_base = agent_contracts.IDEA_PROMPT.format(
        minimum=IDEA_MIN, maximum=IDEA_MAX,
        agents=pipeline_module.IDEA_AGENTS,
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY)
    def ideate(index):
        return ideate_one(root, objective, extra, index)

    try:
        return _run_agents_concurrently(
            [tuple([index]) for index in range(1, pipeline_module.IDEA_AGENTS + 1)], ideate)
    except AgentPhaseIncomplete as exc:
        # An agent that ran out of its five minutes has proposed nothing. The
        # proposals that did arrive are kept: a round needs three distinct
        # proposals to compare, and a missing proposer is not a reason to throw
        # the others away.
        if not exc.results:
            raise
        return exc.results


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


def rank_one(root: Path, payload: str, evidence_text: str, index: int) -> dict:
    """One reviewer over the frozen evidence bundle.

    Same unit as the concurrent runner: the flow dispatches it concurrently so
    every path asks the reviewers exactly the same question.
    """
    task_id = "scoring_" + str(index)
    prompt = (agent_contracts.RANK_PROMPT.format(
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY,
        rubric_version=scoring_module.RUBRIC["version"],
        proposals=payload,
        evidence=evidence_text))
    report = _rank_with_retry(root, task_id, prompt)
    report.setdefault("reviewer", task_id)
    report["rubric_version"] = scoring_module.RUBRIC["version"]
    report["rubric_sha256"] = scoring_module.RUBRIC["sha256"]
    return report


def scoring_inputs(root: Path, ideas: list, *, repo: Path = None,
                   state: dict = None) -> tuple:
    """The frozen bundle, the proposal text and the evidence text of a review.

    One source of truth: the concurrent runner and the flow both build the
    reviewers' input here, so a reviewer sees the same question in either path.
    """
    repo = Path(repo) if repo is not None else Path(root)
    frozen = (state or {}).get("evidence_bundles") or {}
    bundle = {idea["idea_id"]: frozen.get(idea["idea_id"])
              or evidence_module.build_bundle(idea, root=repo)
              for idea in ideas}
    payload = plaintext.render_ideas(ideas)
    evidence_text = "\n\n".join(
        "### " + identifier + "\n" + evidence_module.render_bundle(item)
        for identifier, item in bundle.items())
    return bundle, payload, evidence_text


def rank_order(ideas: list, reports: list) -> dict:
    """The reviewers' aggregate order, the selection input."""
    return _rank_order(ideas, reports)


def apply_bounded_revision(directory: Path, agent_root: Path, repo: Path,
                            state: dict, reports: list, outcome: dict):
    """The one bounded revision the reviewers may buy (public entry)."""
    return _one_bounded_revision(directory, agent_root, repo, state, reports, outcome)


def run_scoring(root: Path, ideas: list, *, repo: Path = None,
                state: dict = None) -> dict:
    """Three independent reviewers over the frozen evidence bundle.

    The bundle is assembled once and handed unchanged to all three, so their
    reviews are independent of each other but not of the evidence. An objection
    that names a mathematical identity, a cancellation, leakage or a dimension
    mismatch buys the proposal ONE bounded revision, and the revision is
    reviewed afresh under a new proposal hash.
    """
    repo = Path(repo) if repo is not None else Path(root)
    bundle, payload, evidence_text = scoring_inputs(root, ideas, repo=repo, state=state)
    def rank(index):
        return rank_one(root, payload, evidence_text, index)

    try:
        reports = _run_agents_concurrently(
            [tuple([index]) for index in range(1, SCORERS + 1)], rank)
    except AgentPhaseIncomplete as exc:
        # A reviewer that ran out of its budget did not rank. The rankings that
        # did arrive are kept: the aggregate is over the reviewers who answered.
        reports = [report for report in exc.results
                   if not (report.get("partial") or report.get("timed_out"))]
        if len(reports) < MIN_SCORERS_FOR_A_MATRIX:
            raise
    # The reviewers rank; the aggregate decides. There is no score, no threshold
    # and no head-to-head phase: the proposal the reviewers place best in total
    # is the round's mechanism.
    order = _rank_order(ideas, reports)
    return {"reports": reports, "bundles": bundle, "comparisons": [],
            "finalists": order,
            "finalist_scores": {identifier: entry["total_rank"]
                                 for identifier, entry in order.items()}}


HARD_OBJECTION_TERMS = ("cancel", "identity", "leak", "dimension", "degenerat",
                        "counterexample", "equivalence", "invariant")

#: How many reviewers a matrix needs. One review that answers is enough to
#: decide; the others may be out of time, and a round is not held hostage by the
#: reviewer that did not finish.
MIN_SCORERS_FOR_A_MATRIX = 1


def _hard_objection(report: dict) -> tuple:
    """The first objection that names a mathematical or leakage defect, if any."""
    for entry in report.get("scores", []):
        objection = str(entry.get("objection", "")).lower()
        counterexample = str(entry.get("counterexample", "")).strip()
        if counterexample or any(term in objection for term in HARD_OBJECTION_TERMS):
            return entry["idea_id"], entry.get("objection", ""), counterexample
    return None


def _one_bounded_revision(directory: Path, agent_root: Path, repo: Path, state: dict,
                          reports: list, outcome: dict) -> tuple:
    """Give a proposal ONE revision when a reviewer names a hard defect.

    The revised proposal gets a new hash, and the three reviewers see it afresh;
    the revision and the original objection stay in the state.
    """
    for report in reports:
        found = _hard_objection(report)
        if not found:
            continue
        identifier, objection, counterexample = found
        idea = ideas_module.idea_index(state["ideas"]).get(identifier)
        if idea is None:
            continue
        revision = int(idea.get("proposal_revision", 0))
        if revision >= 1:
            continue                      # the bounded budget is one revision
        try:
            answer = run_rebuttal(agent_root, idea, objection, counterexample, revision)
        except Exception as exc:                            # noqa: BLE001
            pipeline_module.record_event(state, "rebuttal_failed", idea_id=identifier,
                                         error=str(exc)[:200])
            pipeline_module.save_state(directory, state)
            continue
        updated = dict(idea)
        updated["proposal_revision"] = revision + 1
        updated["rebuttal"] = answer
        if answer.get("remaining"):
            updated["unresolved_objection"] = answer["remaining"]
        pipeline_module.record_event(state, "proposal_revised", idea_id=identifier,
                                     revision=updated["proposal_revision"],
                                     objection=objection[:200])
        state["ideas"] = [updated if entry["idea_id"] == identifier else entry
                          for entry in state["ideas"]]
        pipeline_module.save_state(directory, state)
        # The revised proposal must be scored against the same frozen bundle, so
        # the state has to travel with it; dropping it made the revision path
        # crash on the first re-score.
        outcome = run_scoring(agent_root, state["ideas"], repo=repo, state=state)
        return state, outcome["reports"], outcome
    return state, reports, outcome


def _rank_with_retry(root: Path, task_id: str, prompt: str) -> dict:
    """One ranking, and one bounded retry if its schema is broken.

    The ranking schema is strict on purpose: a missing proposal, a repeated one
    or a tied position is an error, not a partial average. A reviewer that
    breaks it is told what was wrong and asked once more, with a fresh
    workspace, so a stale answer can never be reused.
    """
    attempt = 0
    last_error = None
    # The budget belongs to the task, not to the attempt: a retry does not get
    # a second five minutes.
    import time as _time
    budget = float(settings.AGENT_DEADLINE_SECONDS)
    started = _time.monotonic()
    while attempt < 2:
        remaining = budget - (_time.monotonic() - started)
        if remaining <= 1.0:
            break
        suffix = "" if attempt == 0 else (
            "\n\nYour previous answer was rejected by the schema: " + str(last_error) +
            "\nAnswer again with one complete block per proposal, and keep the five lines "
            "of a block together:\n"
            "SCORE: <idea_id> | behavior_evidence=<0-5> | geometry_necessity=<0-5> | "
            "novelty=<0-5> | downstream_mechanism=<0-5> | falsifiability=<0-5> | "
            "implementability=<0-5> | exploration_value=<0-5>\n"
            "SCOPE: <idea_id> | IN_SCOPE or OUT_OF_SCOPE | <why>\n"
            "EVIDENCE: <idea_id> | <evidence ids>\n"
            "OBJECTION: <idea_id> | <the strongest objection>\n"
            "RATIONALE: <idea_id> | <why these numbers, and the uncertainty>\n"
            "Every dimension value is a number from 0 to 5. A dimension you cannot ground "
            "takes the number you can defend, and the RATIONALE says the dimension is "
            "UNKNOWN; UNKNOWN is never a dimension value. Every proposal you score needs "
            "its own RATIONALE line, carrying its own idea_id.")
        name = "scores.txt" if attempt == 0 else "scores_retry.txt"
        run_id = task_id if attempt == 0 else task_id + "_retry1"
        try:
            report = _run_agent(root, run_id, prompt + suffix, name,
                                plaintext.parse_ranks, role="score",
                                read_paths=[],
                                allowed_roots=_round_roots(root),
                                deadline_seconds=max(1, int(remaining)))
            scoring_module.validate_report(report, reviewer=task_id)
            return report
        except (scoring_module.ScoreError, pipeline_module.RoundError) as exc:
            last_error = exc
            attempt += 1
    raise pipeline_module.RoundError("reviewer " + task_id + " broke the ranking schema "
                                     "twice: " + str(last_error))


def _rank_order(ideas: list, reports: list) -> dict:
    """The aggregate order of the proposals: the reviewers' ranks added up.

    The proposal the reviewers place first most often, and in total least far
    down, is the round's mechanism. The order is returned as a mapping from the
    idea id to its row, already sorted.
    """
    matrix = scoring_module.build_matrix(ideas, reports)
    rows = sorted(matrix["rows"],
                  key=lambda row: (row.get("total_rank", 0),
                                   -row.get("first_places", 0),
                                   row["idea_id"]))
    return {row["idea_id"]: row for row in rows}


def run_rebuttal(root: Path, idea: dict, objection: str, counterexample: str,
                 revision: int) -> dict:
    """One bounded repair of a proposal whose objection names a defect."""
    prompt = agent_contracts.REBUTTAL_PROMPT.format(objection=objection,
                                                    counterexample=counterexample or "none")
    return _run_agent(root, "rebuttal_" + idea["idea_id"] + "_r" + str(revision),
                      prompt, "rebuttal.txt", plaintext.parse_rebuttal, role="score",
                      read_paths=[],
                      allowed_roots=_round_roots(root))


def _repository_root(start: Path) -> Path:
    """The repository root above a round directory, found by the marker."""
    here = Path(start).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() and (parent / "stage2_RQ-VAE").is_dir():
            return parent
    return here


def run_implementation(root: Path, idea: dict, tree: Path) -> dict:
    """One implementation agent per selected idea, in its own tree.

    The tree is a fresh copy of the one permitted baseline, made here, so the
    agent never starts from another candidate's work and never from a mutated
    worktree.
    """
    tree = Path(tree)
    provenance = baseline_module.copy_baseline(tree)
    prompt = (agent_contracts.IMPLEMENT_PROMPT.format(
        idea=plaintext.render_idea(idea), tree=str(tree),
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY))
    # No implementation agent runs in the Issue #121 flow: the framework copies
    # the baseline and the main agent writes the mechanism. If this role is
    # re-enabled, it also needs the Stage 0/1 inputs and the interpreter as read
    # paths (it must run the smoke); the tree alone is not enough.
    report = _run_agent(root, "implementation_" + idea["idea_id"], prompt,
                        "implementation.txt", plaintext.parse_implementation,
                        role="implementation")
    report["provenance"] = provenance
    report["baseline"] = provenance["baseline"]
    report["baseline_digest"] = provenance["baseline_digest"]
    report["candidate_digest_before_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "PROPOSAL.json", idea)
    # Issue #124: the candidate's own path is checked, not just the generic
    # probe. A candidate whose implemented operator is insensitive to the
    # behaviour signal, or that loses the gradient, is blocked here.
    try:
        from . import candidate_checks
        check = candidate_checks.run_candidate_check(tree, repo=_repository_root(Path(root)))
    except Exception as exc:                              # noqa: BLE001
        check = {"verdict": "INCONCLUSIVE_OR_INFRA_ERROR",
                 "reason": "the candidate path check crashed: " + type(exc).__name__}
    report["candidate_path_check"] = check
    report["candidate_digest_after_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "CANDIDATE_CHECK.json", check)
    return report



#: The cap on the diff a reviewer is shown. A revision that rewrites half the
#: tree is reviewed from the whole file list anyway; the patch names what moved.
DIFF_LIMIT = 60000

#: Files that belong to a run, not to the source: never part of a snapshot or a
#: diff, because they say nothing about what was implemented.
_DIFF_IGNORED = ("logs", "results", "dataset", "__pycache__", "ckpt", ".git")


def _source_files(root: Path) -> list:
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in _DIFF_IGNORED for part in relative.parts):
            continue
        files.append(relative)
    return files





















def scope_file(root: Path) -> Path:
    """Where the round's scope contract is written for a reviewer to read."""
    target = Path(root) / "SCOPE_CONTRACT.md"
    if not target.is_file():
        target.write_text(agent_contracts.SCOPE_CONTRACT + "\n\n" +
                          agent_contracts.EXPERIMENT_POLICY + "\n", encoding="utf-8")
    return target


def objective_summary(objective: str) -> str:
    """The objective in one line: the prompt carries the file, not the text."""
    for line in str(objective).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:240]
    return str(objective)[:240]


















def phase_ideation(root: Path, round_id: str, *, log=print) -> dict:
    """Run the ideation agents and merge what they proposed."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "ideation":
        return {"phase": "ideation", "ran": False, "step": state["step"],
                "ideas": len(state.get("ideas", [])),
                "reason": "the round is no longer in the ideation phase"}
    answers = run_ideation(agent_root, state.get("objective", ""))
    state = pipeline_module.apply_ideation(state, answers, require_minimum=False)
    if not state.get("ideas"):
        # Nothing was proposed. That is a result about the round's objective, not
        # a reason to end it: ask the same agents for one more idea each.
        extra = run_ideation(agent_root, state.get("objective", ""), extra=True)
        state = pipeline_module.apply_ideation(state, extra, require_minimum=False)
        if not state.get("ideas"):
            raise pipeline_module.RoundError(
                "a round needs at least one distinct idea, got none")
    pipeline_module.save_state(directory, state)
    log("ideation: %d ideas" % len(state["ideas"]))
    return {"phase": "ideation", "ran": True, "step": state["step"],
            "ideas": len(state["ideas"]),
            "rejected": len(state.get("ideas_rejected", [])),
            "duplicates": len(state.get("duplicates", []))}


def phase_probe_step(root: Path, round_id: str, *, log=print) -> dict:
    """The pre-score probe: measure, then freeze the evidence bundle."""
    return phase_probe(root, round_id, log=log)


def phase_scoring(root: Path, round_id: str, *, log=print) -> dict:
    """Run the three scorers over every idea and select the qualified ones."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "scoring":
        return {"phase": "scoring", "ran": False, "step": state["step"],
                "selected": len((state.get("selection") or {}).get("selected", [])),
                "reason": "the round is no longer in the scoring phase"}
    outcome = run_scoring(agent_root, state["ideas"], repo=Path(root), state=state)
    reports = outcome["reports"]
    state, reports, outcome = _one_bounded_revision(
        directory, agent_root, Path(root), state, reports, outcome)
    state = pipeline_module.apply_scoring(
        state, reports, comparisons=outcome.get("comparisons"),
        bundles=outcome.get("bundles"), directory=directory)
    pipeline_module.save_state(directory, state)
    selection = state.get("selection") or {}
    log("selection: %s" % selection.get("status", "?"))
    return {"phase": "scoring", "ran": True, "step": state["step"],
            "rubric": state.get("rubric_version"),
            "status": selection.get("status"),
            "decision_hash": state.get("decision_hash"),
            "scorers": len(reports),
            "finalists": outcome.get("finalists"),
            "finalist_scores": outcome.get("finalist_scores"),
            "selected": [entry["idea_id"] for entry in selection.get("selected", [])],
            "backups": [entry["idea_id"] for entry in selection.get("backups", [])],
            "rejected": [(entry["idea_id"], entry.get("status"))
                         for entry in selection.get("rejected", [])]}


def _parse_derivation(text: str) -> dict:
    """The first JSON object in an answer, or an empty derivation.

    The derivation's ``code`` field contains dict literals and f-strings, so a
    brace counter sees braces inside strings and closes in the wrong place. The
    decoder knows which braces are quoting: it is the only parser that survives
    a torch implementation embedded as a JSON string.
    """
    import json as _json

    # strict=False: a model writes a raw newline inside a long string (a wrapped
    # torch body, a wrapped expression). The decoder's strict mode refuses the
    # whole answer over one control character; the derivation is then validated
    # for real by sympify and by running it, which is the check that matters.
    decoder = _json.JSONDecoder(strict=False)
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("statements"):
            return payload
    return {}


#: The instance every derivation of a round must use. Two derivations that
#: choose different dimensions are different functions, and the comparison of
#: them says nothing; the instance is therefore fixed by the framework and
#: checked, not chosen by the agents.
DERIVATION_INSTANCE = {"latent_width": 2, "levels": 2, "codes": 2,
                       "latent_name": "z", "codebook_name": "theta",
                       "scale_name": "s", "anchor_name": "v_pi",
                       "output": "loss"}


def derivation_signature() -> tuple:
    """The arguments and the symbols of the fixed instance.

    The framework owns them: the agents fill the statements, the output name
    and the code. An agent that invents its own arguments produces a different
    function, and the comparison of two such answers says nothing.
    """
    instance = DERIVATION_INSTANCE
    width = int(instance["latent_width"])
    args = [{"name": instance["latent_name"], "width": width, "radius": 0.4}]
    symbols = ["%s%d" % (instance["latent_name"], index) for index in range(width)]
    for level in range(int(instance["levels"])):
        for code in range(int(instance["codes"])):
            name = "%s_%d_%d" % (instance["codebook_name"], level + 1, code + 1)
            args.append({"name": name, "width": width, "radius": 0.3})
            symbols.extend("%s%d_%d_%d" % (instance["codebook_name"], level + 1,
                                           code + 1, index) for index in range(width))
    args.append({"name": instance["scale_name"], "scalar": 1.0})
    symbols.append(instance["scale_name"])
    for code in range(int(instance["codes"])):
        name = "%s_%d" % (instance["anchor_name"], code + 1)
        args.append({"name": name, "width": width, "radius": 0.3})
        symbols.extend("%s%d_%d" % (instance["anchor_name"], code + 1, index)
                       for index in range(width))
    return args, symbols


def _instance_codebook_width() -> int:
    instance = DERIVATION_INSTANCE
    return int(instance["levels"]) * int(instance["codes"]) * int(instance["latent_width"])


def _instance_problems(payload: dict) -> list:
    """Where a derivation departs from the fixed instance."""
    instance = DERIVATION_INSTANCE
    problems = []
    arguments = payload.get("args") or []
    widths = {str(spec.get("name")): spec for spec in arguments}
    latent = [spec for spec in arguments
              if str(spec.get("name")).lower().startswith(instance["latent_name"])]
    if not latent:
        problems.append("no latent argument named " + instance["latent_name"])
    else:
        width = int(latent[0].get("width") or 1)
        if width != int(instance["latent_width"]):
            problems.append("the latent width is %d, the round fixes %d"
                            % (width, int(instance["latent_width"])))
    codebook = [spec for spec in arguments
                if str(spec.get("name")).lower().startswith(instance["codebook_name"])]
    if not codebook:
        problems.append("no codebook argument named " + instance["codebook_name"])
    else:
        # The packing of the codebook is a diagnostic, not a failure: a
        # derivation may write one argument per level or one per codeword, and
        # the comparison aligns the arguments by position.
        # The codebook may be one argument or one argument per (level, code)
        # pair: what the round fixes is the TOTAL number of codeword
        # components, not how they are packed.
        total = sum(int(spec.get("width") or 1) for spec in codebook)
    if not any(str(spec.get("name")).lower().startswith(instance["scale_name"])
               and spec.get("scalar") is not None for spec in arguments):
        problems.append("no scalar scale named " + instance["scale_name"])
    output = str(payload.get("output") or "")
    if output != instance["output"]:
        problems.append("the output is %r, the round fixes %r"
                        % (output, instance["output"]))
    return problems


def _code_signature(payload: dict) -> list:
    """The parameter names of a derivation's own code, or an empty list."""
    import ast as _ast

    code = payload.get("code") or ""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "compute":
            names = []
            for argument in node.args.args:
                names.append(argument.arg)
            if node.args.kwarg is not None:
                names = []
            return names
    return []


def _code_accepts(payload: dict, args: list) -> bool:
    """Whether the derivation's code can be called with the round's arguments.

    A code that takes **kwargs accepts everything; a code with named
    parameters must not require a name the round does not declare.
    """
    import ast as _ast

    code = payload.get("code") or ""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False
    declared = {str(spec["name"]) for spec in args}
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "compute":
            if node.args.kwarg is not None:
                return True
            required = [argument.arg for argument in node.args.args
                        if argument.arg not in ("self",)]
            return all(name in declared for name in required)
    return False


def _derivation_prompt(idea: dict) -> str:
    """One specification, no other derivation, and no way to ask for one."""
    instance = DERIVATION_INSTANCE
    args, symbols = derivation_signature()
    argument_lines = "\n".join(
        "      %s: %s" % (spec["name"],
                         ("a scalar" if spec.get("scalar") is not None
                          else "a vector of %d components" % int(spec["width"])))
        for spec in args)
    symbol_lines = ", ".join(symbols)
    return (
        "FIXED INSTANCE - the round fixes it and you must use exactly these "
        "arguments and these symbols. A derivation with its own signature is a "
        "different function and cannot be compared with the others.\n"
        "  * the arguments, in this exact order:\n" + argument_lines + "\n"
        "  * the free symbols, in this exact order (a vector argument of width w "
        "consumes w symbols, one per component): " + symbol_lines + "\n"
        "  * the output is a single per-item vector named '" + instance["output"] +
        "' - the quantity the mechanism changes in the training objective.\n"
        "  * the code is a complete Python module defining compute(**kwargs) with "
        "exactly these argument names, and it must compute the SAME formula as "
        "the statements.\n"
        "  * `symbols` and `args` are NOT part of your answer: the round owns "
        "them. You answer with the statements, the output name, the code and the "
        "excluded surfaces.\n\n"
        "Derive the formula of this mechanism from the specification alone.\n\n"
        "TITLE: " + str(idea.get("title")) + "\n"
        "MECHANISM: " + str(idea.get("mechanism")) + "\n"
        "SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "INPUTS AND OUTPUTS: " + str(idea.get("spec_inputs_outputs")) + "\n\n"
        "FIXED INSTANTIATION - every derivation must use exactly this, or the "
        "two cannot be compared:\n"
        "  * latent width D = 2;\n"
        "  * the smallest complete instance: L = 2 levels, K = 2 codes per level;\n"
        "  * arguments, in this exact order and with these exact names:\n"
        "      z (width 2) - the encoder vector,\n"
        "      theta_1_1, theta_1_2 (width 2), theta_2_1, theta_2_2 (width 2) - "
        "the codebook vectors in (level, code) order,\n"
        "      s (scalar) - the frozen scale.\n"
        "  * output: a single per-item vector named `loss` (one value per row of z), "
        "the quantity the mechanism adds or changes in the training objective.\n"
        "  * a switch or a table that the specification needs may be carried as "
        "an extra argument (a scalar 0/1 or a width-2 vector) placed AFTER s, and "
        "must be named in `args` and `symbols` in the same order.\n"
        "  Symbols follow the arguments: a width-w argument consumes w symbols in "
        "column order (z -> z0, z1; theta_1_1 -> t110, t111 ...).\n\n"
        "Turn the equations into SymPy statements and into a complete torch "
        "module defining compute(**kwargs), as the answer format requires.")


def run_derivations(directory: Path, ideas: list, *, log=print) -> list:
    """Three independent derivations of the one selected mechanism."""
    from ..agents import runtime as agent_runtime

    answers = []
    for idea in ideas:
        # A derivation answers in a workspace that a previous pass already used.
        # Its old answer must not be read as this pass's delivery when the new
        # agent times out: it is moved aside before the agents start.
        for index in range(1, 4):
            stale = directory / "agents" / ("derivation_%d" % index) / "derivation.json"
            if stale.is_file():
                archive = stale.parent / "previous" / str(int(time.time()))
                archive.mkdir(parents=True, exist_ok=True)
                stale.replace(archive / "derivation.json")

        def derive(task_id, _idea=idea):
            return _run_agent(
                directory, task_id, _derivation_prompt(_idea), "derivation.json",
                _parse_derivation, role="derive", read_paths=[],
                allowed_roots=_round_roots(directory))

        task_ids = ["derivation_1", "derivation_2", "derivation_3"]
        outcomes = _run_agents_concurrently([(task_id,) for task_id in task_ids],
                                            lambda task_id: derive(task_id))
        delivered, missing = [], []
        for task_id, outcome in zip(task_ids, outcomes):
            payload = {key: outcome.get(key) for key
                       in ("statements", "output", "code", "excluded",
                           "symbols", "args")
                       if outcome.get(key)}
            if payload.get("statements") and not payload.get("args"):
                args, symbols = derivation_signature()
                payload["args"] = args
                payload["symbols"] = symbols
            if payload.get("statements") and payload.get("code"):
                delivered.append({"idea_id": idea["idea_id"],
                                  "index": int(task_id.split("_")[-1]),
                                  "payload": payload, "receipt": outcome.get("receipt")})
            else:
                missing.append(int(task_id.split("_")[-1]))
            log("%s %s: %s" % (idea["idea_id"], task_id,
                               "ok" if payload.get("statements") else "no readable answer"))
        args, symbols = derivation_signature()
        for answer in list(delivered):
            payload = answer.get("payload") or {}
            problems = _instance_problems(payload) if payload else ["no payload"]
            if not problems and not _code_accepts(payload, args):
                problems = ["the code cannot be called with the round's arguments"]
            if not problems:
                continue
            index = int(answer["index"])
            log("%s derivation %d: the instance does not match (%s); asking again"
                % (idea["idea_id"], index, "; ".join(problems)))
            retry_id = "derivation_%d" % index
            correction = (_derivation_prompt(idea) +
                          "\n\nYOUR PREVIOUS DERIVATION DID NOT FIT THE INSTANCE: " +
                          "; ".join(problems) +
                          ". Use exactly the arguments listed in the FIXED INSTANCE "
                          "block above and nothing else, and your code must define "
                          "compute with exactly these parameter names, in this order: "
                          + ", ".join(spec["name"] for spec in args) +
                          ". Your statements must use exactly these symbols: " +
                          ", ".join(symbols) + ".")
            outcome = _run_agent(directory, retry_id, correction, "derivation.json",
                                 _parse_derivation, role="derive", read_paths=[],
                                 allowed_roots=_round_roots(directory))
            payload = {key: outcome.get(key) for key
                       in ("statements", "output", "code", "excluded",
                           "symbols", "args")
                       if outcome.get(key)}
            if payload.get("statements") and not payload.get("args"):
                args, symbols = derivation_signature()
                payload["args"] = args
                payload["symbols"] = symbols
            answer["payload"] = payload
            answer["revision"] = True
        if missing and len(delivered) < 2:
            # A derivation that delivered nothing is asked once more - but only
            # when the round would otherwise have fewer than two. Waiting for a
            # fifth minute of a third derivation while two are on disk holds the
            # comparison for evidence it does not need; the missing one is
            # recorded as a gap in the verdict instead.
            retry = ["derivation_%d" % index for index in missing]
            log("%s: asking %s again" % (idea["idea_id"], ", ".join(retry)))
            outcomes = _run_agents_concurrently([(task_id,) for task_id in retry],
                                                lambda task_id: derive(task_id))
            for task_id, outcome in zip(retry, outcomes):
                payload = {key: outcome.get(key) for key
                           in ("symbols", "args", "statements", "output", "code", "excluded")
                           if outcome.get(key)}
                delivered.append({"idea_id": idea["idea_id"],
                                  "index": int(task_id.split("_")[-1]),
                                  "payload": payload, "receipt": outcome.get("receipt")})
        answers.extend(delivered)
    return answers


def _run_differential(implementations: list, *, repo: Path) -> dict:
    """The deterministic N-version comparison, in the Stage 2 interpreter."""
    import subprocess
    import tempfile
    from .. import settings

    payload = {"implementations": implementations}
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.write(json.dumps(payload))
    handle.close()
    # The comparison needs sympy, sympytorch and torch: that is the Stage 2
    # interpreter, the one the framework's own checks run in.
    from ..stage2 import contract as stage2_contract

    command = [stage2_contract.STAGE2_PYTHON, "-m",
               "tools.research.rounds.differential",
               "--implementations", handle.name]
    done = subprocess.run(command, cwd=str(repo), capture_output=True, text=True,
                          timeout=1800)
    lines = [line for line in done.stdout.strip().splitlines() if line.startswith("{")]
    if not lines:
        return {"status": "NOT_RUN",
                "reason": "the comparison produced no report: " +
                          (done.stdout + done.stderr).strip()[-300:]}
    return json.loads(lines[-1])


def phase_derivation(root: Path, round_id: str, *, log=print) -> dict:
    """Three independent derivations of the selected mechanism, then compare them."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    selected = pipeline_module.selected_ideas(state)
    if not selected:
        return {"phase": "derivation", "ran": False,
                "reason": "the round has no selected idea"}
    if state.get("derivations"):
        return {"phase": "derivation", "ran": False,
                "reason": "the derivations are on disk",
                "status": (state.get("differential") or {}).get("status")}
    answers = run_derivations(directory, selected, log=log)
    implementations = [{"name": "%s_%d" % (answer["idea_id"], answer["index"]),
                        **(answer.get("payload") or {})}
                       for answer in answers if answer.get("payload")]
    mismatched = []
    comparable = []
    for implementation in implementations:
        problems = _instance_problems(implementation)
        if problems:
            implementation["instance_problems"] = problems
            mismatched.append({"name": implementation["name"],
                                "status": "INSTANCE_MISMATCH",
                                "reason": "; ".join(problems)})
        else:
            comparable.append(implementation)
    implementations = comparable
    verdict = (_run_differential(implementations, repo=Path(root))
               if len(implementations) >= 2
               else {"status": "NOT_APPLICABLE",
                     "reason": "fewer than two derivations answered"})
    state["derivation_instance"] = dict(DERIVATION_INSTANCE)
    state["derivations"] = [{"idea_id": answer["idea_id"],
                             "index": answer["index"],
                             "payload": answer.get("payload") or {},
                             "receipt": answer.get("receipt")}
                            for answer in answers]
    state["differential"] = verdict
    if mismatched:
        state["differential"] = dict(verdict)
        state["differential"]["instance_mismatches"] = mismatched
    pipeline_module.save_state(directory, state)
    log("derivation: %s" % verdict.get("status"))
    return {"phase": "derivation", "ran": True, "status": verdict.get("status"),
            "derivations": len(implementations)}


def phase_implementation(root: Path, round_id: str, *, log=print) -> dict:
    """Build one isolated candidate per selected idea, each from the baseline."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "implementation":
        return {"phase": "implementation", "ran": False, "step": state["step"],
                "implementations": len(state.get("implementations", [])),
                "reason": "the round is no longer in the implementation phase"}
    # The three independent derivations of the selected mechanism come first:
    # registration is refused until they agree.
    derivation = phase_derivation(root, round_id, log=log)
    if derivation.get("status") not in (None, "AGREE", "NOT_APPLICABLE"):
        raise pipeline_module.RoundError(
            "the independent derivations do not agree: " +
            str(derivation.get("status")))
    trees_root = (Path(root) / "results" / "research" / state.get("issue", "issue120")
                  / "trees" / round_id)
    for idea in pipeline_module.selected_ideas(state):
        tree = trees_root / idea["idea_id"]
        if any(record["idea_id"] == idea["idea_id"]
               for record in state.get("implementations", [])):
            continue
        # There is no implementation agent: the main agent writes the mechanism.
        # The candidate tree is prepared here as a fresh copy of the one
        # permitted baseline, and the record waits for the main agent to write
        # the code and register it (research_round_register_implementation).
        try:
            tree.parent.mkdir(parents=True, exist_ok=True)
            # One copy path for every candidate: it renames the framework-owned
            # ARM_NAME to this tree, records the baseline path and both digests,
            # and refuses a copy that changed anything else. A plain copytree left
            # the candidate pointing at the baseline's result directory.
            provenance = baseline_module.copy_baseline(tree)
            record = pipeline_module.implementation_record(idea, {}, tree)
            record["baseline"] = provenance["baseline"]
            record["baseline_digest"] = provenance["baseline_digest"]
            record["arm_renamed"] = provenance["arm_renamed"]
            record["status"] = "awaiting_implementation"
            record["implemented_by"] = "main_agent"
            state.setdefault("implementations", []).append(record)
            pipeline_module.save_state(directory, state)
            log("%s: candidate tree prepared for the main agent; it must declare "
                "its equations in reference_contract.json - registration is "
                "refused until the framework rebuilds them and they agree"
                % idea["idea_id"])
        except baseline_module.BaselineError as exc:
            # The baseline is the one permitted source: if it is missing or
            # unreadable the candidate is not built from anything else.
            state = pipeline_module.mark_blocked(
                state, idea["idea_id"], "baseline unavailable: " + str(exc),
                kind=pipeline_module.BASELINE_UNAVAILABLE)
            pipeline_module.save_state(directory, state)
            continue
        record = [item for item in state["implementations"]
                  if item["idea_id"] == idea["idea_id"]][0]
        # The candidate must write only into its own result directory: a
        # candidate that would overwrite the baseline or a sibling is
        # refused before it can train.
        siblings = [other["tree"] for other in state.get("implementations", [])
                    if other["idea_id"] != idea["idea_id"]]
        try:
            record["output_paths"] = baseline_module.guard_output_paths(
                tree, other_candidates=siblings)
        except baseline_module.BaselineError as exc:
            raise pipeline_module.RoundError(
                idea["idea_id"] + " writes outside its own result directory: " +
                str(exc)) from exc
        pipeline_module.save_state(directory, state)
        log("implemented %s" % idea["idea_id"])
    state["step"] = "stage2"
    pipeline_module.save_state(directory, state)
    return {"phase": "implementation", "ran": True, "step": state["step"],
            "implementations": [{"idea_id": record["idea_id"], "status": record["status"],
                                 "revision": record.get("revision", 0)}
                                for record in state.get("implementations", [])]}






#: The phases of a round, in the order they run. Verification is not a phase:
#: it is the framework's machine admission at launch.
PHASES = ("ideation", "probe", "scoring", "implementation")


def advance(root: Path, round_id: str, *, log=print) -> dict:
    """Run the agent stages of a round until it waits for Stage 2."""
    from . import lock as lock_module
    from . import pipeline as pipeline_module

    directory = pipeline_module.find_round_directory(root, round_id)
    lock = lock_module.RoundLock(directory)
    # One runner per round: the lock keeps a detached worker and a synchronous
    # phase call from paying for the same agents twice.
    lock.acquire(owner="worker:" + str(round_id))
    try:
        for phase in (phase_ideation, phase_probe, phase_scoring, phase_implementation):
            phase(root, round_id, log=log)
    finally:
        lock.release()
    return pipeline_module.summary(
        pipeline_module.load_state(pipeline_module.find_round_directory(root, round_id)))
