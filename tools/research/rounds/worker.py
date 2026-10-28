"""The round worker: the agent half of an Issue #120 round.

It runs the five ideation agents, the three scoring agents, and then, for every
selected idea, one implementation agent followed by three verification agents
and, when they reject, a dedicated repair agent. It never trains anything: the
Stage 2 and Stage 3 jobs are prepared here and executed by the MCP, which owns
the GPU boundary.
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
#: How many independent verifiers read one implementation revision.
VERIFIERS = pipeline_module.VERIFIER_AGENTS


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
    expected_boundary = boundary_fingerprint(workspace, role, read_paths or [],
                                             allow_shell, allow_write, prompt,
                                             settings.AGENT_MODEL,
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
                            binary=settings.AGENT_BINARY, model=settings.AGENT_MODEL,
                            role=role, read_paths=read_paths,
                            allowed_roots=allowed_roots, candidate_tree=candidate_tree,
                            allow_shell=allow_shell, allow_write=allow_write,
                            deadline_seconds=deadline_seconds)
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


def _run_math_subprocess(idea: dict, *, seed: int) -> dict:
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
                process.wait(timeout=120.0)
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


def run_probes(root: Path, ideas: list, *, repo: Path, cache_dir: Path = None) -> dict:
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

    def measure(idea: dict) -> tuple:
        """The three measurements of one proposal, at the same time.

        The data probe and the math checks are separate processes and the
        literature pass is the network: running them one after another made the
        phase cost the sum of their budgets for every proposal.
        """
        identifier = idea["idea_id"]

        def data():
            report = _run_probe_subprocess(
                idea, repo=repo, budget=probes_module.DEFAULT_BUDGET_SECONDS,
                max_users=probes_module.DEFAULT_MAX_USERS,
                seed=probes_module.DEFAULT_SEED)
            report["sha256"] = _probe_digest(report)
            return report

        def math():
            report = _run_math_subprocess(idea, seed=math_checks_module.SEED)
            report["sha256"] = _probe_digest(report)
            return report

        def literature():
            report = literature_module.run_literature(idea, cache_dir=cache_dir)
            report["sha256"] = _probe_digest(report)
            return report

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(data), pool.submit(math), pool.submit(literature)]
            probe, math_report, lit = [future.result() for future in futures]
        return identifier, {"probe": probe, "literature": lit, "math": math_report}

    # The proposals are independent, so they are measured at the same time.
    outcomes = {}
    with ThreadPoolExecutor(max_workers=min(len(ideas), 6) or 1) as pool:
        for identifier, outcome in pool.map(measure, ideas):
            outcomes[identifier] = outcome
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
    outcomes = run_probes(directory, ideas, repo=Path(root))
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
    frozen = state.get("evidence_bundles") or {}
    bundle = {idea["idea_id"]: frozen.get(idea["idea_id"])
              or evidence_module.build_bundle(idea, root=repo)
              for idea in ideas}
    payload = plaintext.render_ideas(ideas)
    evidence_text = "\n\n".join(
        "### " + identifier + "\n" + evidence_module.render_bundle(item)
        for identifier, item in bundle.items())
    def rank(index):
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

#: How many reviewers a matrix needs. The selection rule asks for two
#: reviewers above the threshold, so two reviews are enough to decide; one is
#: not, because a single reviewer cannot show a disagreement.
MIN_SCORERS_FOR_A_MATRIX = 2


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


VERIFIER_ROLES = agent_contracts.VERIFIER_ROLES

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


def snapshot_tree(root: Path, idea_id: str, revision: int, tree: Path) -> str:
    """Keep the source of one reviewed revision, so the next one has a baseline.

    The snapshot is what the NEXT review diffs against: a reviewer of revision r
    sees exactly what changed since revision r-1 was reviewed, not a diff against
    the original baseline that grows with every fix.
    """
    root, tree = Path(root), Path(tree)
    target = root / "snapshots" / (idea_id + "_r" + str(int(revision)))
    if target.exists():
        shutil.rmtree(target)
    for relative in _source_files(tree):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tree / relative, destination)
    return str(target)


def tree_diff(root: Path, idea_id: str, previous_revision: int, tree: Path) -> str:
    """The unified diff between the last reviewed snapshot and the tree now."""
    import difflib

    root, tree = Path(root), Path(tree)
    snapshot = root / "snapshots" / (idea_id + "_r" + str(int(previous_revision)))
    if not snapshot.is_dir():
        return ""
    chunks = []
    for relative in _source_files(tree):
        old_path = snapshot / relative
        new_path = tree / relative
        old = old_path.read_text(encoding="utf-8", errors="replace").splitlines(True) \
            if old_path.is_file() else []
        new = new_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        chunks.extend(difflib.unified_diff(old, new, fromfile="a/" + str(relative),
                                           tofile="b/" + str(relative), n=3))
    removed = []
    for relative in _source_files(snapshot):
        if not (tree / relative).is_file():
            removed.extend(difflib.unified_diff(
                (snapshot / relative).read_text(encoding="utf-8",
                                                errors="replace").splitlines(True),
                [], fromfile="a/" + str(relative), tofile="/dev/null", n=3))
    text = "".join(chunks + removed)
    if len(text) > DIFF_LIMIT:
        text = text[:DIFF_LIMIT] + "\n... truncated: the patch is longer than " + \
            str(DIFF_LIMIT) + " characters, read the files for the rest\n"
    return text


def full_change_list(root: Path, idea_id: str, previous_revision: int, tree: Path) -> list:
    """Every path that was added, modified or deleted since the last snapshot."""
    root, tree = Path(root), Path(tree)
    snapshot = root / "snapshots" / (idea_id + "_r" + str(int(previous_revision)))
    if not snapshot.is_dir():
        return []
    changed = []
    for relative in _source_files(tree):
        old_path = snapshot / relative
        if not old_path.is_file():
            changed.append(str(relative))
            continue
        if old_path.read_bytes() != (tree / relative).read_bytes():
            changed.append(str(relative))
    for relative in _source_files(snapshot):
        if not (tree / relative).is_file():
            changed.append(str(relative))          # a deletion changes the area too
    return changed


def changed_files(patch: str) -> list:
    """The files a unified diff touches, from its headers."""
    import re
    seen = []
    # unified_diff writes "--- a/path" / "+++ b/path" pairs; a git-style
    # "diff --git a/path b/path" header is accepted too, so the parser does not
    # depend on which tool produced the patch.
    for match in re.finditer(r"^(?:diff --git a/(\S+) b/\S+|\+\+\+ b/(\S+))$",
                             patch or "", re.M):
        path = match.group(1) or match.group(2)
        if path and path != "/dev/null" and path not in seen:
            seen.append(path)
    return seen


def affected_roles(patch: str, findings=None, reported=None, files=None) -> list:
    """The roles whose area the patch touches, plus the ones that complained.

    A role that raised a finding keeps its seat until the finding is resolved:
    the fix is exactly what it has to check. When nothing can be attributed, all
    roles are dispatched, because a review with no reviewer is not a review.

    ``files`` is the complete changed-file list when the caller has it: the role
    selection must not depend on the TRUNCATED patch that goes into the prompt,
    or a change past the cutoff is silently left unreviewed.
    """
    files = list(files) if files is not None else changed_files(patch)
    roles = []
    if files:
        for role in VERIFIER_ROLES:
            reads = [str(item) for item in role.get("reads") or ()]
            # "." means "may list the tree", not "owns every file": a role whose
            # scope is the whole tree would otherwise be dispatched for every
            # change and the targeting would be pointless.
            if any(changed == read or changed.startswith(read.rstrip("/") + "/")
                   for changed in files for read in reads if read != "."):
                roles.append(role["id"])
    for finding in findings or []:
        role_id = finding.get("role")
        if not role_id:
            verifier = str(finding.get("verifier") or "")
            for index, role in enumerate(VERIFIER_ROLES, 1):
                if verifier.endswith("_" + str(index)):
                    role_id = role["id"]
                    break
        if role_id and role_id not in roles:
            roles.append(role_id)
    if not roles:
        roles = [role["id"] for role in VERIFIER_ROLES]
    ordered = [role["id"] for role in VERIFIER_ROLES if role["id"] in roles]
    return ordered or [role["id"] for role in VERIFIER_ROLES]


def write_findings_file(root: Path, idea_id: str, revision: int, findings) -> str:
    """The full previous findings, so the prompt never has to truncate them."""
    root = Path(root)
    target = root / "findings" / (idea_id + "_r" + str(int(revision)) + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"idea_id": idea_id, "reviewed_revision": int(revision),
               "findings": list(findings or [])}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(target)


def write_manifest(root: Path, idea_id: str, revision: int, tree: Path) -> str:
    """The list of source files a reviewer must know exists.

    A read scope hides files on purpose, but a reviewer that cannot see a file must
    not conclude it is absent: the manifest names every source file and its size,
    so "there is no model/model.py" is a statement about the tree rather than about
    the allowlist. It is written for every revision and given to every reviewer.
    """
    root, tree = Path(root), Path(tree)
    target = root / "manifests" / (idea_id + "_r" + str(int(revision)) + ".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# source manifest of the candidate tree",
             "# path | lines | bytes",
             "# a file absent from this list does not exist; a file in this list may",
             "# still be outside your read scope, and that is not a defect."]
    for relative in _source_files(tree):
        path = tree / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        lines.append("%s | %d lines | %d bytes" % (relative, text.count("\n") + 1,
                                                   path.stat().st_size))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def role_read_paths(tree: Path, role: dict) -> list:
    """The paths one role may read: its declared scope, under both of its names.

    The framework hands the agent a ``/home/...`` path while ``Path.resolve()``
    answers ``/fs04/...`` on this host, and a permission entry written with one
    name does not match a read made with the other. Both forms are emitted, so the
    allowlist matches whichever alias the agent uses.
    """
    tree = Path(tree)
    paths = []
    for relative in role.get("reads") or ():
        candidate = tree / relative
        if not candidate.exists():
            continue
        for form in (candidate, candidate.resolve()):
            text = str(form)
            if text not in paths:
                paths.append(text)
    # No fallback to the whole tree: a scope whose files are all missing is a
    # coverage gap that the caller records, not a licence to read everything.
    return paths


def read_gaps(paths) -> list:
    """The allowlist entries this process cannot actually open.

    A path that is mounted but unreadable turns a review into an inference from the
    prompt; the gap is recorded so the report says what was not read.
    """
    gaps = []
    for path in paths:
        entry = Path(path)
        try:
            if entry.is_dir():
                next(iter(entry.iterdir()), None)
            else:
                entry.open("rb").close()
        except OSError as exc:
            gaps.append({"path": str(entry), "error": type(exc).__name__})
    return gaps



def _format_role_checks(role: dict) -> str:
    """The CHECK blocks one reviewer must report, as prompt lines."""
    lines = []
    for index, (name, guidance) in enumerate(role.get("checks") or (), 1):
        lines.append("  %d) CHECK: %s -- %s" % (index, name, guidance))
    return "\n".join(lines)


def _format_findings(findings) -> str:
    """The previous review's blocking findings, with where each one was."""
    entries = list(findings or [])
    if not entries:
        return "  none: this is the first review of this revision"
    lines = []
    for finding in entries[:12]:
        lines.append("  - [%s] %s: %s" % (
            str(finding.get("severity") or "blocking"),
            str(finding.get("location") or "?"),
            str(finding.get("detail") or "")[:400]))
    if len(entries) > 12:
        lines.append("  ... and %d more, see the round state" % (len(entries) - 12))
    return "\n".join(lines)


def _format_change_brief(record: dict) -> str:
    """What the main agent changed, in its own words, plus its own evidence."""
    parts = []
    changed = list(record.get("changed_files") or [])
    parts.append("  files: " + (", ".join(changed) if changed else "not stated"))
    for key, label in (("changes", "changes"), ("notes", "mechanism"),
                       ("smoke", "the author's own smoke"), ("deviations", "deviations"),
                       ("assumptions", "assumptions")):
        value = record.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value)
        parts.append("  %s: %s" % (label, str(value)[:1200]))
    return "\n".join(parts)


def run_verification(root: Path, idea: dict, tree: Path, revision: int = 0,
                     record: dict = None) -> list:
    """Five independent verifiers of one implementation revision.

    The five are not five copies of the same review: each has a fixed primary
    area (VERIFIER_ROLES) with its own must-cover items, and all five receive the
    same mechanism brief and the previous review's findings, so a repair is
    checked against what it claimed to fix instead of being re-discovered from
    scratch.
    """
    record = record or {}
    previous_revision = max(0, int(revision) - 1)
    patch = tree_diff(root, str(idea["idea_id"]), previous_revision, tree) \
        if int(revision) > 0 else ""
    moved = full_change_list(root, str(idea["idea_id"]), previous_revision, tree)
    findings = list(record.get("blocking_findings") or [])
    findings_path = write_findings_file(root, str(idea["idea_id"]), int(revision), findings)
    manifest = write_manifest(root, str(idea["idea_id"]), int(revision), tree)
    # The tree this review reads becomes the baseline of the NEXT review's diff:
    # without this the second revision would be diffed against nothing.
    snapshot_tree(root, str(idea["idea_id"]), int(revision), tree)
    wanted = affected_roles(patch, findings, files=moved)
    roles = [role for role in VERIFIER_ROLES if role["id"] in wanted]
    change_brief = _format_change_brief(record)
    previous_findings = _format_findings(findings)
    if not patch:
        change_brief = ("  the tree was renamed or first submitted: there is no previous "
                        "reviewed revision to diff against\n" + change_brief)
    prompts = {}
    reads = {}
    gaps = {}
    role_index = {role["id"]: position for position, role in enumerate(VERIFIER_ROLES, 1)}
    for index, role in enumerate(roles, 1):
        prompts[index] = agent_contracts.VERIFY_PROMPT.format(
            role_index=index, role_total=len(roles), role_name=role["name"],
            role_must="\n".join("  - " + item for item in role["must"]),
            role_scope=role["scope"], role_checks=_format_role_checks(role),
            idea=plaintext.render_idea(idea), tree=str(tree),
            change_brief=change_brief,
            change_diff=(patch or "  none: this is the first review of this candidate"),
            findings_file=findings_path,
            manifest_file=manifest,
            previous_findings=previous_findings,
            objective=agent_contracts.RESEARCH_OBJECTIVE,
            scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
            agent_contracts.EXPERIMENT_POLICY)
        reads[index] = role_read_paths(tree, role) + [findings_path, manifest]
        gaps[index] = read_gaps(reads[index])

    def verify(index):
        # The revision is part of the workspace name: a fresh review after a
        # repair must be a fresh agent, never a replay of the previous answer.
        # The number is the role's position in the FULL role list, not in the
        # filtered dispatch: a stable id is what lets a later review attribute an
        # old finding to the reviewer that raised it.
        task_id = ("verify_" + idea["idea_id"] + "_r" + str(revision) + "_" +
                   str(role_index[roles[index - 1]["id"]]))
        prompt = prompts[index]
        report = _run_agent(root, task_id, prompt, "verification.txt",
                            plaintext.parse_verification,
                            role="verify", read_paths=reads[index],
                            allowed_roots=_round_roots(root),
                            candidate_tree=tree,
                            source_digest=pipeline_module._source_digest(Path(tree)))
        report.setdefault("verifier", task_id)
        report["role"] = roles[index - 1]["id"]
        report["coverage_gaps"] = gaps.get(index) or []
        return report

    try:
        reports = _run_agents_concurrently(
            [tuple([index]) for index in range(1, len(roles) + 1)], verify)
    except AgentPhaseIncomplete as exc:
        # A reviewer that ran out of its budget does not void the review: the
        # verdicts that exist are kept, the missing ones are recorded, and the
        # candidate is judged on the opinions the round actually has. Only a
        # round where no reviewer finished waits.
        reports = [report for report in exc.results
                   if not (report.get("partial") or report.get("timed_out"))]
        if not reports:
            raise RoundIncomplete(str(exc), results=exc.results, failures=exc.failures,
                                  classes=exc.classes) from exc
        missing = sorted(str(entry.get("task", "")) for entry in exc.failures)
        for report in reports:
            report["missing_reviewers"] = missing
        return reports
    # A reviewer that ran out of its budget delivered records, not a verdict: it
    # is not a verdict and it does not block the ones that are.
    finished = [report for report in reports
                if not (report.get("partial") or report.get("timed_out"))]
    unfinished = [report.get("verifier", "?") for report in reports
                  if report.get("partial") or report.get("timed_out")]
    if unfinished:
        missing = sorted(str(name) for name in unfinished)
        for report in finished:
            report["missing_reviewers"] = missing
    return finished


def run_repair_plan(root: Path, idea: dict, tree: Path, findings: list,
                    revision: int = 0) -> dict:
    """The repair PLANNER for one implementation revision.

    The planner may read the tree but must not change it. The rule is enforced,
    not merely requested: the tree's sources are copied aside before the agent
    runs and the digest is compared afterwards, so an agent that edits anyway
    has its edit rolled back and its plan rejected.
    """
    prompt = (agent_contracts.REPAIR_PLAN_PROMPT.format(
        findings=plaintext.render_findings(findings), tree=str(tree),
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY))
    backup = _tree_backup(tree)
    before = pipeline_module._source_digest(tree)
    plan = _run_agent(root, "repair_" + idea["idea_id"] + "_r" + str(revision),
                      prompt, "repair.txt", plaintext.parse_repair_plan,
                      role="repair", read_paths=[tree],
                      allowed_roots=_round_roots(root),
                      candidate_tree=tree)
    after = pipeline_module._source_digest(tree)
    if after != before:
        _tree_restore(tree, backup)
        raise pipeline_module.RoundError(
            "the repair planner modified the candidate tree, which it may not do; "
            "the tree was restored and the plan was discarded")
    plan["tree_digest"] = before
    return plan


def _tree_backup(tree: Path) -> dict:
    """A copy of every source file in the tree, for a rollback."""
    import shutil
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="repair_backup_"))
    files = {}
    for path in sorted(Path(tree).rglob("*.py")):
        if "__pycache__" in path.relative_to(tree).parts:
            continue
        target = directory / path.relative_to(tree)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[str(path.relative_to(tree))] = target
    return {"directory": directory, "files": files}


def _tree_restore(tree: Path, backup: dict) -> None:
    """Put the backed-up sources back and delete what the planner added."""
    import shutil

    tree = Path(tree)
    for path in sorted(tree.rglob("*.py")):
        if "__pycache__" in path.relative_to(tree).parts:
            continue
        relative = str(path.relative_to(tree))
        if relative not in backup["files"]:
            path.unlink()
    for relative, source in backup["files"].items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.rmtree(backup["directory"], ignore_errors=True)


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
    if len(state.get("ideas") or []) < ideas_module.MIN_ROUND_IDEAS:
        # The agents agreed with each other. That is a result about the round's
        # objective, not a reason to end it: ask the same agents for one more
        # idea each, so the round has three distinct proposals to compare.
        extra = run_ideation(agent_root, state.get("objective", ""), extra=True)
        state = pipeline_module.apply_ideation(state, extra, require_minimum=False)
        if len(state.get("ideas") or []) < ideas_module.MIN_ROUND_IDEAS:
            raise pipeline_module.RoundError(
                "a round needs at least " + str(ideas_module.MIN_ROUND_IDEAS) +
                " distinct ideas, got " + str(len(state.get("ideas") or [])))
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


def phase_implementation(root: Path, round_id: str, *, log=print) -> dict:
    """Build one isolated candidate per selected idea, each from the baseline."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "implementation":
        return {"phase": "implementation", "ran": False, "step": state["step"],
                "implementations": len(state.get("implementations", [])),
                "reason": "the round is no longer in the implementation phase"}
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
            log("%s: candidate tree prepared for the main agent" % idea["idea_id"])
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
            record["blocking_findings"] = [{
                "verifier": "preflight", "check": "output isolation",
                "severity": "blocking", "location": "train_rqvae.py",
                "detail": str(exc),
                "reproduction": "read the trainer's output constants",
                "correction": "write only into the candidate's own result directory"}]
            record["status"] = "needs_repair"
            record["failure_signature"] = pipeline_module._failure_signature(
                record["blocking_findings"])
        pipeline_module.save_state(directory, state)
        log("implemented %s" % idea["idea_id"])
    state["step"] = "verification"
    pipeline_module.save_state(directory, state)
    return {"phase": "implementation", "ran": True, "step": state["step"],
            "implementations": [{"idea_id": record["idea_id"], "status": record["status"],
                                 "revision": record.get("revision", 0)}
                                for record in state.get("implementations", [])]}


def phase_repair_plan(root: Path, round_id: str, idea_id: str, *, log=print) -> dict:
    """Ask the repair PLANNER for the plan of one candidate, on the main agent's call.

    The planner reads the findings and the tree and writes the change that would
    fix each finding; it never edits the tree. The main agent applies the plan
    and registers the execution.
    """
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    record = next((item for item in state.get("implementations", [])
                   if item["idea_id"] == idea_id), None)
    if record is None:
        raise pipeline_module.RoundError("no implementation " + idea_id)
    findings = record.get("blocking_findings") or []
    if not findings:
        raise pipeline_module.RoundError(idea_id + " has no blocking finding to plan for")
    idea = ideas_module.idea_index(state["ideas"])[idea_id]
    plan = run_repair_plan(directory, idea, Path(record["tree"]), findings,
                           int(record.get("revision", 0)))
    state = pipeline_module.apply_repair_plan(state, idea_id, plan)
    pipeline_module.save_state(directory, state)
    log("%s: repair plan ready (%d changes)" % (idea_id, len(plan.get("plans", []))))
    return {"phase": "repair_plan", "ran": True, "idea_id": idea_id,
            "revision": record.get("revision"),
            "plans": plan.get("plans", []),
            "remaining_issues": plan.get("remaining_issues", []),
            "notes": plan.get("notes", "")}


def phase_verification(root: Path, round_id: str, *, log=print) -> dict:
    """Run the verifier of each implementation and the repair loop."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "verification":
        return {"phase": "verification", "ran": False, "step": state["step"],
                "verified": len(pipeline_module.verified_implementations(state)),
                "reason": "the round is no longer in the verification phase"}
    for record in list(state.get("implementations", [])):
        # A tree whose source changed since the last review must be reviewed
        # again by a fresh agent: the revision is bumped, so the reviewer works
        # in a new workspace and no earlier answer is reused. This holds for a
        # verified record too: a source change invalidates the approval.
        if record.get("source_digest") and record["source_digest"] != \
                pipeline_module._source_digest(Path(record["tree"])):
            record["revision"] = int(record.get("revision", 0)) + 1
            record["status"] = "implemented"
            record["blocking_findings"] = []
            record["source_digest"] = pipeline_module._source_digest(Path(record["tree"]))
            pipeline_module.record_event(state, "source_changed_before_review",
                                         idea_id=record["idea_id"],
                                         revision=record["revision"])
            pipeline_module.save_state(directory, state)
        # A review produced under another boundary is not an approval under this
        # one: the candidate goes back to the queue and is reviewed again.
        if record.get("boundary_version") and \
                record["boundary_version"] != policy.BOUNDARY_VERSION:
            record["revision"] = int(record.get("revision", 0)) + 1
            record["status"] = "implemented"
            record["blocking_findings"] = []
            pipeline_module.record_event(state, "boundary_changed_before_review",
                                         idea_id=record["idea_id"],
                                         revision=record["revision"],
                                         boundary_version=record["boundary_version"],
                                         current_boundary=policy.BOUNDARY_VERSION)
            pipeline_module.save_state(directory, state)
        if record.get("status") not in ("implemented", "repaired", "agent_error",
                                        "needs_repair"):
            # "repair_planned" waits for the operator: the tree is unchanged, so
            # there is nothing new to review and nothing new to plan.
            continue
        # The output isolation is checked again before the reviewers read the
        # tree: a finding of an earlier pass described an older tree, and the
        # reviewers must not carry it into this one.
        pipeline_module._refresh_output_isolation(state, record)
        if record.get("status") == "needs_repair":
            pipeline_module.save_state(directory, state)
            continue
        idea = ideas_module.idea_index(state["ideas"])[record["idea_id"]]
        try:
            reports = run_verification(agent_root, idea, Path(record["tree"]),
                                       int(record.get("revision", 0)), record=record)
        except RoundIncomplete as exc:
            # Some reviewers answered and some did not. The answers that exist
            # are kept and the candidate WAITS: a missing reviewer is not a
            # verdict on the mechanism. An authentication or quota fault pauses
            # the service instead of spending the retry budget on a call that
            # cannot succeed.
            report = {"idea_id": record["idea_id"], "partial_verdicts": [
                str(item.get("verdict", "")).upper() for item in exc.results],
                "failures": exc.failures, "classes": sorted(exc.classes),
                "waiting": exc.waiting}
            state = pipeline_module.mark_agent_error(
                state, record["idea_id"],
                "verification incomplete: " + str(exc), details=report)
            pipeline_module.save_state(directory, state)
            log("%s: waiting for %d reviewer(s) (%s)" % (
                record["idea_id"], len(exc.failures), ",".join(sorted(exc.classes))))
            if exc.waiting:
                break
            continue
        except Exception as exc:                           # noqa: BLE001
            # A reviewer that crashed says nothing about the mechanism: the
            # branch is retried, and only an exhausted retry is recorded as
            # an infrastructure failure.
            state = pipeline_module.mark_agent_error(
                state, record["idea_id"], "verification failed: " + str(exc))
            pipeline_module.save_state(directory, state)
            continue
        state = pipeline_module.apply_verification(state, record["idea_id"], reports)
        pipeline_module.save_state(directory, state)
        if record.get("blocking_findings"):
            if int(record.get("review_rounds", 0)) >= pipeline_module.MAX_REVIEW_ROUNDS:
                # The candidate used up its repair rounds: it is recorded with
                # its findings instead of asking for another agent. The counter
                # is the number of times the verifiers rejected it, not the
                # number of source changes: a performance fix must not cost the
                # candidate the same budget as a mechanism that was rejected
                # again and again.
                state = pipeline_module.mark_revision_capped(
                    state, record["idea_id"],
                    "the verifiers still reject the implementation after " +
                    str(record.get("review_rounds", 0)) + " review rounds, which is the cap")
                pipeline_module.save_state(directory, state)
                log("%s: revision capped" % record["idea_id"])
                continue
            # The findings are reported and the round waits. Whether they need a
            # repair, and which of them do, is the main agent's decision: it can
            # ask for a plan (research_round_plan_repair) or accept the candidate
            # as it stands.
            pipeline_module.save_state(directory, state)
            log("%s: %s (%d blocking findings)" % (
                record["idea_id"], record.get("status"),
                len(record.get("blocking_findings") or [])))
    # Stay in verification until every implementation is resolved: a repair
    # reopens the review, and the round must not move on with work in flight.
    pending = [record for record in state.get("implementations", [])
               if record.get("status") in ("implemented", "repaired", "needs_repair",
                                           "repair_planned", "verified",
                                           "awaiting_implementation")
               or (record.get("status") == "agent_error"
                   and int(record.get("agent_error_attempts", 0))
                   <= pipeline_module.MAX_AGENT_RETRIES)]
    state["step"] = "verification" if pending else "stage2"
    pipeline_module.save_state(directory, state)
    return {"phase": "verification", "ran": True, "step": state["step"],
            "verified": len([record for record in state.get("implementations", [])
                             if record.get("status") == "verified"]),
            "accepted": [record["idea_id"] for record
                         in pipeline_module.verified_implementations(state)],
            "verdicts": [{"idea_id": record["idea_id"],
                          "status": record.get("status"),
                          "verdicts": [str(report.get("verdict", "")).upper()
                                       for report in (record.get("verdicts") or [])],
                          "findings": [{"verifier": finding.get("verifier", ""),
                                        "location": finding.get("location", ""),
                                        "detail": str(finding.get("detail", ""))[:400]}
                                       for finding in (record.get("blocking_findings") or [])],
                          "agreement": record.get("finding_agreement", {})}
                         for record in state.get("implementations", [])],
            "pending": [record["idea_id"] for record in pending],
            "plans_pending": [record["idea_id"] for record in pending
                              if record.get("status") == "repair_planned"],
            "blocked": [entry["idea_id"] for entry in state.get("blocked", [])],
            "agent_errors": [entry["idea_id"] for entry in state.get("agent_errors", [])]}


#: The phases of a round, in the order they run.
PHASES = ("ideation", "probe", "scoring", "implementation", "verification")


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
        for phase in (phase_ideation, phase_probe, phase_scoring, phase_implementation,
                      phase_verification):
            phase(root, round_id, log=log)
    finally:
        lock.release()
    return pipeline_module.summary(
        pipeline_module.load_state(pipeline_module.find_round_directory(root, round_id)))
