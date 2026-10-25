"""The round worker: the agent half of an Issue #120 round.

It runs the three ideation agents, the three scoring agents, and then, for every
selected idea, one implementation agent followed by three verification agents
and, when they reject, a dedicated repair agent. It never trains anything: the
Stage 2 and Stage 3 jobs are prepared here and executed by the MCP, which owns
the GPU boundary.
"""
from __future__ import annotations

import itertools
import json
import shutil
import time
from pathlib import Path

from .. import settings
from ..agents.runner import run_agent
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

IDEA_MIN = 1
IDEA_MAX = 3
SCORERS = 3
VERIFIERS = 1


def _workspace(root: Path, task_id: str) -> Path:
    return Path(root) / "agents" / task_id


def _run_agent(root: Path, task_id: str, prompt: str, output_name: str, parser,
               role: str = "verify") -> dict:
    """Run one agent in its own workspace and parse its plain-text answer.

    The agent writes a labelled text file, not JSON: a language model that
    drops one comma must not lose the whole answer. The parser turns the text
    into the record the round keeps.
    """
    # The runner gives the agent a workspace but not a destination; without this
    # the agent answers in prose and writes no file.
    destination = Path(root) / "agents" / task_id / output_name
    # Resumability: an answer that is already on disk is reused, so a round that
    # was interrupted does not pay for the same agents twice.
    if destination.is_file():
        return parser(destination.read_text(encoding="utf-8", errors="replace"))
    prompt = (prompt + "\n\nWrite the answer to " + str(destination) +
              " (this absolute path). Write nowhere else.")
    run_agent(task_id, prompt, output_name, Path(root),
              binary=settings.AGENT_BINARY, model=settings.AGENT_MODEL,
              timeout=settings.AGENT_TIMEOUT_SECONDS, role=role)
    # run_agent returns its execution receipt, not the answer: locate the file
    # the agent wrote, accepting the same bounded relocation the runner allows.
    workspace = Path(root) / "agents" / task_id
    answer = workspace / output_name
    if not answer.is_file():
        matches = [match for match in sorted(workspace.rglob(output_name))
                   if match.is_file()
                   and not any(part.startswith("attempt-")
                               for part in match.relative_to(workspace).parts)]
        if not matches:
            raise pipeline_module.RoundError("agent " + task_id + " wrote no " + output_name)
        answer = matches[0]
    return parser(answer.read_text(encoding="utf-8", errors="replace"))


def run_ideation(root: Path, objective: str) -> list:
    """Three independent ideation agents; each returns one to three ideas."""
    prompt_base = agent_contracts.IDEA_PROMPT.format(
        minimum=IDEA_MIN, maximum=IDEA_MAX,
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT)
    answers = []
    for index in range(1, pipeline_module.IDEAE_AGENTS + 1):
        task_id = "ideation_" + str(index)
        prompt = (prompt_base + "\n\nThe round objective: " + (objective or "improve the "
                     "item semantic IDs for a generative recommender") +
                     "\nYour answer must be a different set of ideas from the other two "
                     "agents, which you cannot see.")
        answers.append((task_id, _run_agent(root, task_id, prompt, "ideas.txt",
                                            plaintext.parse_ideas,
                                            role="ideate")))
    return answers


def _rubric_text() -> str:
    """The rubric as the reviewers read it: names, weights and anchored meanings."""
    rubric = scoring_module.RUBRIC
    lines = []
    for entry in rubric["dimensions"]:
        lines.append("  " + entry["name"] + " (" + str(int(entry["weight"])) + "%)")
        for anchor in sorted(entry["anchors"]):
            lines.append("    " + anchor + ": " + entry["anchors"][anchor])
    return "\n".join(lines)


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
    outcomes = {}
    for idea in ideas:
        identifier = idea["idea_id"]
        probe = probes_module.run_probe(idea, repo=repo)
        probe["sha256"] = _probe_digest(probe)
        literature = literature_module.run_literature(idea, cache_dir=cache_dir)
        literature["sha256"] = _probe_digest(literature)
        math_report = math_checks_module.run_math_checks(idea, repo=repo)
        math_report["sha256"] = _probe_digest(math_report)
        outcomes[identifier] = {"probe": probe, "literature": literature,
                                "math": math_report}
    return outcomes


def _probe_digest(payload: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def phase_probe(root: Path, round_id: str, *, log=print) -> dict:
    """Run the evidence probe of every proposal and freeze the bundles."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    if state["step"] not in ("probe", "scoring"):
        return {"phase": "probe", "ran": False, "step": state["step"],
                "reason": "the round is not in the probe phase"}
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
    reports = []
    for index in range(1, SCORERS + 1):
        task_id = "scoring_" + str(index)
        prompt = (agent_contracts.SCORE_PROMPT.format(
            objective=agent_contracts.RESEARCH_OBJECTIVE,
            scope=agent_contracts.SCOPE_CONTRACT,
            rubric_version=scoring_module.RUBRIC["version"],
            rubric=_rubric_text()) +
            "\n\nThe proposals, all of them:\n" + payload +
            "\n\nThe frozen evidence bundle:\n" + evidence_text)
        report = _review_with_retry(root, task_id, prompt)
        report.setdefault("scorer", task_id)
        report["rubric_version"] = scoring_module.RUBRIC["version"]
        report["rubric_sha256"] = scoring_module.RUBRIC["sha256"]
        reports.append(report)
    finalists = _finalist_ids(ideas, reports)
    comparisons = run_pairwise(root, ideas, finalists, evidence_text) if len(finalists) >= 2 else []
    return {"reports": reports, "bundles": bundle, "comparisons": comparisons,
            "finalists": finalists}


HARD_OBJECTION_TERMS = ("cancel", "identity", "leak", "dimension", "degenerat",
                        "counterexample", "equivalence", "invariant")


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
        outcome = run_scoring(agent_root, state["ideas"], repo=repo)
        return state, outcome["reports"], outcome
    return state, reports, outcome


def _review_with_retry(root: Path, task_id: str, prompt: str) -> dict:
    """One review, and one bounded retry if its schema is broken.

    The rubric schema is strict on purpose: a renamed, unknown or missing field
    is an error, not a partial average. A reviewer that breaks it is told what
    was wrong and asked once more, with a fresh workspace, so a stale answer can
    never be reused.
    """
    attempt = 0
    last_error = None
    while attempt < 2:
        suffix = "" if attempt == 0 else (
            "\n\nYour previous answer was rejected by the schema: " + str(last_error) +
            "\nAnswer again, with one complete block per proposal (SCORE with all seven "
            "dimensions, SCOPE, EVIDENCE, OBJECTION, RATIONALE).")
        name = "scores.txt" if attempt == 0 else "scores_retry.txt"
        run_id = task_id if attempt == 0 else task_id + "_retry1"
        try:
            report = _run_agent(root, run_id, prompt + suffix, name,
                                plaintext.parse_scores, role="score")
            scoring_module.validate_report(report, scorer=task_id)
            return report
        except (scoring_module.ScoreError, pipeline_module.RoundError) as exc:
            last_error = exc
            attempt += 1
    raise pipeline_module.RoundError("reviewer " + task_id + " broke the rubric schema "
                                     "twice: " + str(last_error))


def _finalist_ids(ideas: list, reports: list) -> list:
    """The top few proposals by weighted score, computed without mutating state."""
    try:
        matrix = scoring_module.build_matrix(ideas, reports)
    except scoring_module.ScoreError:
        return []
    rule = scoring_module.DEFAULT_SELECTION
    rows = [row for row in matrix["rows"] if row.get("weighted_score") is not None]
    rows.sort(key=lambda row: (-(row["weighted_score"] or 0.0), row["idea_id"]))
    return [row["idea_id"] for row in rows[:rule["pairwise_finalists"]]]


def run_pairwise(root: Path, ideas: list, finalists: list, evidence_text: str) -> list:
    """Order-balanced A-vs-B comparisons of the finalists, one per reviewer."""
    by_id = {idea["idea_id"]: idea for idea in ideas}
    pairs = []
    for index, (left, right) in enumerate(itertools.combinations(finalists, 2)):
        # Balance the printed order across the three reviewers.
        first, second = (left, right) if index % 2 == 0 else (right, left)
        pairs.append((first, second))
    comparisons = []
    for index in range(1, SCORERS + 1):
        task_id = "pairwise_" + str(index)
        blocks = []
        for position, (first, second) in enumerate(pairs):
            prompt = agent_contracts.PAIRWISE_PROMPT.format(
                rubric_version=scoring_module.RUBRIC["version"],
                rubric=_rubric_text(),
                first=plaintext.render_idea(by_id[first]),
                second=plaintext.render_idea(by_id[second]),
                evidence=evidence_text)
            report = _run_agent(root, task_id + "_" + str(position), prompt,
                                "pairwise.txt", plaintext.parse_pairwise, role="score")
            report["reviewer"] = task_id
            report["first"] = first
            report["second"] = second
            comparisons.append(report)
            blocks.append(report)
        if not blocks:
            continue
    return comparisons


def run_rebuttal(root: Path, idea: dict, objection: str, counterexample: str,
                 revision: int) -> dict:
    """One bounded repair of a proposal whose objection names a defect."""
    prompt = agent_contracts.REBUTTAL_PROMPT.format(objection=objection,
                                                    counterexample=counterexample or "none")
    return _run_agent(root, "rebuttal_" + idea["idea_id"] + "_r" + str(revision),
                      prompt, "rebuttal.txt", plaintext.parse_rebuttal, role="score")


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
        scope=agent_contracts.SCOPE_CONTRACT))
    report = _run_agent(root, "implementation_" + idea["idea_id"], prompt,
                        "implementation.txt", plaintext.parse_implementation,
                        role="implementation")
    report["provenance"] = provenance
    report["baseline"] = provenance["baseline"]
    report["baseline_digest"] = provenance["baseline_digest"]
    report["candidate_digest_before_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "PROPOSAL.json", idea)
    return report


def run_verification(root: Path, idea: dict, tree: Path, revision: int = 0) -> list:
    """Three independent verifiers of one implementation revision."""
    prompt = (agent_contracts.VERIFY_PROMPT.format(
        idea=plaintext.render_idea(idea), tree=str(tree),
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT))
    reports = []
    for index in range(1, VERIFIERS + 1):
        # The revision is part of the workspace name: a fresh review after a
        # repair must be a fresh agent, never a replay of the previous answer.
        task_id = "verify_" + idea["idea_id"] + "_r" + str(revision) + "_" + str(index)
        report = _run_agent(root, task_id, prompt, "verification.txt",
                            plaintext.parse_verification,
                            role="verify")
        report.setdefault("verifier", task_id)
        reports.append(report)
    return reports


def run_repair(root: Path, idea: dict, tree: Path, findings: list,
               revision: int = 0) -> dict:
    """The dedicated repair agent for one implementation revision."""
    prompt = (agent_contracts.REPAIR_PROMPT.format(
        findings=plaintext.render_findings(findings), tree=str(tree),
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT))
    return _run_agent(root, "repair_" + idea["idea_id"] + "_r" + str(revision),
                      prompt, "repair.txt", plaintext.parse_repair,
                      role="repair")


def phase_ideation(root: Path, round_id: str, *, log=print) -> dict:
    """Run the three ideation agents and merge what they proposed."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "ideation":
        return {"phase": "ideation", "ran": False, "step": state["step"],
                "ideas": len(state.get("ideas", [])),
                "reason": "the round is no longer in the ideation phase"}
    answers = run_ideation(agent_root, state.get("objective", ""))
    state = pipeline_module.apply_ideation(state, answers)
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
        # The record exists before the agent runs, so a failure of the
        # agent is recorded against the candidate instead of losing it.
        state.setdefault("implementations", []).append(
            pipeline_module.implementation_record(idea, {}, tree))
        pipeline_module.save_state(directory, state)
        try:
            report = run_implementation(agent_root, idea, tree)
        except baseline_module.BaselineError as exc:
            # The baseline is the one permitted source: if it is missing or
            # unreadable the candidate is not built from anything else.
            state = pipeline_module.mark_blocked(
                state, idea["idea_id"], "baseline unavailable: " + str(exc),
                kind=pipeline_module.BASELINE_UNAVAILABLE)
            pipeline_module.save_state(directory, state)
            continue
        except Exception as exc:                           # noqa: BLE001 - recorded, not raised
            state = pipeline_module.mark_agent_error(
                state, idea["idea_id"], "implementation failed: " + str(exc))
            pipeline_module.save_state(directory, state)
            continue
        record = [item for item in state["implementations"]
                  if item["idea_id"] == idea["idea_id"]][0]
        record.update(pipeline_module.implementation_record(idea, report, tree))
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
        if record.get("status") not in ("implemented", "repaired", "agent_error",
                                        "needs_repair"):
            continue
        idea = ideas_module.idea_index(state["ideas"])[record["idea_id"]]
        try:
            reports = run_verification(agent_root, idea, Path(record["tree"]),
                                       int(record.get("revision", 0)))
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
            if int(record.get("revision", 0)) >= pipeline_module.MAX_REVISIONS:
                # The candidate used up its repair rounds: it is recorded with
                # its findings instead of asking for another agent.
                state = pipeline_module.mark_revision_capped(
                    state, record["idea_id"],
                    "the verifiers still reject the implementation after " +
                    str(record.get("revision", 0)) + " repair rounds, which is the cap")
                pipeline_module.save_state(directory, state)
                log("%s: revision capped" % record["idea_id"])
                continue
            try:
                repair = run_repair(agent_root, idea, Path(record["tree"]),
                                    record.get("blocking_findings", []),
                                    int(record.get("revision", 0)))
                state = pipeline_module.apply_repair(state, record["idea_id"], repair)
                # The repair may have fixed the output paths: re-run the
                # preflight so a stale finding cannot block the candidate.
                running = set()
                job_file = directory / "jobs.json"
                if job_file.is_file():
                    import json as _json
                    jobs = _json.loads(job_file.read_text(encoding="utf-8"))
                    running = {job["idea_id"] for job in jobs.get("jobs", [])
                               if job.get("kind") == "stage2"
                               and job.get("status") == "running"}
                state = pipeline_module.recheck_output_paths(
                    state, record["idea_id"], skip_if_running=running)
            except Exception as exc:                       # noqa: BLE001
                state = pipeline_module.mark_agent_error(
                    state, record["idea_id"], "repair failed: " + str(exc))
            pipeline_module.save_state(directory, state)
            log("%s: %s" % (record["idea_id"], record.get("status")))
    # Stay in verification until every implementation is resolved: a repair
    # reopens the review, and the round must not move on with work in flight.
    pending = [record for record in state.get("implementations", [])
               if record.get("status") in ("implemented", "repaired", "needs_repair")
               or (record.get("status") == "agent_error"
                   and int(record.get("agent_error_attempts", 0))
                   <= pipeline_module.MAX_AGENT_RETRIES)]
    state["step"] = "verification" if pending else "stage2"
    pipeline_module.save_state(directory, state)
    return {"phase": "verification", "ran": True, "step": state["step"],
            "verified": len(pipeline_module.verified_implementations(state)),
            "pending": [record["idea_id"] for record in pending],
            "blocked": [entry["idea_id"] for entry in state.get("blocked", [])],
            "agent_errors": [entry["idea_id"] for entry in state.get("agent_errors", [])]}


#: The phases of a round, in the order they run.
PHASES = ("ideation", "probe", "scoring", "implementation", "verification")


def advance(root: Path, round_id: str, *, log=print) -> dict:
    """Run the agent stages of a round until it waits for Stage 2."""
    for phase in (phase_ideation, phase_probe, phase_scoring, phase_implementation,
                  phase_verification):
        phase(root, round_id, log=log)
    return pipeline_module.summary(
        pipeline_module.load_state(pipeline_module.find_round_directory(root, round_id)))
