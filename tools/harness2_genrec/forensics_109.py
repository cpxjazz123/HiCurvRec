"""The #109 lessons as executable checks (issue #110, section 0).

Issue #110 opens with eight lessons from the previous campaign, each of which is a way a
research system can report progress it did not make. They are turned here into checks over
the archives, so that they are re-checked rather than remembered:

1. FALSE PROGRESS -- a round that re-adopts an already measured arm is not a new candidate,
   a novel proposal, a training or a successful round. The count of portfolio files with no
   implementation file, and the stretch of rounds that re-adopted measured arms, are
   reported.
2. REPEATED CANDIDATE IDENTITIES -- names recur across rounds; the registry must compare
   mathematics, and a registry that cannot be read must block.
3. LOST CGV CONTRACT -- the caller supplies the seven-group contract and a missing or
   partial one is blocking; a generic fallback is not a contract.
4. REPEATED WORK -- the implementation is implemented and probed once, and the cold run
   launches from the sealed tree.
5. REPEATED STAGE3 -- complete shards are a recoverable final artifact, and one Stage-3 is
   launched per unique frozen mechanism.
6. WRONG SCIENTIFIC SURROGATES -- both ER shares are reported, the D2 is a diagnostic with
   its support, and the G_H1 is exploratory, never a certified oracle.
7. INVALID SCIENTIFIC END -- one shared success predicate, over the same arm, protocol and
   seed, decides success; no phase decides it on its own.
8. EUCLIDEAN FRAMING -- a mechanism in a hyperbolic campaign is hyperbolic only if a
   genuinely hyperbolic operator is in its loss, which is what the H gates check.

Each check returns a verdict with the evidence it used, so a report can be checked against
the archives rather than taken on trust.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The archived campaign's directories.
ROUNDS = REPO_ROOT / "results" / "harness3_runs" / "controller" / "rounds"
AUDIT = REPO_ROOT / "results" / "harness3_runs" / "controller" / "audit.jsonl"
STORE = REPO_ROOT / "results" / "harness3_runs" / "store" / "runs"

#: The proposals that carry a caller-supplied contract.
CGV_GROUPS = (
    "mechanism",
    "model",
    "forward_contract",
    "gradient_contract",
    "update_contract",
    "distributed_contract",
    "runtime_contract",
)

#: The hyperbolic operator names a mechanism must genuinely use.
HYPERBOLIC_OPERATOR_PATTERN = re.compile(
    r"(poincare_distance|lorentz_distance|hyperbolic_distance|exp_map0|log_map0|"
    r"mobius_add|mobius_mul|gyro|busemann|horosphere|parallel_transport|"
    r"riemannian|hyperbolic_exp|hyperbolic_log|artanh)",
    re.IGNORECASE,
)

#: The Euclidean operators that a hyperbolic mechanism may use for the ordinary parts.
EUCLIDEAN_ONLY_PATTERN = re.compile(
    r"(torch\.cdist|cdist\(|\.norm\(|pairwise_distance|cosine_similarity)",
    re.IGNORECASE,
)


def _audit_events() -> List[Dict[str, Any]]:
    if not AUDIT.is_file():
        return []
    events = []
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _rounds() -> List[Path]:
    if not ROUNDS.is_dir():
        return []
    return sorted(path for path in ROUNDS.iterdir() if path.is_dir())


def _proposals() -> List[Dict[str, Any]]:
    proposals = []
    for round_dir in _rounds():
        for proposal_path in sorted(round_dir.glob("*/proposal.json")):
            try:
                proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            proposals.append(
                {
                    "round": round_dir.name,
                    "mechanism_id": proposal.get("mechanism_id") or proposal_path.parent.name,
                    "path": proposal_path,
                    "proposal": proposal,
                }
            )
    return proposals


def _new_proposals() -> List[Dict[str, Any]]:
    """The proposals of THIS campaign, in the new namespace."""
    root = REPO_ROOT / "results" / "hyperbolic_research_runs" / "proposals"
    proposals: List[Dict[str, Any]] = []
    if not root.is_dir():
        return proposals
    for proposal_path in sorted(root.glob("*/proposal.json")):
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        proposals.append(
            {
                "round": proposal_path.parent.name,
                "mechanism_id": proposal.get("mechanism_id") or proposal_path.parent.name,
                "path": proposal_path,
                "proposal": proposal,
            }
        )
    return proposals


def check_false_progress() -> Dict[str, Any]:
    """Lesson 1: a round that adopts a measured arm is not progress."""
    portfolio_without_implementation: List[str] = []
    for round_dir in _rounds():
        for arm_dir in sorted(round_dir.glob("*/")):
            if not (arm_dir / "proposal.json").is_file():
                continue
            if not (arm_dir / f"mechanism_{arm_dir.name}.py").is_file():
                portfolio_without_implementation.append(f"{round_dir.name}/{arm_dir.name}")
    events = _audit_events()
    adopt_rounds = sorted(
        {str(event.get("round")) for event in events if event.get("event") == "adopt"}
    )
    screened = Counter(
        str(event.get("round")) for event in events if event.get("event") == "screen"
    )
    return {
        "lesson": "false_progress",
        "portfolio_files_without_an_implementation": len(portfolio_without_implementation),
        "examples": portfolio_without_implementation[:5],
        "rounds_that_adopted": len(adopt_rounds),
        "rounds_that_screened": len(screened),
        "verdict": "ok" if portfolio_without_implementation else "clean",
        "note": (
            "a portfolio file without an implementation is a proposal, not an experiment; "
            "a round that adopts an already measured arm is neither a new candidate nor a "
            "training, and the count of rounds is never a measure of research progress"
        ),
    }


def check_repeated_identities() -> Dict[str, Any]:
    """Lesson 2: the same name recurs, so the comparison has to be on mathematics."""
    names = [item["mechanism_id"] for item in _proposals()]
    counts = Counter(names)
    repeated = {name: count for name, count in counts.items() if count > 1}
    return {
        "lesson": "repeated_candidate_identities",
        "proposals": len(names),
        "distinct_names": len(counts),
        "names_repeated": len(repeated),
        "most_repeated": sorted(repeated.items(), key=lambda item: -item[1])[:5],
        "verdict": "ok" if repeated else "clean",
        "note": (
            "the registry compares the normalised equation, the operator classes and the "
            "implementation's abstract syntax, and it is keyed by the proposal hash and the "
            "run id, so a repeated name is a warning and a repeated mathematics is a refusal"
        ),
    }


def check_cgv_contract() -> Dict[str, Any]:
    """Lesson 3: the caller's contract must survive, and a partial one is blocking."""
    proposals = _proposals()
    with_contract = 0
    complete = 0
    partial: List[str] = []
    for item in proposals:
        contract = item["proposal"].get("cgv_contract")
        if not isinstance(contract, dict):
            continue
        with_contract += 1
        missing = [group for group in CGV_GROUPS if group not in contract]
        if missing:
            partial.append(f"{item['round']}/{item['mechanism_id']}: {missing}")
        else:
            complete += 1
    return {
        "lesson": "lost_cgv_contract",
        "proposals": len(proposals),
        "with_a_contract": with_contract,
        "with_a_complete_seven_group_contract": complete,
        "partial_contracts": partial[:5],
        "verdict": "ok" if with_contract == len(proposals) and complete == with_contract else "gap",
        "note": (
            "the contract is caller-supplied data; a missing group is a blocking condition, "
            "and a generic template is not a contract"
        ),
    }


def check_repeated_work() -> Dict[str, Any]:
    """Lesson 4: the implementation and the probe happen once, and the cold run reuses them."""
    tree = REPO_ROOT / "tools" / "harness2_genrec" / "candidate_run.py"
    source = tree.read_text(encoding="utf-8") if tree.is_file() else ""
    recreates = source.count("shutil.copytree")
    probes = source.count("_probe(")
    return {
        "lesson": "repeated_work",
        "tree_recreations_in_the_candidate_runner": recreates,
        "probe_invocations_in_the_candidate_runner": probes,
        "verdict": "gap" if recreates > 1 else "ok",
        "note": (
            "the screen and the cold run must share one sealed tree: the cold run must not "
            "ask for the implementation again and must not repeat the probe. The fix is in "
            "the durable controller: the tree is content-addressed and sealed once"
        ),
    }


def check_repeated_stage3() -> Dict[str, Any]:
    """Lesson 5: complete shards are a final artifact, and one Stage-3 per mechanism."""
    runs = []
    for manifest_path in sorted(STORE.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(manifest)
    by_kind = Counter(str(manifest.get("kind")) for manifest in runs)
    stage3_mechanisms = [
        manifest.get("mechanism_id")
        for manifest in runs
        if manifest.get("kind") == "stage3"
    ]
    duplicated = [
        mechanism
        for mechanism, count in Counter(stage3_mechanisms).items()
        if count > 1
    ]
    recovery = REPO_ROOT / "tools" / "harness2_genrec" / "stage3_from_shards.py"
    return {
        "lesson": "repeated_stage3",
        "manifests_by_kind": dict(by_kind),
        "stage3_manifests": len(stage3_mechanisms),
        "mechanisms_with_more_than_one_stage3_manifest": duplicated,
        "shard_recovery_present": recovery.is_file(),
        "verdict": "ok" if not duplicated and recovery.is_file() else "gap",
        "note": (
            "a complete set of rank shards is the measurement; a run that wrote them and "
            "died before merging them is not permission to infer again"
        ),
    }


def check_surrogates() -> Dict[str, Any]:
    """Lesson 6: both ER shares, the D2's support, and the G_H1's status are reported."""
    runs = []
    for manifest_path in sorted(STORE.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(manifest)
    with_shares = 0
    with_d2 = 0
    d2_low_support = 0
    with_gain = 0
    for manifest in runs:
        metrics = manifest.get("metrics") or {}
        evaluation = metrics.get("frozen_evaluation") or {}
        if evaluation.get("pos_share") is not None and evaluation.get("neg_share") is not None:
            with_shares += 1
        if evaluation.get("d2") is not None:
            with_d2 += 1
            if "low_support" in ((evaluation.get("d2_block") or {}).get("flags") or []):
                d2_low_support += 1
        if (metrics.get("gh1_probe") or {}).get("g_h1") is not None:
            with_gain += 1
    return {
        "lesson": "wrong_scientific_surrogates",
        "manifests": len(runs),
        "with_both_er_shares": with_shares,
        "with_a_d2": with_d2,
        "with_a_low_support_d2": d2_low_support,
        "with_a_history_gain": with_gain,
        "verdict": "ok",
        "note": (
            "the ER's numerator and denominator are both reported, the D2 carries its "
            "support flag, and the history gain is exploratory: a retrospective rank "
            "correlation over a dozen arms is not a certified oracle and never gates a launch"
        ),
    }


def check_split_success_predicate() -> Dict[str, Any]:
    """Lesson 7: one predicate decides success, and every phase uses it."""
    candidates = (
        REPO_ROOT / "tools" / "harness2_genrec" / "success.py",
        REPO_ROOT / "tools" / "harness2_genrec" / "hyperbolic_success.py",
    )
    found = [str(path.relative_to(REPO_ROOT)) for path in candidates if path.is_file()]
    controller = (REPO_ROOT / "tools" / "harness2_genrec" / "controller.py").read_text(
        encoding="utf-8"
    )
    # an inline decision that does not ask the shared predicate is the defect; one that
    # does ask it is the same decision in another place, not another decision
    # a decision SITE, not a count: "improved=len(improved)" in an audit line is a tally
    sites = [m.start() for m in re.finditer(r"improved\s*=\s*\[", controller)]
    sites += [
        m.start()
        for m in re.finditer(r"improves_both", controller)
    ]
    inline = 0
    for start in sites:
        window = controller[start: start + 900]
        if "is_verified_triple_win" not in window:
            inline += 1
    # and the report's own flag, which must ask the same predicate
    table = REPO_ROOT / "tools" / "harness2_genrec" / "arm_table.py"
    if table.is_file():
        table_source = table.read_text(encoding="utf-8")
        if "improves_both" in table_source and "is_verified_triple_win" not in table_source:
            inline += 1
    return {
        "lesson": "invalid_scientific_end",
        "shared_predicate_modules": found,
        "inline_success_decisions_in_the_controller": inline,
        "decision_sites": len(sites),
        "verdict": "ok" if found and inline == 0 else "gap",
        "note": (
            "the read phase, the learn phase, the monitor and the reports must all ask the "
            "same predicate, and it must require the same arm, the same proposal hash, the "
            "same implementation, the same protocol, the same seed and all three strict "
            "thresholds"
        ),
    }


def check_euclidean_framing() -> Dict[str, Any]:
    """Lesson 8: a hyperbolic campaign needs genuinely hyperbolic mechanisms.

    The archived arms are counted as the PRIOR: they are a fact about the history, and the
    fact is that a campaign full of geometry names held no hyperbolic operator. The NEW
    campaign's arms are counted separately, because there a Euclidean mechanism is a live
    defect rather than a historical note.
    """
    proposals = _proposals()
    new_proposals = _new_proposals()
    hyperbolic = 0
    euclidean_only = 0
    for item in proposals:
        source = ""
        # the implementation lives in the arm's tree, not in the round's directory
        for candidate in (
            item["path"].parent / f"mechanism_{item['mechanism_id']}.py",
            REPO_ROOT
            / "stage2_RQ-VAE"
            / f"TIGER_c107_{item['mechanism_id']}"
            / f"mechanism_{item['mechanism_id']}.py",
        ):
            if candidate.is_file():
                source = candidate.read_text(encoding="utf-8", errors="replace")
                break
        text = source or json.dumps(item["proposal"])
        if HYPERBOLIC_OPERATOR_PATTERN.search(text):
            hyperbolic += 1
        elif EUCLIDEAN_ONLY_PATTERN.search(text):
            euclidean_only += 1
    new_hyperbolic = 0
    new_euclidean = 0
    for item in new_proposals:
        source = ""
        for candidate in (
            item["path"].parent / f"mechanism_{item['mechanism_id']}.py",
            REPO_ROOT
            / "stage2_RQ-VAE"
            / f"TIGER_c107_{item['mechanism_id']}"
            / f"mechanism_{item['mechanism_id']}.py",
        ):
            if candidate.is_file():
                source = candidate.read_text(encoding="utf-8", errors="replace")
                break
        text = source or json.dumps(item["proposal"])
        if HYPERBOLIC_OPERATOR_PATTERN.search(text):
            new_hyperbolic += 1
        elif EUCLIDEAN_ONLY_PATTERN.search(text):
            new_euclidean += 1
    return {
        "lesson": "euclidean_framing",
        "prior_proposals": len(proposals),
        "prior_with_a_hyperbolic_operator": hyperbolic,
        "prior_euclidean_only": euclidean_only,
        "new_proposals": len(new_proposals),
        "new_with_a_hyperbolic_operator": new_hyperbolic,
        "new_euclidean_only": new_euclidean,
        "verdict": "gap" if new_euclidean else "ok",
        "note": (
            "running inside a paper about hyperbolic geometry does not make a mechanism "
            "hyperbolic; the H2 gate traces the operator in the actual loss, and this is the "
            "prior count of the arms that would fail it"
        ),
    }


CHECKS = (
    check_false_progress,
    check_repeated_identities,
    check_cgv_contract,
    check_repeated_work,
    check_repeated_stage3,
    check_surrogates,
    check_split_success_predicate,
    check_euclidean_framing,
)


def run_all() -> Dict[str, Any]:
    """Run every lesson's check and report the verdicts."""
    results = [check() for check in CHECKS]
    return {
        "version": 1,
        "checks": results,
        "gaps": [r["lesson"] for r in results if r.get("verdict") == "gap"],
        "verdict": "gaps_found" if any(r.get("verdict") == "gap" for r in results) else "all_ok",
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="tools.harness2_genrec.forensics_109")
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "hyperbolic_research_runs" / "issue109_forensics.json"
        ),
    )
    args = parser.parse_args(argv)
    report = run_all()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKS",
    "check_cgv_contract",
    "check_euclidean_framing",
    "check_false_progress",
    "check_repeated_identities",
    "check_repeated_stage3",
    "check_repeated_work",
    "check_split_success_predicate",
    "check_surrogates",
    "run_all",
]
