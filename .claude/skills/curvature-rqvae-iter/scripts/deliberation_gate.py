#!/usr/bin/env python3
"""2+1 deliberation gate for curvature-RQ-VAE iterations.

Run with NO CLI arguments from:
  stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/

Mode is inferred from canonical artifacts:
- PRE_STAGE2: always checks S00..S09.
- CLOSURE: if result-classification artifacts exist, additionally checks S10..S13.
- GLOBAL_REVIEW: if a global-review artifact exists, additionally checks S14.\n- ABORTED: stops at the first Judge C ABORT_ITERATION and validates the abort artifact.

The gate enforces process evidence and autonomous continuation. Scientific-decision rounds require independent A/B plus Judge; operational-repair rounds may use repair_record.md plus one repair verifier/Judge without redundant A/B regeneration. Judge artifacts must declare USER_INPUT_REQUIRED=NO and a concrete AUTONOMOUS_NEXT_ACTION. It does not substitute for scientific judgment or the mechanism-specific preflight/MVG.
"""
from __future__ import annotations

import re
from pathlib import Path


PRE_STAGE2 = [
    ("S00_SOURCE_TRUTH", ["source_snapshot_iter{n}.md"]),
    ("S01_PROTOCOL_LOCK", ["protocol_manifest_iter{n}.md"]),
    ("S02_HYPOTHESIS", ["hypothesis_iter{n}.md"]),
    ("S03_PROVENANCE", ["mechanism_manifest_iter{n}.md"]),
    ("S04_CONTRACT", ["mechanism_contract_iter{n}.json"]),
    ("S05_ONE_FACTOR", ["one_factor_diff_iter{n}.md"]),
    ("S06_IMPLEMENTATION", ["implementation_plan_iter{n}.md"]),
    ("S07_PREFLIGHT", ["preflight_contract_iter{n}.log"]),
    ("S08_MVG", ["mvg_check_iter{n}.log"]),
    ("S09_STAGE2_EXECUTION", ["stage2_execution_plan_iter{n}.md"]),
]

POST_STAGE2 = [
    ("S10_STAGE2_ANALYSIS", ["sid_geometry_iter{n}.md"]),
    (
        "S11_STAGE3_EVALUATION",
        ["stage3_evaluation_plan_iter{n}.md", "stage3_outcome_iter{n}.md"],
    ),
    (
        "S12_RESULT_CLASSIFICATION",
        ["failure_attribution_iter{n}.md", "gate_decision_iter{n}.md"],
    ),
    ("S13_GIT_CLOSURE", ["git_closure_iter{n}.md"]),
]

GLOBAL_REVIEW = [
    ("S14_GLOBAL_REVIEW", ["global_review_after_iter{n}.md"]),
]

ALLOWED_VERDICTS = {"ACCEPT_A", "ACCEPT_B", "MERGE_AB", "ABORT_ITERATION"}
INTERMEDIATE_REPAIR_VERDICT = "REPAIR_AND_RERUN"
REPAIR_PASS_VERDICT = "REPAIR_PASS"


def fail(message: str) -> None:
    raise RuntimeError(f"DELIBERATION_GATE_FAIL: {message}")


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def iter_id(work: Path) -> str:
    match = re.fullmatch(r"curvature_RQ-VAE_iter(\d+)", work.name)
    if not match:
        fail(f"run from curvature_RQ-VAE_iter<N>, got: {work}")
    return match.group(1)


def highest_round(stage_dir: Path) -> tuple[int, Path]:
    rounds: list[tuple[int, Path]] = []
    for child in stage_dir.iterdir() if stage_dir.is_dir() else []:
        match = re.fullmatch(r"round_(\d+)", child.name)
        if match and child.is_dir():
            rounds.append((int(match.group(1)), child))
    if not rounds:
        fail(f"no deliberation round found under {stage_dir}")
    rounds.sort(key=lambda item: item[0])
    number, path = rounds[-1]
    return number, path


def require_worker(path: Path, role: str, stage_id: str, source_packet: Path) -> None:
    text = read(path)
    required = [
        f"ROLE={role}",
        f"STAGE_ID={stage_id}",
        "INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.",
        "SOURCE_PACKET=",
    ]
    for marker in required:
        if marker not in text:
            fail(f"{path} missing required marker: {marker}")

    packet_name = source_packet.as_posix()
    declared = re.search(r"^SOURCE_PACKET=(.+)\s*$", text, re.M)
    if declared and declared.group(1).strip() not in {packet_name, str(source_packet)}:
        # Relative declarations are allowed if they end with the exact packet path.
        if not packet_name.endswith(declared.group(1).strip()):
            fail(f"{path} SOURCE_PACKET does not match stage packet")


def forbidden_action_is_positive(action: str, pattern: str) -> bool:
    """Return True only when a forbidden research direction is positively proposed.

    Mentions inside explicit negation/prohibition (e.g. 'do not use matched-seed
    replication' or 'without parameter sweep') are audit constraints, not proposals.
    """
    for match in re.finditer(pattern, action, re.I):
        prefix = action[max(0, match.start() - 64):match.start()].lower()
        if re.search(
            r"(?:do\s+not|don['’]?t|must\s+not|should\s+not|never|without|forbid(?:den)?|avoid)\b[^.;:]{0,48}$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False

def require_judge(
    path: Path,
    stage_id: str,
    canonical_names: list[str],
) -> tuple[str, str]:
    text = read(path)
    if f"STAGE_ID={stage_id}" not in text:
        fail(f"{path} has wrong/missing STAGE_ID")

    match = re.search(
        r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|REJECT_BOTH|REPAIR_AND_RERUN|ABORT_ITERATION)\s*$",
        text,
        re.M,
    )
    if not match:
        fail(f"{path} missing valid VERDICT")
    verdict = match.group(1)

    if verdict == INTERMEDIATE_REPAIR_VERDICT:
        if "SAME_ITERATION_REPAIR=AUTHORIZED" not in text:
            fail(f"{path} REPAIR_AND_RERUN requires SAME_ITERATION_REPAIR=AUTHORIZED")
        fail(
            f"{stage_id} latest round requests same-iteration repair/rerun; "
            "complete the repaired verification round before passing the gate"
        )
    if verdict not in ALLOWED_VERDICTS:
        fail(f"{stage_id} final verdict is {verdict}; stage is blocked")

    required = [
        "HARD_GATE_A=",
        "HARD_GATE_B=",
        "CANONICAL_DECISION=",
        "CANONICAL_ARTIFACT=",
        "CONFIDENCE=",
        "USER_INPUT_REQUIRED=NO",
        "ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM",
        "SWEEP_OR_REPLICATION_ITERATION=NO",
        "ROOT_CAUSE_ITERATION=NO",
        "AUTONOMOUS_NEXT_ACTION=",
    ]
    for marker in required:
        if marker not in text:
            fail(f"{path} missing judge field: {marker}")

    action = re.search(r"^AUTONOMOUS_NEXT_ACTION=(.+)\s*$", text, re.M)
    if not action or not action.group(1).strip():
        fail(f"{path} requires a non-empty AUTONOMOUS_NEXT_ACTION")

    forbidden_user_decision_patterns = [
        r"ask\s+the\s+user",
        r"wait\s+for\s+(the\s+)?user",
        r"need\s+user\s+decision",
        r"pause\s+for\s+direction",
        r"which\s+option\s+do\s+you\s+want",
        r"should\s+i\s+continue",
        r"do\s+you\s+want\s+me\s+to",
        r"what\s+should\s+the\s+next\s+iteration",
    ]
    for pattern in forbidden_user_decision_patterns:
        if re.search(pattern, text, re.I):
            fail(f"{path} contains forbidden user-decision escalation: {pattern}")

    autonomous_action = action.group(1).strip()
    forbidden_iteration_patterns = [
        r"\bparameter\s+sweep\b",
        r"\bgrid\s+search\b",
        r"\brandom\s+search\b",
        r"\bbayesian\s+(optimization|search)\b",
        r"\bmulti[- ]seed\b",
        r"\bmatched[- ]seed\b",
        r"\bseed\s+replication\b",
        r"\breplicat(e|ion)\b.*\bseed",
        r"\bnoise\s+estimation\b",
        r"\bvariance\s+estimation\b",
        r"\bsensitivity\s+(study|analysis|test)\b",
        r"\bablation[- ]only\b",
        r"\broot[- ]cause\b",
        r"\bwhy\s+.*(failed|worked|improved|dropped)\b",
        r"\bmicro[- ]delta\b",
    ]
    for pattern in forbidden_iteration_patterns:
        if forbidden_action_is_positive(autonomous_action, pattern):
            fail(
                f"{path} AUTONOMOUS_NEXT_ACTION proposes forbidden iteration type: {pattern}"
            )

    if verdict == "ACCEPT_A" and "WHY_NOT_B=" not in text:
        fail(f"{path} ACCEPT_A requires WHY_NOT_B")
    if verdict == "ACCEPT_B" and "WHY_NOT_A=" not in text:
        fail(f"{path} ACCEPT_B requires WHY_NOT_A")
    if verdict == "MERGE_AB":
        if "MERGE_COMPONENTS_A=" not in text or "MERGE_COMPONENTS_B=" not in text:
            fail(f"{path} MERGE_AB requires MERGE_COMPONENTS_A/B")
    if verdict == "ABORT_ITERATION":
        for marker in ("ABORT_REASON=", "ABORT_EVIDENCE=", "NEXT_ITERATION_CONSTRAINTS="):
            if marker not in text:
                fail(f"{path} ABORT_ITERATION requires {marker}")

    canonical_decl = re.search(r"^CANONICAL_ARTIFACT=(.+)\s*$", text, re.M)
    if canonical_decl:
        declared = canonical_decl.group(1)
        for name in canonical_names:
            if name not in declared:
                fail(f"{path} does not declare canonical artifact {name}")

    return verdict, text


def require_repair_record(path: Path, stage_id: str) -> str:
    text = read(path)
    required = [
        "ROUND_TYPE=OPERATIONAL_REPAIR",
        "LOCKED_SCIENCE_CHANGED=NO",
    ]
    for marker in required:
        if marker not in text:
            fail(f"{path} missing repair marker: {marker}")

    if "PARALLEL_EXECUTION=YES" not in text:
        reason = re.search(r"^SERIALIZATION_REASON=(.+)\s*$", text, re.M)
        if not reason or not reason.group(1).strip():
            fail(
                f"{path} must declare PARALLEL_EXECUTION=YES or a concrete SERIALIZATION_REASON"
            )

    if "CHECKS_PASS=YES" not in text:
        fail(f"{path} repair-only round has not established CHECKS_PASS=YES")

    return text


def require_repair_judge(
    path: Path, stage_id: str, canonical_names: list[str]
) -> tuple[str, str]:
    text = read(path)
    if f"STAGE_ID={stage_id}" not in text:
        fail(f"{path} has wrong/missing STAGE_ID")
    if "ROUND_TYPE=OPERATIONAL_REPAIR" not in text:
        fail(f"{path} repair judge missing ROUND_TYPE=OPERATIONAL_REPAIR")

    match = re.search(
        r"^VERDICT=(REPAIR_PASS|REPAIR_AND_RERUN|ABORT_ITERATION)\s*$",
        text,
        re.M,
    )
    if not match:
        fail(f"{path} repair judge missing valid repair verdict")
    verdict = match.group(1)

    required = [
        "LOCKED_SCIENCE_CHANGED=NO",
        "CANONICAL_DECISION=",
        "CANONICAL_ARTIFACT=",
        "USER_INPUT_REQUIRED=NO",
        "AUTONOMOUS_NEXT_ACTION=",
    ]
    for marker in required:
        if marker not in text:
            fail(f"{path} missing repair-judge field: {marker}")

    if verdict == INTERMEDIATE_REPAIR_VERDICT:
        if "SAME_ITERATION_REPAIR=AUTHORIZED" not in text:
            fail(f"{path} REPAIR_AND_RERUN requires SAME_ITERATION_REPAIR=AUTHORIZED")
        fail(
            f"{stage_id} latest operational-repair round still requires repair/rerun"
        )

    if verdict == "ABORT_ITERATION":
        for marker in (
            "ABORT_REASON=", "ABORT_EVIDENCE=", "NEXT_ITERATION_CONSTRAINTS="
        ):
            if marker not in text:
                fail(f"{path} ABORT_ITERATION requires {marker}")

    canonical_decl = re.search(r"^CANONICAL_ARTIFACT=(.+)\s*$", text, re.M)
    if canonical_decl:
        declared = canonical_decl.group(1)
        for name in canonical_names:
            if name not in declared:
                fail(f"{path} does not declare canonical artifact {name}")

    return verdict, text

def check_stage(
    logs: Path,
    deliberation_root: Path,
    n: str,
    stage_id: str,
    canonical_templates: list[str],
) -> tuple[str, int, str, list[str]]:
    stage_dir = deliberation_root / stage_id
    round_number, round_dir = highest_round(stage_dir)

    source_packet = round_dir / "source_packet.md"
    agent_a = round_dir / "agent_a.md"
    agent_b = round_dir / "agent_b.md"
    repair_record = round_dir / "repair_record.md"
    judge = round_dir / "judge.md"

    read(source_packet)

    canonical_paths = [logs / template.format(n=n) for template in canonical_templates]
    canonical_names = [path.name for path in canonical_paths]

    # Operational-repair rounds intentionally bypass redundant A/B regeneration.
    if repair_record.is_file():
        require_repair_record(repair_record, stage_id)
        verdict, _ = require_repair_judge(judge, stage_id, canonical_names)
        if verdict == REPAIR_PASS_VERDICT:
            for canonical in canonical_paths:
                if not canonical.is_file():
                    fail(
                        f"{stage_id} repair passed but canonical artifact missing: {canonical}"
                    )
            return stage_id, round_number, verdict, canonical_names
        if verdict == "ABORT_ITERATION":
            abort_path = logs / f"iteration_abort_iter{n}.md"
            read(abort_path)
            return stage_id, round_number, verdict, [abort_path.name]
        fail(f"{stage_id} unresolved operational repair")

    # Normal scientific-decision round: independent A/B evidence is required.
    require_worker(agent_a, "AGENT_A", stage_id, source_packet)
    require_worker(agent_b, "AGENT_B", stage_id, source_packet)

    # Parse verdict before enforcing the normal stage canonical artifact.
    judge_text = read(judge)
    match = re.search(
        r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|REJECT_BOTH|REPAIR_AND_RERUN|ABORT_ITERATION)\s*$",
        judge_text,
        re.M,
    )
    if not match:
        fail(f"{judge} missing valid VERDICT")
    preliminary_verdict = match.group(1)

    if preliminary_verdict == "ABORT_ITERATION":
        abort_path = logs / f"iteration_abort_iter{n}.md"
        verdict, _ = require_judge(judge, stage_id, [abort_path.name])
        abort_text = read(abort_path)
        required_abort = [
            "STATUS=ITERATION_ABORTED_INFEASIBLE",
            "ABORT_STAGE=",
            "ABORT_EVIDENCE=",
            "WHY_SAME_ITERATION_REPAIR_INVALID=",
            "NEXT_ITERATION_CONSTRAINTS=",
        ]
        for marker in required_abort:
            if marker not in abort_text:
                fail(f"{abort_path} missing abort field: {marker}")

        repairable_abort_patterns = [
            r"omitted.*batch.*id",
            r"missing.*batch.*id",
            r"omitted.*runtime.*device",
            r"missing.*runtime.*device",
            r"missing.*input.*hash",
            r"missing.*checkpoint.*hash",
            r"evidence[- ]capture",
            r"logging.*(missing|omitted)",
        ]
        if any(re.search(p, abort_text, re.I | re.S) for p in repairable_abort_patterns):
            scientific_infeasibility_patterns = [
                r"mechanism.*inactive",
                r"mathematical.*invalid",
                r"numerical.*invalid",
                r"requires changing.*(equation|constant|protocol|one-factor|parent)",
                r"cannot.*execute.*as specified",
            ]
            if not any(
                re.search(p, abort_text, re.I | re.S)
                for p in scientific_infeasibility_patterns
            ):
                fail(
                    f"{abort_path} appears to abort for a repairable evidence/logging "
                    "failure without direct mechanism infeasibility; use "
                    "REPAIR_AND_RERUN in the same iteration"
                )
        return stage_id, round_number, verdict, [abort_path.name]

    verdict, _ = require_judge(judge, stage_id, canonical_names)

    for canonical in canonical_paths:
        if not canonical.is_file():
            fail(f"{stage_id} judge completed but canonical artifact missing: {canonical}")

    return stage_id, round_number, verdict, canonical_names


def main() -> None:
    work = Path.cwd().resolve()
    n = iter_id(work)
    logs = work / "logs"
    deliberation_root = logs / "deliberation"

    stages = list(PRE_STAGE2)
    phase = "PRE_STAGE2"

    gate_decision = logs / f"gate_decision_iter{n}.md"
    failure_attribution = logs / f"failure_attribution_iter{n}.md"
    if gate_decision.is_file() or failure_attribution.is_file():
        stages.extend(POST_STAGE2)
        phase = "CLOSURE"

    global_review = logs / f"global_review_after_iter{n}.md"
    if global_review.is_file():
        if phase != "CLOSURE":
            fail("global review exists before closure/result-classification artifacts")
        stages.extend(GLOBAL_REVIEW)
        phase = "GLOBAL_REVIEW"

    summary = []
    for stage_id, templates in stages:
        result = check_stage(logs, deliberation_root, n, stage_id, templates)
        summary.append(result)
        if result[2] == "ABORT_ITERATION":
            print("DELIBERATION_ABORT_CONFIRMED")
            print(f"iter={n}")
            print("phase=ABORTED")
            print(f"abort_stage={stage_id}")
            for sid, round_number, verdict, canonical_names in summary:
                joined = ",".join(canonical_names)
                print(f"{sid}: round_{round_number} {verdict} -> {joined}")
            return

    print("DELIBERATION_GATE_PASS")
    print(f"iter={n}")
    print(f"phase={phase}")
    for stage_id, round_number, verdict, canonical_names in summary:
        joined = ",".join(canonical_names)
        print(
            f"{stage_id}: round_{round_number} {verdict} -> {joined}"
        )


if __name__ == "__main__":
    main()
