#!/usr/bin/env python3
"""2+1 deliberation gate for curvature-RQ-VAE iterations.

Run with NO CLI arguments from:
  stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/

Mode is inferred from canonical artifacts:
- PRE_STAGE2: always checks S00..S09.
- CLOSURE: if result-classification artifacts exist, additionally checks S10..S13.
- GLOBAL_REVIEW: if a global-review artifact exists, additionally checks S14.\n- ABORTED: stops at the first Judge C ABORT_ITERATION and validates the abort artifact.

The gate enforces process evidence. It does not substitute for scientific
judgment or the mechanism-specific preflight/MVG.
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
    if number > 2:
        fail(f"automatic deliberation round exceeds maximum 2: {path}")
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


def require_judge(
    path: Path,
    stage_id: str,
    canonical_names: list[str],
) -> tuple[str, str]:
    text = read(path)
    if f"STAGE_ID={stage_id}" not in text:
        fail(f"{path} has wrong/missing STAGE_ID")

    match = re.search(
        r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|REJECT_BOTH)\s*$",
        text,
        re.M,
    )
    if not match:
        fail(f"{path} missing valid VERDICT")
    verdict = match.group(1)

    if verdict not in ALLOWED_VERDICTS:
        fail(f"{stage_id} final verdict is {verdict}; stage is blocked")

    required = [
        "HARD_GATE_A=",
        "HARD_GATE_B=",
        "CANONICAL_DECISION=",
        "CANONICAL_ARTIFACT=",
        "CONFIDENCE=",
    ]
    for marker in required:
        if marker not in text:
            fail(f"{path} missing judge field: {marker}")

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
    judge = round_dir / "judge.md"

    read(source_packet)
    require_worker(agent_a, "AGENT_A", stage_id, source_packet)
    require_worker(agent_b, "AGENT_B", stage_id, source_packet)

    canonical_paths = [logs / template.format(n=n) for template in canonical_templates]
    canonical_names = [path.name for path in canonical_paths]

    # Parse verdict before enforcing the normal stage canonical artifact.
    judge_text = read(judge)
    match = re.search(
        r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|REJECT_BOTH|ABORT_ITERATION)\s*$",
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
