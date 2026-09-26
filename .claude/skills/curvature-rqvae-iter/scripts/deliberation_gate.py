#!/usr/bin/env python3
"""Pre-Stage2 2+1 deliberation gate.

Run with NO CLI arguments from:
  stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/

The gate verifies that every pre-Stage2 pipeline stage has:
- a same-source packet,
- independent Agent A and Agent B artifacts,
- a Judge C artifact with an acceptable verdict,
- the expected canonical artifact materialized.

It does not judge scientific quality itself; it enforces that the required
independent deliberation happened before Stage2 launches.
"""
from __future__ import annotations

import re
from pathlib import Path


STAGES = [
    ("S00_SOURCE_TRUTH", "source_snapshot_iter{n}.md"),
    ("S01_PROTOCOL_LOCK", "protocol_manifest_iter{n}.md"),
    ("S02_HYPOTHESIS", "hypothesis_iter{n}.md"),
    ("S03_PROVENANCE", "mechanism_manifest_iter{n}.md"),
    ("S04_CONTRACT", "mechanism_contract_iter{n}.json"),
    ("S05_ONE_FACTOR", "one_factor_diff_iter{n}.md"),
    ("S06_IMPLEMENTATION", "implementation_plan_iter{n}.md"),
    ("S07_PREFLIGHT", "preflight_contract_iter{n}.log"),
    ("S08_MVG", "mvg_check_iter{n}.log"),
    ("S09_STAGE2_EXECUTION", "stage2_execution_plan_iter{n}.md"),
]

ALLOWED_VERDICTS = {"ACCEPT_A", "ACCEPT_B", "MERGE_AB"}


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


def highest_round(stage_dir: Path) -> Path:
    rounds = []
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
    return path


def require_worker(path: Path, role: str, stage_id: str) -> None:
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


def require_judge(path: Path, stage_id: str) -> str:
    text = read(path)
    if f"STAGE_ID={stage_id}" not in text:
        fail(f"{path} has wrong/missing STAGE_ID")

    match = re.search(r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|REJECT_BOTH)\s*$", text, re.M)
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

    return verdict


def main() -> None:
    work = Path.cwd().resolve()
    n = iter_id(work)
    logs = work / "logs"
    deliberation_root = logs / "deliberation"

    summary = []

    for stage_id, canonical_template in STAGES:
        stage_dir = deliberation_root / stage_id
        round_dir = highest_round(stage_dir)

        source_packet = round_dir / "source_packet.md"
        agent_a = round_dir / "agent_a.md"
        agent_b = round_dir / "agent_b.md"
        judge = round_dir / "judge.md"

        read(source_packet)
        require_worker(agent_a, "AGENT_A", stage_id)
        require_worker(agent_b, "AGENT_B", stage_id)
        verdict = require_judge(judge, stage_id)

        canonical = logs / canonical_template.format(n=n)
        if not canonical.is_file():
            fail(f"{stage_id} judge completed but canonical artifact missing: {canonical}")

        summary.append((stage_id, round_dir.name, verdict, canonical.name))

    print("DELIBERATION_GATE_PASS")
    print(f"iter={n}")
    for stage_id, round_name, verdict, canonical in summary:
        print(f"{stage_id}: {round_name} {verdict} -> {canonical}")


if __name__ == "__main__":
    main()
