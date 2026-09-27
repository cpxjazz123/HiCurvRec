#!/usr/bin/env python3
"""Main-only workflow gate for curvature-RQ-VAE iterations.

Main owns the iteration end-to-end. This checker validates canonical artifacts
and deterministic outputs only. Historical review directories are not
authorization inputs.
"""
from __future__ import annotations
import re
from pathlib import Path

PRE_STAGE2 = [
    "source_snapshot_iter{n}.md",
    "protocol_manifest_iter{n}.md",
    "hypothesis_iter{n}.md",
    "mechanism_manifest_iter{n}.md",
    "mechanism_contract_iter{n}.json",
    "one_factor_diff_iter{n}.md",
    "preflight_contract_iter{n}.log",
    "mvg_check_iter{n}.log",
]
POST_STAGE3 = [
    "sid_geometry_iter{n}.md",
    "stage3_protocol_gate_iter{n}.log",
    "stage3_outcome_iter{n}.md",
    "failure_attribution_iter{n}.md",
    "gate_decision_iter{n}.md",
    "git_closure_iter{n}.md",
]

def fail(message: str) -> None:
    raise RuntimeError(f"WORKFLOW_GATE_FAIL: {message}")

def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")

def iter_id(work: Path) -> str:
    match = re.fullmatch(r"curvature_RQ-VAE_iter(\d+)", work.name)
    if not match:
        fail(f"run from curvature_RQ-VAE_iter<N>, got: {work}")
    return match.group(1)

def require_files(logs: Path, n: str, templates: list[str]) -> None:
    for template in templates:
        read(logs / template.format(n=n))

def require_protocol_manifest(logs: Path, n: str) -> None:
    text = read(logs / f"protocol_manifest_iter{n}.md")
    fields = [
        "PROTOCOL_ID=",
        "STAGE3_TRAINER_PATH=",
        "STAGE3_TRAINER_SHA256=",
        "STAGE3_SEED=",
        "STAGE3_EPOCHS=",
        "STAGE3_EARLY_STOP=",
        "STAGE3_NO_EVAL=",
        "STAGE3_SKIP_TEST=",
        "STAGE3_BEAM_SIZE=",
        "STAGE3_CODE_PATH=",
        "STAGE3_LOG_PATH=",
        "STAGE3_SAVE_PATH=",
    ]
    missing = [field for field in fields if field not in text]
    if missing:
        completed_historical = (
            (logs / f"stage3_outcome_iter{n}.md").is_file()
            and (logs / f"gate_decision_iter{n}.md").is_file()
        )
        if not completed_historical:
            fail("protocol manifest missing machine fields: " + ", ".join(missing))

def require_pre_stage2(logs: Path, n: str) -> None:
    if "MECHANISM_CONTRACT_PASS" not in read(logs / f"preflight_contract_iter{n}.log"):
        fail("preflight missing MECHANISM_CONTRACT_PASS")
    if "MVG PASS" not in read(logs / f"mvg_check_iter{n}.log"):
        fail("MVG missing MVG PASS")

def require_result_record(logs: Path, n: str) -> None:
    failure = read(logs / f"failure_attribution_iter{n}.md").upper()
    gate = read(logs / f"gate_decision_iter{n}.md").upper()
    if "PROMOTION" not in gate:
        fail("gate decision missing promotion status")
    if not any(
        marker in failure
        for marker in ("ACTIVE_POSITIVE", "ACTIVE_NEUTRAL", "ACTIVE_NEGATIVE", "INVALID")
    ):
        fail("failure attribution missing result classification")

def require_abort(logs: Path, n: str) -> None:
    text = read(logs / f"iteration_abort_iter{n}.md")
    for marker in (
        "STATUS=ITERATION_ABORTED_INFEASIBLE",
        "ABORT_STAGE=",
        "ABORT_EVIDENCE=",
        "NEXT_ITERATION_CONSTRAINTS=",
    ):
        if marker not in text:
            fail(f"abort record missing {marker}")

def main() -> None:
    work = Path.cwd().resolve()
    n = iter_id(work)
    logs = work / "logs"

    if (logs / f"iteration_abort_iter{n}.md").is_file():
        require_abort(logs, n)
        print("WORKFLOW_ABORT_CONFIRMED")
        print(f"iter={n}")
        print("phase=ABORTED")
        print("owner=MAIN")
        return

    require_files(logs, n, PRE_STAGE2)
    require_protocol_manifest(logs, n)
    require_pre_stage2(logs, n)
    phase = "PRE_STAGE2"

    closure_started = any(
        (logs / template.format(n=n)).is_file()
        for template in (
            "stage3_outcome_iter{n}.md",
            "gate_decision_iter{n}.md",
            "git_closure_iter{n}.md",
        )
    )

    if closure_started:
        require_files(logs, n, POST_STAGE3)
        stage3_gate = read(logs / f"stage3_protocol_gate_iter{n}.log")
        if "STAGE3_PROTOCOL_PASS" not in stage3_gate:
            fail("Stage3 protocol gate did not pass")
        require_result_record(logs, n)
        phase = "CLOSURE"

    if (logs / f"global_review_after_iter{n}.md").is_file():
        if phase != "CLOSURE":
            fail("global review exists before closure")
        phase = "GLOBAL_REVIEW"

    print("WORKFLOW_GATE_PASS")
    print(f"iter={n}")
    print(f"phase={phase}")
    print("owner=MAIN")

if __name__ == "__main__":
    main()
