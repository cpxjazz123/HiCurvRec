#!/usr/bin/env python3
"""Non-mutating revalidation of a persisted probe's math checks.

Issue #124 found that the shared probe wrote a single ``overall`` PASS for a
set of checks whose dependency evidence was weaker than the verdict implied: the
proxy assigned ``both_on == behavior_off`` and still passed, because the
verdict was an ``OR`` over two interventions. A positive row-scalar weight
cannot move an arg-min at all:

    argmin_k ((1 + w_i) * d(i, k)) == argmin_k d(i, k)   for w_i > 0

This tool never edits the historical artefacts. It reads them, hashes them,
re-runs the current checks on the same persisted proposal and writes a separate
``math_validation_audit.json`` next to the probe.

    python3 tools/research/rounds/revalidation.py \
        --round results/research/issue121/rounds/round_20261025_181918_856 \
        --idea successor_entropy_curvature
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The files whose hashes the audit back-references.
ARTEFACTS = ("probe.json", "math_checks.json", "evidence_bundle.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_idea(round_dir: Path, idea_id: str) -> dict:
    state = json.loads((round_dir / "state.json").read_text())
    for idea in state.get("ideas") or []:
        if isinstance(idea, dict) and idea.get("idea_id") == idea_id:
            return idea
    raise KeyError(f"{idea_id} is not in {round_dir / 'state.json'}")


def row_scalar_invariance(weights, distances) -> dict:
    """The row-scalar invariance, measured on the numbers it applies to.

    ``weights`` is per item and identical for every codeword of that item, so a
    positive factor scales a whole row of the score matrix and cannot move its
    arg-min.
    """
    import numpy as np

    weights = np.asarray(weights, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    scaled = distances * (1.0 + weights)[:, None]
    plain = distances.argmin(axis=-1)
    weighted = scaled.argmin(axis=-1)
    return {
        "row_scalar_multiplier": [float(value) for value in (1.0 + weights)[:8]],
        "plain_argmin": [int(value) for value in plain[:8]],
        "weighted_argmin": [int(value) for value in weighted[:8]],
        "assignments_identical": bool((plain == weighted).all()),
        "proof": ("argmin_k ((1 + w_i) * d(i, k)) == argmin_k d(i, k) for w_i > 0, so a "
                  "positive row-scalar behaviour weight cannot change an assignment; a "
                  "codeword-independent offset cannot either"),
    }


def revalidate(round_dir: Path, idea_id: str) -> dict:
    from tools.research.rounds.math_checks import run_math_checks

    probe_dir = round_dir / "probes" / idea_id
    original = {}
    for name in ARTEFACTS:
        path = probe_dir / name
        if path.exists():
            original[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    decision = round_dir / "decision.json"
    if decision.exists():
        original["decision.json"] = {"sha256": _sha256(decision),
                                     "bytes": decision.stat().st_size}
    idea = _load_idea(round_dir, idea_id)
    original_checks = json.loads((probe_dir / "math_checks.json").read_text())
    current = run_math_checks(idea, repo=REPO)
    return {
        "schema": 1,
        "kind": "math_validation_audit",
        "idea_id": idea_id,
        "round": str(round_dir.relative_to(REPO)),
        "non_mutating": True,
        "original_artifacts": original,
        "original_verdict": {
            "status": original_checks.get("status"),
            "schema": original_checks.get("schema"),
            "what_it_established": (
                "the operator family is internally consistent (the maps invert each "
                "other, the curvature is negative, the domain guard holds and the "
                "distance is not Euclidean). It says NOTHING about whether the "
                "candidate's own path is sensitive to the behaviour term."),
            "flaw": (
                "the proxy check compared the interventions with an OR: the "
                "behaviour-off assignment was identical to both-on "
                f"{original_checks['checks'][-1].get('detail', '')}, which a positive "
                "row-scalar weight provably cannot change, and the OR was satisfied "
                "by the geometry-off branch alone."),
        },
        "current_checks": current,
        "row_scalar_invariance": row_scalar_invariance(
            [0.5, 0.25, 0.125, 0.75], [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1],
                                      [0.9, 0.8, 0.7], [0.05, 0.4, 0.2]]),
        "candidate_path": {
            "status": "CANDIDATE_PATH_NOT_RUN",
            "reason": ("this audit revalidates the persisted operator evidence; the "
                       "candidate's own forward path is exercised by the candidate "
                       "admission check, not by the probe"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, help="round directory")
    parser.add_argument("--idea", required=True, help="idea id")
    args = parser.parse_args()
    round_dir = Path(args.round)
    if not round_dir.is_absolute():
        round_dir = REPO / round_dir
    audit = revalidate(round_dir, args.idea)
    out = round_dir / "probes" / args.idea / "math_validation_audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out.relative_to(REPO)),
                      "original_verdict": audit["original_verdict"]["status"],
                      "current_status": audit["current_checks"].get("status"),
                      "row_scalar_invariance": audit["row_scalar_invariance"]["assignments_identical"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
