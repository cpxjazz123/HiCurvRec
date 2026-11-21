"""The artifact boundary: what a run exported, checked without trusting the mechanism.

Two adversaries in `results/research/qda/round11/adversarial_probe.py` survive the in-tree
guard, both by construction rather than by accident:

    bypass_indices      the mechanism overrides `forward` and never calls `_indices`, so the
                        floor has nothing to repair and the call-site check never runs
    encoder_collapse    every codeword is used, but only because every item's cost vector is
                        identical and the assignment is decided by index order

Neither can be caught inside the tree -- the tree is the mechanism's own code. Both are
visible in what the run *produces*: a level that uses one codeword, or a level whose code
distribution is a single spike. This module reads the exported identifier and says so.

The rule is the minimal non-degenerate one: **every level must use at least two codewords.**
That is not a quality threshold and not a Stage3 gate (CLAUDE.md 19/89 keep those where they
are); it is the definition of "this level carries no information at all".
"""
from __future__ import annotations

import json
from pathlib import Path

#: A level that uses fewer codewords than this carries no information: one codeword is a
#: constant, and the floor cannot make it informative because there is nothing to choose.
MIN_CODES_PER_LEVEL = 2

#: Stage3 refuses an identifier whose three-code collision reaches this (CLAUDE.md 89,
#: owner instruction 2026-11-21 raised it from 0.10 to 0.40).
#: It is the same limit `tools/research/stage2/sid_quality.py` enforces at launch; the two
#: are kept in step by `tests/test_sid_quality_gate.py::test_artifact_gate_matches_the_launch_gate`.
COLLISION_GATE = 0.4

#: The identifier is `CODEBOOK_NUM` codes followed by the collision-disambiguation column
#: (`train_rqvae.py::_extend_collisions`), which is not a quantiser level: auditing it would
#: fail a perfectly fine SID whose suffix happens to be constant.
CODEBOOK_NUM = 3


def audit_sid(path: Path | str, levels: int | None = CODEBOOK_NUM,
              enforce_gate: bool = False) -> dict:
    """Per-level usage of an exported identifier, and whether any level is degenerate."""
    path = Path(path)
    if not path.is_file():
        return {"status": "FAIL", "detail": "no identifier at " + str(path)}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        return {"status": "FAIL", "detail": "the identifier is not a non-empty mapping"}
    rows = [value for value in raw.values() if isinstance(value, list) and value]
    if not rows:
        return {"status": "FAIL", "detail": "the identifier holds no code lists"}
    available = min(len(row) for row in rows)
    width = levels or available
    width = max(1, min(int(width), available))

    report = {"items": len(rows), "levels": []}
    for level in range(width):
        counts: dict[int, int] = {}
        for row in rows:
            code = int(row[level])
            counts[code] = counts.get(code, 0) + 1
        distinct = len(counts)
        biggest = max(counts.values())
        report["levels"].append({"level": level + 1, "distinct": distinct,
                                 "max_share": round(biggest / len(rows), 6),
                                 "min_count": min(counts.values())})
    triples = {tuple(int(value) for value in row[:width]) for row in rows}
    report["distinct_sids"] = len(triples)
    report["collision"] = round(1.0 - len(triples) / len(rows), 6)
    # The same constant the Stage3 admission uses (CLAUDE.md 89). Reported here so the
    # number is visible at the earliest stage; the refusal itself stays at stage3_start.
    report["collision_gate"] = ("REFUSED" if report["collision"] >= COLLISION_GATE
                               else "allowed")

    if enforce_gate and report["collision"] >= COLLISION_GATE:
        report["status"] = "FAIL"
        report["detail"] = ("three-code collision " + str(report["collision"])
                            + " reaches the stage3 gate " + str(COLLISION_GATE))
        return report

    dead = [entry["level"] for entry in report["levels"]
            if entry["distinct"] < MIN_CODES_PER_LEVEL]
    if dead:
        report["status"] = "FAIL"
        report["detail"] = ("level(s) " + ", ".join(str(level) for level in dead)
                            + " use fewer than " + str(MIN_CODES_PER_LEVEL)
                            + " codewords across " + str(len(rows)) + " items")
    else:
        report["status"] = "PASS"
        report["detail"] = "levels " + "/".join(str(entry["distinct"])
                                                for entry in report["levels"])
    return report
