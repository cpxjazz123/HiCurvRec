"""Build the cross-track decision table for issue #86 and pick at most one R7-01 candidate.

The table is assembled from what each track's own report says, not from what a track hoped. A
track that is blocked or inconclusive contributes "no evidence", and a candidate can only be
promoted when its own track resolved it AND its causal track (A) established parity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
RESULTS = REPO_ROOT / "results/stage2_RQ-VAE"
OUTPUT = RESULTS / "r7_synthesis"

TRACKS = {
    "A": {"issue": 80, "dir": "r7_a_parity"},
    "B": {"issue": 81, "dir": "r7_b_gradients"},
    "C": {"issue": 82, "dir": "r7_c_geometry"},
    "D": {"issue": 83, "dir": "r7_d_backtest"},
    "E": {"issue": 84, "dir": "r7_e_voronoi"},
    "F": {"issue": 85, "dir": "r7_f_separation"},
}


def read(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def track_status(letter: str) -> dict[str, Any]:
    directory = RESULTS / TRACKS[letter]["dir"]
    if not directory.is_dir():
        return {"status": "blocked", "why": f"{directory} does not exist: the track produced no "
                                           f"report, so it has no evidence to contribute"}
    summary = read(directory / "report.json")
    if summary is None:
        return {"status": "blocked", "why": f"{directory}/report.json is missing or unreadable"}
    statuses = [arm.get("status") for arm in summary.get("arms", []) if isinstance(arm, dict)]
    if summary.get("status"):
        statuses.append(summary["status"])
    if not statuses:
        return {"status": "blocked", "why": "no arm reported a status"}
    resolved = [value for value in statuses if value == "supported"]
    if len(resolved) == len(statuses):
        return {"status": "supported", "arms": len(statuses)}
    if resolved:
        return {"status": "inconclusive",
                "why": f"{len(resolved)}/{len(statuses)} arms resolved; the rest are inconclusive"}
    return {"status": "inconclusive", "why": f"statuses were {sorted(set(statuses))}"}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    statuses = {letter: track_status(letter) for letter in TRACKS}

    parity = statuses["A"].get("status") == "supported"
    noise = read(RESULTS / "r7_a_parity/noise_floor/report.json")

    rows = [
        {
            "track": "A", "issue": 80,
            "hypothesis": "an audited trajectory can be branched counterfactually without the "
                          "measurement perturbing it",
            "evidence": statuses["A"],
            "strongest_negative_control": "NO-DEEP at a zero-deep-weight step must reproduce FULL "
                                         "bitwise; two hook-free runs establish whether the "
                                         "environment is bit-reproducible at all",
            "cost": "4 arms x 2 captures + 3 branch replays per fork step",
            "blocker": None if parity else "control parity not established on every arm",
            "r7_01_candidate": None,
        },
        {
            "track": "B", "issue": 81,
            "hypothesis": "healthy arms differ from failed ones in applied gradient DIRECTION, not "
                          "in raw magnitude (the 314.7x ratio argues against magnitude)",
            "evidence": statuses["B"],
            "strongest_negative_control": "observer ON/OFF must reproduce every parameter digest",
            "cost": "reuses A's bundles; no extra capture",
            "blocker": None if statuses["B"].get("status") == "supported" else "no measurement",
            "r7_01_candidate": None,
        },
        {
            "track": "C", "issue": 82,
            "hypothesis": "early SID failure is driven by margin fragility, occupancy collapse or "
                          "codebook drift, and these are distinguishable",
            "evidence": statuses["C"],
            "strongest_negative_control": "instrumented and hook-free runs must agree",
            "cost": "8 snapshots per arm on the fixed train-item population",
            "blocker": None if statuses["C"].get("status") == "supported" else "no measurement",
            "r7_01_candidate": None,
        },
        {
            "track": "D", "issue": 83,
            "hypothesis": "a 160-step indicator separates healthy from failed arms, and a "
                          "threshold fixed on some arms predicts the rest",
            "evidence": statuses["D"],
            "strongest_negative_control": "evaluate only on arms whose labels were not used to "
                                         "fix the threshold",
            "cost": "6 arms, protocol pre-registered before any indicator was computed",
            "blocker": None if statuses["D"].get("status") == "supported" else "no measurement",
            "r7_01_candidate": None,
        },
        {
            "track": "E", "issue": 84,
            "hypothesis": "projecting the deep update onto L1 margin half-spaces preserves winner "
                          "identities where clipping and full-vector PCGrad do not",
            "evidence": statuses["E"],
            "strongest_negative_control": "recompute the TRUE margins after the real AdamW step; "
                                         "a linearisation that looks satisfied but still crosses "
                                         "must be counted",
            "cost": "one extra autograd per guarded item",
            "blocker": "AdamW is not linear in the gradient, so a gradient-side constraint is "
                       "not a guarantee on the parameter update",
            "r7_01_candidate": "TENTATIVE: only if the realised violation rate is low",
        },
        {
            "track": "F", "issue": 85,
            "hypothesis": "a train-only soft repulsion on the L1 posterior separates unrelated "
                          "items without touching the hard SID",
            "evidence": statuses["F"],
            "strongest_negative_control": "a uniform posterior scores well on the loss while "
                                         "collapsing hard diversity; entropy, effective codes "
                                         "and margins are monitored separately",
            "cost": "one softmax over the L1 codebook per step",
            "blocker": "a soft term is a relaxation and cannot promise distinct hard codes",
            "r7_01_candidate": "TENTATIVE: only if hard diversity holds while the loss falls",
        },
    ]

    candidates = [row for row in rows if row["r7_01_candidate"] and not row["blocker"]]
    promoted = candidates[0] if candidates and parity else None

    summary = {
        "program": "R7 parallel workstream matrix (issue #86)",
        "tracks": rows,
        "control_parity_established": parity,
        "noise_floor": {
            "environment_bit_reproducible": (noise or {}).get("runs_identical"),
            "measured_spread": (noise or {}).get("per_step"),
            "why_it_matters": "issue #79 compared an audited trajectory against an independent "
                              "probe as if two runs of the same code were identical. They are not. "
                              "A cross-run difference must be read against this floor before it "
                              "is called a mechanism effect.",
        } if noise else {"note": "the noise floor was not measured"},
        "r7_01_selection": {
            "promoted": promoted,
            "rule": "at most one full 3000-epoch candidate, and only if its own track resolved it "
                    "AND track A established control parity; otherwise the honest outcome is that "
                    "no candidate survived",
            "outcome": (promoted["track"] if promoted
                        else "no candidate survived: track A parity is the gate for any causal "
                             "claim, and neither E nor F has a resolved measurement yet"),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({letter: statuses[letter]["status"] for letter in sorted(statuses)}, indent=2))
    print("control_parity_established:", parity)
    print("selection:", summary["r7_01_selection"]["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
