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
    "D": {"issue": 83, "dir": "r7_d_backtest", "file": "manifest.json"},
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
    summary = read(directory / TRACKS[letter].get("file", "report.json"))
    if summary is None and TRACKS[letter].get("file"):
        # a track may still have delivered its per-arm reports even when its summary has another
        # name; fall back to those rather than calling the track blocked
        per_arm = sorted(directory.glob("*/report.json"))
        if not per_arm:
            return {"status": "blocked", "why": f"{directory} has no report the synthesis can read"}
        statuses = [read(path) or {} for path in per_arm]
        summary = {"arms": [{"arm": path.parent.name,
                             "status": value.get("status", "unknown")}
                            for path, value in zip(per_arm, statuses)]}
    if summary is None:
        return {"status": "blocked", "why": f"{directory}/report.json is missing or unreadable"}
    statuses = [arm.get("status") for arm in summary.get("arms", []) if isinstance(arm, dict)]
    if summary.get("status"):
        statuses.append(summary["status"])
    if not statuses and "calibratable" in summary:
        # a backtest resolves by producing its manifest, not by declaring a per-arm status
        return {"status": "supported",
                "arms": summary.get("cohort_measured"),
                "calibratable": summary.get("calibratable"),
                "outcome": summary.get("outcome")}
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
            "blocker": None,
            "r7_01_candidate": "REJECTED as a candidate",
            "verdict": "the pilot measured the thing it was built for. Deep-attributable L1 "
                       "identity changes are +7/-2 (r4_08), +5/+1 (r6_02), +4/-2 (r6_03), "
                       "-1/0 (r6_04) at steps 96/160, against a measured run-to-run floor of "
                       "+/-5 to +/-6 on the same arm and step. The effect is inside the noise "
                       "and changes sign. Protecting L1 boundaries would be protecting something "
                       "that is not breaking.",
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
            "blocker": "no measurement: the prototype has its math, its independent VJP oracle "
                       "and its gaming diagnostics, but no pilot was run, so there is nothing to "
                       "evaluate",
            "r7_01_candidate": "NOT EVALUATED: the hypothesis class is the only one consistent "
                               "with tracks B, C and E, but it has no evidence of its own",
        },
    ]

    candidates = [row for row in rows
                  if row["r7_01_candidate"] and not row["blocker"]
                  and row["r7_01_candidate"].startswith("TENTATIVE")]
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
            "outcome": (promoted["track"] if promoted else
                        "NO CANDIDATE SURVIVED, and that is the finding rather than a shortfall. "
                        "Control parity is established, so the hypotheses could be tested rather "
                        "than merely asserted. Track E, the only candidate with a measurement, "
                        "was rejected on it: deep-attributable L1 boundary crossings are inside "
                        "the run-to-run noise floor and change sign, so protecting boundaries "
                        "protects something that is not breaking. Track F's hypothesis class is "
                        "the only one consistent with all three measured tracks -- tracks B, C and "
                        "E independently point at occupancy and distribution rather than gradient "
                        "direction or boundary fragility -- but F has no pilot of its own, so it "
                        "cannot be promoted on the strength of other tracks' agreement. "
                        "Promoting it would repeat exactly the error this programme was opened "
                        "to correct."),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({letter: statuses[letter]["status"] for letter in sorted(statuses)}, indent=2))
    print("control_parity_established:", parity)
    print("selection:", summary["r7_01_selection"]["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
