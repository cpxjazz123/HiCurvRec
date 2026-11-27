#!/usr/bin/env python3
"""Issue #274: inherit the hypothesis ledger and add this round's entries.

The ledger is never reset: the #273 entries are carried over unchanged and the #274 entries are
appended with their evidence paths and their current status.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "results/research/issue273/HYPOTHESES.json"
TARGET = REPO / "results/research/issue274/HYPOTHESES.json"

NEW = [
    {
        "id": "H274-1",
        "statement": "The representation degradation of the #269-#273 family is caused by regressing the "
                     "curvature *value* onto the encoder latent; a loss that uses the curvature only as an "
                     "offline ordering key and acts on the code geometry keeps the encoder's dispersion.",
        "status": "UNTESTED",
        "evidence": ["results/research/issue274/DIAGNOSIS.md",
                     "results/research/issue274/offline_diagnostics.json"],
        "origin_issue": 274,
        "opened_round": "round-1",
        "last_update": "2026-11-28",
        "reopen_reason": None,
    },
    {
        "id": "H274-2",
        "statement": "Inside one centre and one structure class the curvature's ordering carries information "
                     "beyond the closed/open label: it tracks the outer pair's common-neighbour count.",
        "status": "SUPPORTED",
        "evidence": ["results/research/issue274/offline_diagnostics.json",
                     "results/research/issue274/class_direction_map.json"],
        "origin_issue": 274,
        "opened_round": "round-1",
        "last_update": "2026-11-28",
        "reopen_reason": None,
    },
    {
        "id": "H274-3",
        "statement": "The raw graph curvature is comparable across the closed-triangle and the open-wedge "
                     "populations, so a cross-population curvature comparison could define the positive and "
                     "the negative role of a pair.",
        "status": "REFUTED",
        "evidence": ["results/research/issue274/offline_diagnostics.json",
                     "results/research/issue274/triplet_survey.json"],
        "origin_issue": 274,
        "opened_round": "round-1",
        "last_update": "2026-11-28",
        "reopen_reason": None,
    },
    {
        "id": "H274-4",
        "statement": "The two-pair ranking supervision improves the hold-out ranking accuracy and the "
                     "fixed-margin satisfaction rate over the frozen TIGER at the same caliber.",
        "status": "UNTESTED",
        "evidence": ["results/research/issue274/baselines.json"],
        "origin_issue": 274,
        "opened_round": "round-1",
        "last_update": "2026-11-28",
        "reopen_reason": None,
    },
    {
        "id": "H274-5",
        "statement": "The pair ranking does not degrade the SID capacity or the reconstruction, because it "
                     "acts on the code geometry and never on the encoder latent.",
        "status": "UNTESTED",
        "evidence": ["results/research/issue274/baselines.json"],
        "origin_issue": 274,
        "opened_round": "round-1",
        "last_update": "2026-11-28",
        "reopen_reason": None,
    },
    {
        "id": "H274-6",
        "statement": "The relative ordering of the outer pairs is already implied by the graph adjacency, "
                     "so the curvature adds nothing to the supervision's semantics.",
        "status": "INCONCLUSIVE",
        "evidence": ["results/research/issue274/offline_diagnostics.json"],
        "origin_issue": 274,
        "opened_round": "round-1",
        "last_update": "2026-11-28",
        "reopen_reason": "H274-2 shows an independent within-class signal, but the missing measurement is "
                          "whether the *selected extremal* representative is closer to the centre's other "
                          "items than a random representative of the same class; that comparison is not "
                          "yet computed.",
    },
]


def main() -> None:
    ledger = json.loads(SOURCE.read_text())
    ledger["issue"] = 274
    ledger["inherited_from"] = "results/research/issue273/HYPOTHESES.json"
    known = {entry["id"] for entry in ledger["hypotheses"]}
    for entry in NEW:
        if entry["id"] not in known:
            ledger["hypotheses"].append(entry)
    TARGET.write_text(json.dumps(ledger, indent=1, sort_keys=False) + "\n")
    counts = {}
    for entry in ledger["hypotheses"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print(json.dumps({"entries": len(ledger["hypotheses"]), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
