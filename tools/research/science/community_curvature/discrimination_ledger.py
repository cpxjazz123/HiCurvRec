"""The discrimination ledger: every arm's level-1 behaviour discrimination, one yardstick.

The campaign found that the recall follows one quantity -- the level-1 code's
ratio of the held-out behaviour positives' share to the negatives' -- and that
the community arms and the behaviour-contrast arms both move it the wrong way,
by imposing a behaviour structure on the code space. The arm that reached
0.06374 moves it the right way and imposes neither, which means one of its
*other* terms is responsible, and its terms are all in the repository already.

This script measures every arm whose SID is on disk with the same yardstick, on
the same independent sample the quality gate uses, and reports the three numbers
that separate the mechanisms: the positive share, the negative share and their
ratio, plus the level-2 prefix count and the Stage-3 recall when it exists.

No training is involved; it reads artifacts.

Run:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/discrimination_ledger.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.science.community_curvature import data  # noqa: E402

REPO = data.REPO
SID_ROOTS = [
    REPO / "results" / "stage2_RQ-VAE",
    REPO / "results" / "stage2_RQ-VAE" / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE",
]
PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"
STAGE3 = REPO / "results" / "stage3_T5Train"
OUT = REPO / "results" / "research" / "issue121" / "community_curvature"


import numpy as np


def load_pairs(path: Path):
    payload = np.load(path, allow_pickle=True)
    return (np.asarray(payload["anchors"], dtype=np.int64),
            np.asarray(payload["positives"], dtype=np.int64),
            np.asarray(payload["negatives"], dtype=np.int64))


def shares(codes: np.ndarray, anchors: np.ndarray, positives: np.ndarray,
           negatives: np.ndarray) -> dict:
    keep_pos = ((anchors < codes.size) & (positives < codes.size) & (anchors >= 0)
                & (positives >= 0) & (anchors != positives))
    positive = float((codes[anchors[keep_pos]] == codes[positives[keep_pos]]).mean())
    negative_column = negatives[:, 0]
    keep_neg = ((anchors < codes.size) & (negative_column < codes.size) & (anchors >= 0)
                & (negative_column >= 0) & (anchors != negative_column))
    negative = float((codes[anchors[keep_neg]] == codes[negative_column[keep_neg]]).mean())
    return {"positive": positive, "negative": negative,
            "ratio": (positive / negative) if negative else float("inf")}


def codes_of(sid_path: Path) -> tuple[np.ndarray, int, int]:
    table = json.loads(sid_path.read_text())
    values = [int(row[0]) for row in table.values()]
    ranking = {value: index for index, value in enumerate(sorted(set(values)))}
    codes = np.zeros(len(table), dtype=np.int64)
    for position in range(len(values)):
        codes[position] = ranking[values[position]]
    prefixes = len({tuple(row[:2]) for row in table.values()})
    triples = len({tuple(row[:3]) for row in table.values()})
    return codes, prefixes, triples


def recall_of(arm: str) -> float | None:
    found = sorted((STAGE3 / arm).rglob("test_final.json")) if (STAGE3 / arm).is_dir() else []
    if not found:
        return None
    return float(json.loads(found[-1].read_text())["test_recall@10"])


def main() -> int:
    anchors, positives, negatives = load_pairs(PAIRS)
    rows = []
    seen = set()
    for root in SID_ROOTS:
        if not root.is_dir():
            continue
        for sid in sorted(root.glob("*/item_sids.json")):
            arm = sid.parent.name
            if arm in seen:
                continue
            seen.add(arm)
            try:
                codes, prefixes, triples = codes_of(sid)
            except (ValueError, KeyError):
                continue
            if codes.size < anchors.max():
                continue
            shares_value = shares(codes, anchors, positives, negatives)
            rows.append({"arm": arm, "positive": shares_value["positive"],
                         "negative": shares_value["negative"],
                         "ratio": shares_value["ratio"],
                         "l2_prefixes": prefixes, "l3_triples": triples,
                         "recall@10": recall_of(arm)})
    # The coherence flag: the positives' share is the number the campaign found
    # the recall actually follows (the ratio's numerator). An arm that has spent
    # its behaviour coherence is visible here, without a GPU.
    baseline = next((row for row in rows if row["arm"] == "TIGER_RQ-VAE"), None)
    if baseline:
        reference = baseline["positive"]
        for row in rows:
            row["coherence_vs_baseline"] = row["positive"] / reference if reference else None
    rows.sort(key=lambda row: -row["ratio"])
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "discrimination_ledger.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "discrimination_ledger.json").write_text(json.dumps(rows, indent=2) + "\n",
                                                    encoding="utf-8")

    with_recall = [row for row in rows if row["recall@10"] is not None]
    print("%-46s %8s %8s %8s %8s %9s %8s"
          % ("arm", "pos", "neg", "ratio", "L2", "R@10", "coher."))
    for row in rows[:18]:
        print("%-46s %8.4f %8.4f %8.2f %8d %9s %8s"
              % (row["arm"][:46], row["positive"], row["negative"], row["ratio"],
                 row["l2_prefixes"],
                 ("%.5f" % row["recall@10"]) if row["recall@10"] else "-",
                 ("%.2f" % row["coherence_vs_baseline"])
                 if row.get("coherence_vs_baseline") else "-"))
    print("... %d arms measured, %d with a Stage 3 result" % (len(rows), len(with_recall)))
    print("written:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
