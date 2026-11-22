#!/usr/bin/env python3
"""C22: how often do the three structures ask the same item pair to move opposite ways?

The issue asks for this *before* a mechanism is chosen, because the answer decides whether a
single supervision term can satisfy all three structures at once. The audit is read-only: it
takes the frozen supervision banks and reports, per unordered pair,

* the number of instances that want it **near** (the adjacent or sibling position) and the
  number that want it **far** (the endpoint or parent position);
* the pairs where both counts are non-zero — those cannot be satisfied by one prefix geometry
  without one structure paying;
* the same table for pairs shared *across* structures, which are the ones a single shared unit
  of gradient cannot resolve.

Run it directly::

    python3 -m analysis_sid_recall.graphlet_conflicts --banks results/research/issue230/banks
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KINDS = ("chain", "fork", "merge")
#: ``near`` positions and ``far`` positions per kind, over the role tuple (A, B, C).
NEAR = {"chain": ((0, 1), (1, 2)), "fork": ((1, 2),), "merge": ((1, 2),)}
FAR = {"chain": ((0, 2),), "fork": ((0, 1), (0, 2)), "merge": ((0, 1), (0, 2))}


def unordered(a: int, b: int) -> tuple:
    return (a, b) if a <= b else (b, a)


def tally(bank: np.ndarray, kind: str) -> tuple[dict, dict]:
    """Count, per unordered pair, how many instances want it near and how many want it far."""
    near, far = defaultdict(int), defaultdict(int)
    for row in bank.tolist():
        for i, j in NEAR[kind]:
            near[unordered(row[i], row[j])] += 1
        for i, j in FAR[kind]:
            far[unordered(row[i], row[j])] += 1
    return near, far


def report(banks: Path, full: bool = False) -> dict:
    """Intra- and inter-structure conflict table.

    Two different questions, both the issue's:

    * **within a structure** (C22): does one item pair get pushed both near and far by the
      instances of that structure? A pair that is an endpoint of one chain and an adjacent
      pair of another cannot be satisfied by one prefix geometry -- those are the
      unsatisfiable constraints the issue asks to measure.
    * **across structures**: does the same pair serve as near evidence for one structure and
      far evidence for another? A single shared supervision unit cannot serve both.

    ``full=False`` reads the frozen supervision banks (what the loss sees); ``full=True``
    reads the whole sampled bank, the population the question is about.
    """
    near_pair = {}
    far_pair = {}
    per_kind = {}
    for kind in KINDS:
        name = f"bank_{kind}.npy" if full else f"{kind}_supervision.npy"
        path = banks / name
        if not path.is_file():
            per_kind[kind] = {"missing": str(path)}
            continue
        near, far = tally(np.load(path), kind)
        both = set(near) & set(far)
        per_kind[kind] = {
            "instances": int(len(np.load(path))),
            "pairs_near": len(near),
            "pairs_far": len(far),
            "pairs_both": len(both),
            "both_share": float(len(both) / max(1, len(set(near) | set(far)))),
            "both_examples": sorted(both)[:5],
            "both_near_mass": int(sum(near[p] for p in both)),
            "both_far_mass": int(sum(far[p] for p in both)),
        }
        near_pair[kind] = near
        far_pair[kind] = far
    across = {}
    kinds = [k for k in KINDS if k in near_pair]
    for i, left in enumerate(kinds):
        for right in kinds:
            if i == 0 and left == right:
                continue
            if left == right:
                continue
            # near evidence for `left` against far evidence for `right`, both directions
            clash = len(set(near_pair[left]) & set(far_pair[right]))
            if clash or left < right:
                across[f"near({left})∩far({right})"] = clash
                across[f"near({right})∩far({left})"] = len(set(near_pair[right]) & set(far_pair[left]))
    return {"banks": str(banks), "full": full, "kinds": per_kind,
            "contested_across": across}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--full", action="store_true",
                        help="audit the whole sampled bank, not only the supervision split")
    args = parser.parse_args()
    result = report(args.banks, full=args.full)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
