#!/usr/bin/env python3
"""Where can an L2 rule for forks actually act?

A fork hinge only decides something for a pair that already shares the first code: below
that, the second code is never compared.  This measures, per structure and per arm, how
often the "should be close" and "should be far" pair sets share L1 and, conditional on
sharing L1, how often they share L2.  The conditional share is the volume of decisions the
rule has; a tiny value explains a capped effect and points at conditioning the rule on the
L1-shared subset instead of spreading it over every pair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import sid_alignment_v2 as chains  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]


def codes_of(arm: str) -> np.ndarray:
    payload = json.loads((ROOT / "results" / "stage2_RQ-VAE" / arm / "item_sids.json").read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def pair_sets(rng, banks):
    out = {}
    for kind, bank in banks.items():
        a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
        if kind == "chain":
            out[kind] = {"near": np.concatenate([np.stack([a, b], 1), np.stack([b, c], 1)]),
                         "far": np.stack([a, c], 1)}
        elif kind == "fork":
            out[kind] = {"near": np.stack([b, c], 1),
                         "far": np.concatenate([np.stack([a, b], 1), np.stack([a, c], 1)])}
        else:
            out[kind] = {"near": np.stack([b, c], 1),
                         "far": np.concatenate([np.stack([b, a], 1), np.stack([c, a], 1)])}
    return out


def rates(codes: np.ndarray, pairs: np.ndarray) -> dict:
    left, right = pairs[:, 0], pairs[:, 1]
    share1 = codes[left, 0] == codes[right, 0]
    share2 = share1 & (codes[left, 1] == codes[right, 1])
    conditional = float(share2.sum() / share1.sum()) if share1.sum() else float("nan")
    return {"pairs": int(len(pairs)), "share_l1": float(share1.mean()),
            "share_l1l2": float(share2.mean()), "share_l2_given_l1": conditional}


def main() -> None:
    rng = np.random.default_rng(42)
    rows, cols, _ = chains.loaded_edges()
    out_map, in_map = chains.adjacency(rows, cols)
    edges = np.stack([rows, cols], axis=1)
    banks = {"chain": chains.chain3(out_map, edges, rng, chains.CAP),
             "fork": chains.fan(out_map, in_map, rng, chains.CAP, "fork"),
             "merge": chains.fan(out_map, in_map, rng, chains.CAP, "merge")}
    sets = pair_sets(rng, banks)
    arms = sys.argv[1:] or ["TIGER_RQ-VAE", "TIGER_chain_align_rq"]
    report = {}
    for arm in arms:
        codes = codes_of(arm)
        report[arm] = {kind: {side: rates(codes, pairs) for side, pairs in sides.items()}
                       for kind, sides in sets.items()}
        for kind, sides in report[arm].items():
            near, far = sides["near"], sides["far"]
            print(f"{arm:26s} {kind:6s} near: share_l1 {near['share_l1']:.4f} "
                  f"share_l2|share_l1 {near['share_l2_given_l1']:.4f} ({near['pairs']} pairs) | "
                  f"far: share_l1 {far['share_l1']:.4f} "
                  f"share_l2|share_l1 {far['share_l2_given_l1']:.4f} ({far['pairs']} pairs)")
    (HERE / "fork_decision_volume.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
