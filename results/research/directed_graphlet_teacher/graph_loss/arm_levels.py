#!/usr/bin/env python3
"""Per-level alignment of a finished arm.

The cumulative share S_k counts how many of the first k codes two items share, so the three
levels are read by differencing: L1 alone, the L1L2 prefix, and the full three-code identity.
Reporting them separately says where a rule actually moved the SID -- a rule that only shifts
the second code cannot claim an L1 effect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import sid_alignment_v2 as chains  # noqa: E402
import prototype_chain_long as proto  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]


def codes_of(arm: str) -> np.ndarray:
    payload = json.loads((ROOT / "results" / "stage2_RQ-VAE" / arm / "item_sids.json").read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def cumulative(codes, left, right, depth):
    shared = np.cumprod(codes[left] == codes[right], axis=1)
    return shared[:, :depth].sum(1)


def level_delta(codes, bank, kind, depth):
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    pair = lambda x, y: cumulative(codes, x, y, depth)
    if kind == "chain":
        near = (pair(a, b) + pair(b, c)) / 2.0
        far = pair(a, c)
    else:
        near = pair(b, c)
        far = (pair(a, b) + pair(a, c)) / 2.0
    return float(near.mean() - far.mean())


def main() -> None:
    rng = np.random.default_rng(42)
    rows, cols, _ = chains.loaded_edges()
    out_map, in_map = chains.adjacency(rows, cols)
    edges = np.stack([rows, cols], axis=1)
    banks = {"chain": chains.chain3(out_map, edges, rng, chains.CAP),
             "fork": chains.fan(out_map, in_map, rng, chains.CAP, "fork"),
             "merge": chains.fan(out_map, in_map, rng, chains.CAP, "merge")}
    # the same pair-disjoint evaluation banks the headline numbers use; measuring on the
    # training banks would not be comparable with arm_alignment_*.json
    banks = {kind: proto.split_pair_disjoint(bank)[1] for kind, bank in banks.items()}
    report = {}
    for arm in sys.argv[1:] or ["TIGER_RQ-VAE", "TIGER_chain_align_rq"]:
        codes = codes_of(arm)
        report[arm] = {kind: {f"k{depth}": level_delta(codes, bank, kind, depth)
                              for depth in (1, 2, 3)}
                       for kind, bank in banks.items()}
        for kind, levels in report[arm].items():
            print(f"{arm:26s} {kind:6s} " + " ".join(f"{k} {v:+.5f}" for k, v in levels.items())
                  + f"  (incremental L2 {levels['k2'] - levels['k1']:+.5f}, "
                    f"L3 {levels['k3'] - levels['k2']:+.5f})")
    (HERE / "arm_levels.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
