#!/usr/bin/env python3
"""Re-measure finished arms on several evaluation banks.

The special-triple banks are built from a seeded RNG, and the alignment metric is a mean over
those triples, so the bank itself is a source of variance that no amount of triple resampling
can see.  This rebuilds the banks from five seeds and reports, per structure, each arm's level
and the paired arm-minus-baseline difference, so the spread across banks is explicit.
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
ARMS = ("TIGER_RQ-VAE", "TIGER_chain_align_rq", "TIGER_graphlet_align_rq")
SEEDS = (1, 2, 3, 4, 5)


def codes_of(arm: str) -> np.ndarray:
    payload = json.loads((ROOT / "results" / "stage2_RQ-VAE" / arm / "item_sids.json").read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def delta(codes: np.ndarray, bank: np.ndarray, kind: str) -> float:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    shared = lambda left, right: np.cumprod(codes[left] == codes[right], axis=1).sum(1)
    if kind == "chain":
        near, far = (shared(a, b) + shared(b, c)) / 2.0, shared(a, c)
    else:
        near, far = shared(b, c), (shared(a, b) + shared(a, c)) / 2.0
    return float(near.mean() - far.mean())


def main() -> None:
    codes = {arm: codes_of(arm) for arm in ARMS}
    levels: dict[tuple[str, str], list[float]] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rows, cols, _ = chains.loaded_edges()
        out_map, in_map = chains.adjacency(rows, cols)
        edges = np.stack([rows, cols], axis=1)
        banks = {"chain": chains.chain3(out_map, edges, rng, chains.CAP),
                 "fork": chains.fan(out_map, in_map, rng, chains.CAP, "fork"),
                 "merge": chains.fan(out_map, in_map, rng, chains.CAP, "merge")}
        for kind, bank in banks.items():
            held_out = proto.split_pair_disjoint(bank)[1]
            for arm in ARMS:
                levels.setdefault((kind, arm), []).append(delta(codes[arm], held_out, kind))
    summary = {}
    for kind in ("chain", "fork", "merge"):
        base = np.array(levels[(kind, ARMS[0])])
        summary[kind] = {"baseline": levels[(kind, ARMS[0])]}
        for arm in ARMS[1:]:
            value = np.array(levels[(kind, arm)])
            summary[kind][arm] = {"values": levels[(kind, arm)],
                                  "difference_vs_baseline": (value - base).tolist(),
                                  "mean_difference": float((value - base).mean()),
                                  "sd_difference": float((value - base).std())}
    print(f"seeds {SEEDS}")
    for kind, entry in summary.items():
        print(f"{kind:6s} baseline sd {np.std(entry['baseline']):.5f}")
        for arm in ARMS[1:]:
            block = entry[arm]
            print(f"       {arm:26s} mean diff {block['mean_difference']:+.5f} "
                  f"sd {block['sd_difference']:.5f}  per-seed "
                  + " ".join(f"{x:+.4f}" for x in block["difference_vs_baseline"]))
    (HERE / "arm_banks.json").write_text(
        json.dumps({"seeds": list(SEEDS), "arms": list(ARMS), "levels": summary}, indent=1) + "\n")


if __name__ == "__main__":
    main()
