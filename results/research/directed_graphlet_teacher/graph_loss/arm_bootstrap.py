#!/usr/bin/env python3
"""Paired bootstrap of the alignment difference between two arms.

The metric is a mean over a fixed bank of special triples, so the only sampling uncertainty it
carries is over those triples.  Resampling triple indices in lockstep for two arms gives the
distribution of the *difference* on the same bank; a 95% interval that excludes zero says the
observed gap is larger than pair sampling noise.  It says nothing about training variance --
two runs of the same recipe differ far more, and that variance is unmeasured here because seed
replication is not allowed.
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
RESAMPLES = 2000


def codes_of(arm: str) -> np.ndarray:
    payload = json.loads((ROOT / "results" / "stage2_RQ-VAE" / arm / "item_sids.json").read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def per_triple(codes, bank, kind):
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    shared = lambda left, right: np.cumprod(codes[left] == codes[right], axis=1).sum(1)
    if kind == "chain":
        near = (shared(a, b) + shared(b, c)) / 2.0
        far = shared(a, c)
    else:
        near = shared(b, c)
        far = (shared(a, b) + shared(a, c)) / 2.0
    return np.stack([near, far], axis=1)


def main() -> None:
    arm_a, arm_b = (sys.argv[1:3] if len(sys.argv) > 2
                    else ["TIGER_chain_align_rq", "TIGER_RQ-VAE"])
    rng = np.random.default_rng(42)
    rows, cols, _ = chains.loaded_edges()
    out_map, in_map = chains.adjacency(rows, cols)
    edges = np.stack([rows, cols], axis=1)
    banks = {"chain": chains.chain3(out_map, edges, rng, chains.CAP),
             "fork": chains.fan(out_map, in_map, rng, chains.CAP, "fork"),
             "merge": chains.fan(out_map, in_map, rng, chains.CAP, "merge")}
    banks = {kind: proto.split_pair_disjoint(bank)[1] for kind, bank in banks.items()}
    codes = {arm: codes_of(arm) for arm in (arm_a, arm_b)}
    out = {}
    boot_rng = np.random.default_rng(7)
    for kind, bank in banks.items():
        parts = {arm: per_triple(code, bank, kind) for arm, code in codes.items()}
        delta = {arm: float((p[:, 0].mean() - p[:, 1].mean())) for arm, p in parts.items()}
        diffs = np.empty(RESAMPLES)
        n = len(bank)
        for index in range(RESAMPLES):
            take = boot_rng.integers(0, n, n)
            diffs[index] = ((parts[arm_a][take, 0].mean() - parts[arm_a][take, 1].mean())
                            - (parts[arm_b][take, 0].mean() - parts[arm_b][take, 1].mean()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[kind] = {"arm_a": delta[arm_a], "arm_b": delta[arm_b],
                     "difference": delta[arm_a] - delta[arm_b],
                     "ci95": [float(lo), float(hi)],
                     "excludes_zero": bool(lo > 0 or hi < 0), "triples": int(n)}
        print(f"{kind:6s} {arm_a} {delta[arm_a]:+.5f}  {arm_b} {delta[arm_b]:+.5f}  "
              f"diff {delta[arm_a] - delta[arm_b]:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  "
              f"{'excludes 0' if out[kind]['excludes_zero'] else 'includes 0'}")
    name = HERE / f"bootstrap_{arm_a}_vs_{arm_b}.json"
    name.write_text(json.dumps({"resamples": RESAMPLES, "result": out}, indent=1) + "\n")
    print("written", name)


if __name__ == "__main__":
    main()
