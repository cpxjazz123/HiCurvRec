"""Issue #194 acceptance 4, second half: what share of the distillation's own candidate pairs fall inside the support?

The #193 figure (94.95% of pairs share no code level) was measured over the teacher's ``T`` support.  The distillation step does not use that population --
it draws candidate sets per anchor (one hop, two hops, global) -- so the same quantity has to be counted on the population the mechanism actually sees.
This script does that with the #181 arm's shipped SID and this tree's own candidate-set rule, so the number is comparable to the audit's 94.95%.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARM_DIR = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_sdh_rq"
SID_PATH = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq" / "item_sids.json"
ANCHORS = 4000
SEED = 42


def main() -> int:
    sys.path.insert(0, str(ARM_DIR))
    spec = importlib.util.spec_from_file_location("arm_sdh", ARM_DIR / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["arm_sdh"] = arm
    spec.loader.exec_module(arm)
    from model import struct_distill

    frame = pd.read_parquet(arm.TRAIN_FILE)
    edges = sorted({(int(h[-1]), int(t)) for h, t in
                    zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                    if h is not None and len(h) > 0})
    n_items = int(max(max(edge) for edge in edges)) + 1

    neighbours: dict[int, list[int]] = {}
    for i, j in edges:
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    two_hop = {node: struct_distill.two_hop_of(node, neighbours) for node in list(neighbours)}

    exported = json.loads(SID_PATH.read_text())
    sids = np.full((n_items, 3), -1, dtype=np.int64)
    for key, value in exported.items():
        row = int(key)
        if row < n_items:
            sids[row] = value[:3]

    rng = np.random.default_rng(SEED)
    anchors = [int(a) for a in rng.choice(n_items, size=ANCHORS, replace=False) if sids[int(a)][0] >= 0]
    anchors = [a for a in anchors if a in neighbours]
    sets = struct_distill.candidate_sets(anchors, neighbours, two_hop, n_items, seed=SEED)

    depth_counts = np.zeros(4, dtype=np.int64)
    pairs = 0
    anchors_with_support = 0
    for anchor, chosen in zip(anchors, sets):
        if not chosen:
            continue
        codes = sids[np.asarray(chosen)]
        shared = (codes == sids[anchor][None, :]).sum(axis=1)
        for depth in range(4):
            depth_counts[depth] += int((shared == depth).sum())
        pairs += len(chosen)
        anchors_with_support += int((shared >= 1).any())

    report = {
        "sid_source": str(SID_PATH.relative_to(REPO_ROOT)),
        "train_file": str(arm.TRAIN_FILE.relative_to(REPO_ROOT)),
        "anchors_requested": ANCHORS,
        "anchors_used": len(anchors),
        "anchors_with_non_empty_support": anchors_with_support,
        "pairs": pairs,
        "share_by_shared_levels": {str(d): float(depth_counts[d] / max(pairs, 1)) for d in range(4)},
        "support_share_at_least_one_level": float(1.0 - depth_counts[0] / max(pairs, 1)),
        "audit_193_reference_share_none": 0.9494785269223038,
        "note": "the #193 reference was measured over the teacher's T support; this one over the distillation's candidate sets, which is the population "
                "the support mask acts on",
    }
    (HERE / "support_shares.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
