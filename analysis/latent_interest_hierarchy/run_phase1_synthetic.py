"""Positive control for phase 1: does this experiment DETECT a hierarchy when one is
known to exist?

A null result on real data is only meaningful if the experiment can detect the effect
it is looking for. Here the sequences are GENERATED from a known two-level interest
hierarchy (8 parents x 8 children over the same catalogue size), and the same five
arms are run through the same code path. If `hier` does not beat `random_hier` and
`flat` here, the real-data comparison has no power and its negative result cannot be
interpreted. If it does, the experiment is sensitive and the real-data null stands.

The generator never uses coordinates in any manifold, so this control tests the
experiment's sensitivity to a HIERARCHY, not to a geometry.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/latent_interest_hierarchy/run_phase1_synthetic.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import metrics as M  # noqa: E402
import models as MD  # noqa: E402
import run_phase1 as R  # noqa: E402

N_SYNTH_USERS = 20000
SYNTH_SEQ_MIN, SYNTH_SEQ_MAX = 3, 12
# How strongly a user sticks to their preferred leaf: the generative signal a
# hierarchy-aware model should be able to exploit and a flat one should not.
ALPHA_OWN_LEAF = 0.70


def make_synthetic(seed: int):
    """Users drawn from a known two-level interest hierarchy over real catalogue size."""
    rng = np.random.default_rng(seed)
    n_items = C.N_ITEMS
    # balanced item -> leaf assignment (the planted hierarchy)
    leaf_of_item = np.arange(n_items) % C.N_INTEREST
    items_of_leaf = [np.nonzero(leaf_of_item == c)[0] for c in range(C.N_INTEREST)]

    rows = []
    truth = []
    for u in range(N_SYNTH_USERS):
        parent = int(rng.integers(0, C.N_COARSE))
        child = int(rng.integers(0, C.N_FINE))
        leaf = parent * C.N_FINE + child
        L = int(rng.integers(SYNTH_SEQ_MIN, SYNTH_SEQ_MAX + 1))
        # noise leaves: other children of the same parent (a within-parent confusion)
        sib = [parent * C.N_FINE + int(rng.integers(0, C.N_FINE))
               for _ in range(2)]
        seq = []
        for _ in range(L + 1):
            r = rng.random()
            if r < ALPHA_OWN_LEAF:
                pool = items_of_leaf[leaf]
            elif r < ALPHA_OWN_LEAF + (1 - ALPHA_OWN_LEAF) / 2:
                pool = items_of_leaf[sib[0]]
            else:
                pool = items_of_leaf[sib[1]]
            seq.append(int(pool[rng.integers(0, len(pool))]))
        rows.append(seq)
        truth.append((parent, leaf))
    # a synthetic next-item pair: the sequence itself is the history and the last item
    # the target, with the SAME length profile as the real study
    pairs = [(u, s[:len(s) - 1][-C.MAX_HISTORY:], s[-1]) for u, s in enumerate(rows)]
    train = [(u, s[:max(1, len(s) - 2)], s[-1]) for u, s in enumerate(rows)]
    return train, pairs, truth, leaf_of_item


def main() -> None:
    (C.OUT_DIR / "synthetic").mkdir(parents=True, exist_ok=True)
    out_path = C.OUT_DIR / "synthetic/phase1_synth_runs.json"
    log = lambda m: print(m, flush=True)
    log(f"=== phase 1 positive control, device={R.DEVICE} ===")

    results = {"config": {"n_users": N_SYNTH_USERS, "alpha_own_leaf": ALPHA_OWN_LEAF,
                          "n_coarse": C.N_COARSE, "n_fine": C.N_FINE,
                          "epochs": C.EPOCHS, "train_users": C.TRAIN_SUBSET_USERS},
               "runs": {}}
    train, pairs, truth, leaf_of_item = make_synthetic(C.SEED_BASE)
    rng = np.random.default_rng(C.SEED_BASE)
    pick = rng.permutation(N_SYNTH_USERS)[:C.TRAIN_SUBSET_USERS]
    sub_train = [train[int(i)] for i in pick]
    log(f"planted hierarchy: {C.N_INTEREST} leaves over {C.N_ITEMS} items; "
        f"{len(sub_train)} training pairs, {len(pairs)} evaluation pairs")

    for arm in C.ARMS:
        mode = C.ARM_TO_MODE[arm]
        for seed in C.SEEDS:
            key = f"{arm}_s{seed}"
            if key in results["runs"]:
                continue
            rows = sub_train
            if arm in C.SHUFFLED_ARMS:
                sh = {}
                for u, h, t in sub_train:
                    perm = list(h) + [t]
                    np.random.default_rng(seed + (u % 100003)).shuffle(perm)
                    sh[u] = perm
                rows = [(u, perm[:max(1, len(perm) - 1)], perm[-1]) for u, perm in sh.items()]
            model, res = R.train_one(mode, seed, rows, pairs, log)
            assign = M.learned_assignment(model)
            results["runs"][key] = {"arm": arm, "mode": mode, "seed": seed,
                                    "nll": res["nll"],
                                    "recall_at_10": res["recall_at_10"],
                                    "per_user_nll": res["per_user_nll"].tolist(),
                                    "per_user_hit": res["per_user_hit"].tolist(),
                                    "assignment": assign.tolist()}
            log(f"  [{key}] nll={res['nll']:.4f} recall@10={res['recall_at_10']:.4f}")
            out_path.write_text(json.dumps(results))

    # does the hierarchical arm recover the PLANTED grouping?
    gp = {}
    for arm in ("hier", "random_hier"):
        a = np.asarray(results["runs"][f"{arm}_s{C.SEEDS[0]}"]["assignment"])
        leaf_to_parent = np.empty(C.N_INTEREST, dtype=np.int64)
        for p in range(C.N_COARSE):
            for f in range(C.N_FINE):
                leaf_to_parent[p * C.N_FINE + f] = p
        planted = leaf_to_parent[leaf_of_item]           # item -> true parent
        # fold the learned leaf assignment to a parent by majority within each leaf
        pred = np.empty(C.N_ITEMS, dtype=np.int64)
        for c in range(C.N_INTEREST):
            idx = np.nonzero(leaf_of_item == c)[0]
            pred[idx] = np.bincount(a[idx], minlength=C.N_COARSE).argmax()
        gp[arm] = M.partition_agreement(pred, planted, C.N_COARSE)
        log(f"  {arm}: agreement with the PLANTED parent partition = {gp[arm]:.4f}")
    results["planted_recovery"] = gp
    out_path.write_text(json.dumps(results))
    log("positive control complete")


if __name__ == "__main__":
    main()