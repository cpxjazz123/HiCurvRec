"""Phase 1: is there a stable, predictive latent interest hierarchy in real purchase
sequences?

Five matched arms, three seeds each. Every arm has identical parameters, identical
data volume, identical optimisation budget and an identical next-item objective; the
only thing that changes is the structure of the bottleneck over the same 64
centroids, or whether the purchase order is destroyed.

  flat            ordered data, flat bottleneck over 64 interests
  hier            ordered data, 8 coarse x 8 fine bottleneck
  random_hier     ordered data, SAME bottleneck with a FIXED RANDOM partition
  flat_shuffled   order-destroyed data, flat bottleneck
  hier_shuffled   order-destroyed data, hierarchical bottleneck

`hier` vs `random_hier` is the falsifying comparison for the hierarchy itself:
identical capacity, identical compute, a grouping that either means something or does
not. `flat` vs `hier` is the pre-registered hypothesis. `*_shuffled` says how much
of any signal is order rather than interest.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/latent_interest_hierarchy/run_phase1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import data as D  # noqa: E402
import metrics as M  # noqa: E402
import models as MD  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_batches(rows, batch_size, rng):
    idx = rng.permutation(len(rows))
    for s in range(0, len(idx), batch_size):
        yield [rows[int(i)] for i in idx[s:s + batch_size]]


def pad(hists):
    L = max((len(h) for h in hists), default=1)
    L = max(L, 1)
    H = np.zeros((len(hists), L), dtype=np.int64)
    for i, h in enumerate(hists):
        if h:
            H[i, L - len(h):] = [x + 1 for x in h]
    return H


def train_one(mode: str, seed: int, train_rows, eval_rows, log) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MD.InterestModel(C.N_ITEMS + 1, C.EMBED_DIM, mode, seed=seed).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    t0 = time.time()
    for ep in range(C.EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for batch in build_batches(train_rows, C.BATCH_SIZE, rng):
            hists = [h for _u, h, _t in batch]
            tgts = torch.as_tensor([t + 1 for _u, _h, t in batch], dtype=torch.long,
                                   device=DEVICE)
            Ht = torch.as_tensor(pad(hists), dtype=torch.long, device=DEVICE)
            mask = (Ht != 0).float().unsqueeze(-1)
            emb = model.embed(Ht) * mask
            logits = model(emb, mask)
            loss = torch.nn.functional.cross_entropy(logits, tgts)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tot += float(loss) * len(batch)
            n += len(batch)
        log(f"    epoch {ep + 1}/{C.EPOCHS} train_loss={tot / max(n, 1):.4f} "
            f"({time.time() - t0:.0f}s)")
    res = M.evaluate(model, eval_rows, DEVICE, topk=C.TOPK)
    res["mode"] = mode
    res["seed"] = seed
    res["train_seconds"] = time.time() - t0
    return model, res


def main() -> None:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.OUT_DIR / "phase1_runs.json"

    log = lambda m: print(m, flush=True)

    log(f"=== phase 1, device={DEVICE} ===")
    tr = pd.read_parquet(C.TRAIN_PARQUET)
    va = pd.read_parquet(C.VALID_PARQUET)
    seqs = D.user_train_sequences(tr)
    log(f"users={len(seqs)} train_instances={sum(len(s) - 1 for s in seqs.values())}")
    eval_rows = D.evaluation_rows(va, seqs)          # raises if any history is not the prefix
    log(f"valid rows={len(eval_rows)} histories verified against the train prefix")
    freq = D.item_frequency(seqs)
    tr_items = set(np.nonzero(freq)[0].tolist())
    v_tgts = set(_t for _u, _h, _t in eval_rows)
    log(f"valid targets: {len(v_tgts)} distinct items, "
        f"{len(v_tgts - tr_items)} never appear as a train target")
    pop = D.popularity_baseline(eval_rows, freq)
    log(f"popularity baseline: nll={pop['nll']:.4f} recall@10={pop['recall_at_10']:.4f}")

    rng = np.random.default_rng(C.SEED_BASE)
    users = sorted(seqs)
    pick = rng.permutation(len(users))[:C.TRAIN_SUBSET_USERS]
    subset = {users[int(i)]: seqs[users[int(i)]] for i in pick}
    log(f"training on {len(subset)} users (identical subset in every arm and seed)")

    ordered = D.train_instances(subset)
    shuffled = {s: D.shuffled_sequences(subset, seed=s) for s in C.SEEDS}
    log(f"ordered instances={len(ordered)}")

    results = {"config": {"seeds": list(C.SEEDS), "arms": list(C.ARMS),
                          "embed_dim": C.EMBED_DIM, "n_interest": C.N_INTEREST,
                          "n_coarse": C.N_COARSE, "n_fine": C.N_FINE,
                          "epochs": C.EPOCHS, "batch_size": C.BATCH_SIZE,
                          "lr": C.LR, "train_users": C.TRAIN_SUBSET_USERS,
                          "max_history": C.MAX_HISTORY, "n_params": None},
               "popularity_baseline": {k: v for k, v in pop.items()},
               "runs": {}}
    if out_path.exists():
        results = json.loads(out_path.read_text())
        results["config"]["n_params"] = MD.InterestModel(
            C.N_ITEMS + 1, C.EMBED_DIM, "flat").n_parameters()

    plans = []
    for arm in C.ARMS:
        mode = C.ARM_TO_MODE[arm]        # never inferred from the name's prefix
        is_shuf = arm in C.SHUFFLED_ARMS
        for seed in C.SEEDS:
            plans.append((arm, mode, seed, is_shuf))

    for arm, mode, seed, is_shuf in plans:
        key = f"{arm}_s{seed}"
        if key in results["runs"]:
            log(f"  [{key}] already done, skipping")
            continue
        log(f"  [{key}] training")
        rows = D.train_instances(shuffled[seed]) if is_shuf else ordered
        model, res = train_one(mode, seed, rows, eval_rows, log)
        assign = M.learned_assignment(model)
        tm = M.transition_mass(eval_rows, lambda i: int(assign[i]))
        results["runs"][key] = {
            "arm": arm, "mode": mode, "seed": seed, "shuffled": is_shuf,
            "n_params": model.n_parameters(),
            "nll": res["nll"], "recall_at_10": res["recall_at_10"],
            "per_user_nll": res["per_user_nll"].tolist(),
            "per_user_hit": res["per_user_hit"].tolist(),
            "valid_hist_len": np.array([len(h) for _u, h, _t in eval_rows]).tolist(),
            "valid_target": [t for _u, _h, t in eval_rows],
            "transition": tm,
            "assignment": assign.tolist(),
        }
        log(f"  [{key}] nll={res['nll']:.4f} recall@10={res['recall_at_10']:.4f} "
            f"within-group transitions={tm['within_group_transition_rate']:.4f}")
        results["config"]["n_params"] = model.n_parameters()
        out_path.write_text(json.dumps(results))
    log("phase 1 runs complete")


if __name__ == "__main__":
    main()