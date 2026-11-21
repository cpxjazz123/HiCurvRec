#!/usr/bin/env python3
"""Clean A/B: does the direction ranking loss help, with everything else held fixed?

Two defects in the previous five-arm run made its B arm uninformative:

* the one-way filter tested `(b, a) not in train_unit`, where `train_unit` holds
  *unordered* pairs, so the test was always true and mutual edges were trained as if they
  had a direction;
* `aggregate` received the behaviour weights in every arm, so A and B were already
  weighted and the C ablation measured nothing.

This script removes both: mean aggregation (no weights, no attention, no role features)
and a one-way filter against the **directed** training edge set.  Only the direction loss
differs between A and B.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import augment_gnn as base  # noqa: E402
from direction_arms import build_model  # noqa: E402

OUT = HERE / "dir_ablation.json"
STEPS = base.STEPS
BATCH = base.BATCH
ALPHA = base.ALPHA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(base.FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    semantics = np.load(base.SEMANTICS).astype(np.float32)
    x = torch.from_numpy(semantics).to(device)
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    edge_index = torch.from_numpy(np.stack([msg_rows, msg_cols]).astype(np.int64)).to(device)
    # mean aggregation in both arms: weight None selects the unweighted branch
    edges = {"out": (edge_index[0], edge_index[1], None),
             "in": (edge_index[1], edge_index[0], None)}
    msg_directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))

    rng = np.random.default_rng(base.SEED)
    pos = np.stack([msg_rows, msg_cols], axis=1)
    neg = []
    while len(neg) < len(pos):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit:
            continue
        neg.append((a, b))
    neg = np.array(neg, dtype=np.int64)
    pairs = torch.from_numpy(np.concatenate([pos, neg]).astype(np.int64)).to(device)
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)
    # true one-way pairs only: the reverse direction is absent from the DIRECTED edge set
    # a pair carries a direction to rank only when the forward edge exists AND the reverse
    # does not; testing the reverse alone let unrelated (negative) pairs into the margin
    is_one_way = np.array([(a, b) in msg_directed and (b, a) not in msg_directed
                           for a, b in pairs.tolist()])
    one_way_mask = torch.from_numpy(is_one_way).to(device)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way_held = held_pairs[np.array([(b, a) not in held_directed
                                        for a, b in held_pairs.tolist()])]
    report = {"split": {"message_edges": int(len(msg_rows)),
                        "held_edges": int(len(held_rows)),
                        "one_way_share_in_supervision": float(is_one_way.mean()),
                        "one_way_requires_forward_edge": True,
                        "one_way_held_for_probe": int(len(one_way_held))},
              "protocol": {"steps": STEPS, "batch": BATCH, "seed": base.SEED, "alpha": ALPHA,
                           "aggregation": "unweighted mean in both arms",
                           "direction_loss": "-log sigmoid(s_A.t_B - s_B.t_A), applied only "
                                             "to pairs whose reverse is absent from the "
                                             "directed training edge set",
                           "probe": "seeded, item-isolated"},
              "arms": {}}
    for arm in ("A_bce_only", "B_plus_direction_loss"):
        torch.manual_seed(base.SEED)
        model = build_model(semantics.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(base.SEED)
        for step in range(STEPS):
            model.train()
            pick = torch.randint(0, len(pairs), (BATCH,), generator=generator)
            out = model(x, edges, n_items, ALPHA)
            left, right = pairs[pick, 0], pairs[pick, 1]
            target = labels[pick]
            forward_score = (out["s"][left] * out["t"][right]).sum(-1)
            backward_score = (out["s"][right] * out["t"][left]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward_score, target)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                (out["s2"][left] * out["t2"][right]).sum(-1), target)
            if arm == "B_plus_direction_loss":
                rows_ok = one_way_mask[pick]
                if bool(rows_ok.any()):
                    margin = forward_score[rows_ok] - backward_score[rows_ok]
                    loss = loss - torch.log_softmax(
                        torch.stack([torch.zeros_like(margin), margin], dim=-1), -1)[:, 1].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == STEPS - 1:
                print(f"    {arm} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, edges, n_items, ALPHA)
            fused = out["fused"].cpu().numpy().astype(np.float32)
            pair_t = torch.from_numpy(one_way_held).to(device)
            head = float((((out["s"][pair_t[:, 0]] * out["t"][pair_t[:, 1]]).sum(-1))
                          > ((out["s"][pair_t[:, 1]] * out["t"][pair_t[:, 0]]).sum(-1))
                          ).float().mean())
        probe = base.probe(fused, one_way_held, device)
        report["arms"][arm] = {"edge_head_direction": head,
                               "shift_l2": float(np.linalg.norm(fused - semantics)
                                                 / max(len(fused), 1)),
                               "probe": probe}
        print(f"  {arm}: head {head:.4f} probe {probe['direction_accuracy']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    se = 0.5 / math.sqrt(report["arms"]["A_bce_only"]["probe"]["pairs_eval"])
    delta = (report["arms"]["B_plus_direction_loss"]["probe"]["direction_accuracy"]
             - report["arms"]["A_bce_only"]["probe"]["direction_accuracy"])
    report["readings"] = {"se": round(se, 4), "B_minus_A": round(delta, 4),
                          "B_minus_A_sigma": round(delta / se, 2)}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
