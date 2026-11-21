#!/usr/bin/env python3
"""Local supervision: positives, reversed pairs and non-edges from the *same* subgraph.

Global sampling hands the model pairs drawn from the whole catalogue, so "is there an edge"
can be answered from item popularity alone.  Here the three classes come from one cand
subgraph's own members:

    within a 3-node instance:  true directed edge   -> positive
                               its reverse          -> hard negative
                               a member pair with no edge either way -> easy negative

Both arms use the same encoder, budget, seed and evaluation; only the sampling distribution
differs.  Evaluation stays on held-out edges, which are absent from the message graph, from
training positives and from training negatives.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import augment_gnn as base  # noqa: E402
from direction_arms import build_model  # noqa: E402

INSTANCES = TEACHER / ".." / "directed_graphlet_role" / "instances.npz"
OUT = HERE / "local_pairs.json"
STEPS = 1200
ALPHA = base.ALPHA


def build_globals(device):
    graph = np.load(base.FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    return (rows, cols, n_items, (msg_rows, msg_cols), (held_rows, held_cols),
            train_unit, held_unit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cols, n_items, (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = \
        build_globals(device)
    semantics = np.load(base.SEMANTICS).astype(np.float32)
    x = torch.from_numpy(semantics).to(device)
    edge_index = torch.from_numpy(np.stack([msg_rows, msg_cols]).astype(np.int64)).to(device)
    # direction_arms' model takes (x, edges, node_count, alpha); mean aggregation = None
    edges = {"out": (edge_index[0], edge_index[1], None),
             "in": (edge_index[1], edge_index[0], None)}
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    rng = np.random.default_rng(base.SEED)

    # ---- local triples: drawn from the training instances of the 3-node pool
    pool = np.load(INSTANCES)
    instances = np.concatenate([pool["P3_nodes"][pool["P3_split"] == 0],
                                pool["K3_nodes"][pool["K3_split"] == 0]])
    rng.shuffle(instances)
    local = []
    for members in instances[:200000].tolist():
        # a pair from the held-out slice is never used, in either direction
        if any((min(a, b), max(a, b)) in held_unit for a in members for b in members if a != b):
            continue
        edges_here = [(a, b) for a in members for b in members
                      if a != b and (a, b) in directed]
        for a, b in edges_here:
            if (b, a) not in directed:
                local.append((a, b, 1.0))
                local.append((b, a, 0.0))
        for a in members:
            for b in members:
                if a == b or (a, b) in directed or (b, a) in directed:
                    continue
                local.append((a, b, 0.0))
    local_pairs = torch.tensor([[r[0], r[1]] for r in local], dtype=torch.long, device=device)
    local_labels = torch.tensor([r[2] for r in local], dtype=torch.float32, device=device)
    print(f"  local samples {len(local)} "
          f"(positive share {float(local_labels.mean()):.3f})", flush=True)

    # ---- global triples for the control arm
    pos = np.stack([msg_rows, msg_cols], axis=1)
    pos = pos[rng.permutation(len(pos))[:80000]]
    rev = np.ascontiguousarray(pos[:, ::-1])[
        np.array([(int(b), int(a)) not in directed for a, b in pos.tolist()])]
    neg = []
    while len(neg) < len(rev):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit or (min(a, b), max(a, b)) in held_unit:
            continue
        neg.append((a, b))
    global_pairs = torch.tensor(
        [tuple(x) for x in np.concatenate([pos, rev, neg])], dtype=torch.long, device=device)
    # positives only for real edges; the reversed pair and the random pair are both
    # negatives.  The previous line marked the reversed edges positive as well, so the
    # control arm was trained to call a direction-less pair "an edge".
    global_labels = torch.cat([torch.ones(len(pos)),
                               torch.zeros(len(rev) + len(neg))]).to(device)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    eval_pos = held_pairs[np.array([(b, a) not in held_directed
                                    for a, b in held_pairs.tolist()])][:8000]
    eval_rev = np.ascontiguousarray(eval_pos[:, ::-1])
    one_way = eval_pos

    report = {"sampling": {"local_samples": int(len(local)),
                           "global_samples": int(len(global_pairs))},
              "protocol": {"encoder": "dual DiGAE (from direction_arms), unweighted mean",
                           "steps": STEPS, "batch": base.BATCH, "seed": base.SEED,
                           "local": "positives, reverses and non-edges from one 3-node "
                                    "instance; held-out pairs excluded",
                           "global": "random negatives over the catalogue, degree-matched",
                           "eval": "8,000 held-out one-way edges, direct s.t and probe"},
              "arms": {}}
    for arm, pairs, labels in (("A_global", global_pairs, global_labels),
                               ("B_local_subgraph", local_pairs, local_labels)):
        torch.manual_seed(base.SEED)
        model = build_model(semantics.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(base.SEED)
        for step in range(STEPS):
            model.train()
            pick = torch.randint(0, len(pairs), (base.BATCH,), generator=generator)
            out = model(x, edges, n_items, ALPHA)
            left, right = pairs[pick, 0], pairs[pick, 1]
            target = labels[pick]
            logits = (out["s"][left] * out["t"][right]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                (out["s2"][left] * out["t2"][right]).sum(-1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == STEPS - 1:
                print(f"    {arm} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, edges, n_items, ALPHA)
            s_np = out["s"].cpu().numpy()
            t_np = out["t"].cpu().numpy()
            fused = out["fused"].cpu().numpy().astype(np.float32)
        direct = lambda p: (s_np[p[:, 0]] * t_np[p[:, 1]]).sum(-1)
        probe = base.probe(fused, one_way, device, steps=2500)
        # null control for the probe itself: permuting the direction labels of the
        # evaluation pairs must bring it back to chance; if it does not, the probe is
        # measuring something other than direction and its column cannot be quoted
        rng_null = np.random.default_rng(base.SEED + 31)
        shuffled = one_way.copy()
        flip = rng_null.random(len(shuffled)) < 0.5
        shuffled[flip] = shuffled[flip][:, ::-1]
        probe_null = base.probe(fused, shuffled, device, steps=2500)
        report["arms"][arm] = {
            "direct_direction": float((direct(eval_pos) > direct(eval_rev)).mean()),
            "probe_direction": probe["direction_accuracy"],
            "probe_null_control": probe_null["direction_accuracy"],
            "probe_pairs_eval": probe["pairs_eval"]}
        print(f"  {arm}: direct dir {report['arms'][arm]['direct_direction']:.4f} "
              f"probe {probe['direction_accuracy']:.4f} "
              f"(null {probe_null['direction_accuracy']:.4f})", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
