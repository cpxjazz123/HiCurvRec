#!/usr/bin/env python3
"""Split the prediction into "is there a connection" and "which way", with local sampling.

A single inner product has to serve two targets at once: the pair A->B and its reverse B->A
are both "a connection exists", and only one of them is the edge.  This script separates it:

    existence  p_conn(A,B) = sigmoid(g([x_A, x_B, x_A*x_B, |x_A - x_B|]))   symmetric input
    direction  p_dir(A->B) = sigmoid((s_A . t_B - s_B . t_A) / 2)          antisymmetric
    p(A->B)    = p_conn * p_dir,  p(B->A) = p_conn * (1 - p_dir)

Training data comes from the local 3-node instances (true edges, reversed pairs, and
member pairs with no edge either way), and the two losses are separate:

    L = BCE(p_conn, has_a_connection) + BCE(p_dir, is_this_the_direction) + L_rqvae_free

Two arms share the encoder, the samples, the budget and the evaluation:

    A  single inner product, BCE on the ordered pair     (the previous protocol)
    B  decomposed existence x direction, plus a direction ranking loss on one-way edges
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
OUT = HERE / "decomposed.json"
STEPS = 1200
ALPHA = base.ALPHA


def local_samples(msg_rows, msg_cols, held_unit, n_items, seed):
    """Positives, reversed pairs and non-edges, all from one 3-node instance."""
    pool = np.load(INSTANCES)
    instances = np.concatenate([pool["P3_nodes"][pool["P3_split"] == 0],
                                pool["K3_nodes"][pool["K3_split"] == 0]])
    rng = np.random.default_rng(seed)
    rng.shuffle(instances)
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    samples = []
    for members in instances[:200000].tolist():
        if any((min(a, b), max(a, b)) in held_unit
               for a in members for b in members if a != b):
            continue
        for a in members:
            for b in members:
                if a == b:
                    continue
                forward, backward = (a, b) in directed, (b, a) in directed
                if forward and not backward:
                    samples.append((a, b, 1.0, 1.0))    # exists, this is the direction
                    samples.append((b, a, 1.0, 0.0))    # exists, other direction
                elif forward and backward:
                    samples.append((a, b, 1.0, 0.5))    # exists, direction undefined
                else:
                    samples.append((a, b, 0.0, 0.5))    # no connection
    return samples


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
    edges = {"out": (edge_index[0], edge_index[1], None),
             "in": (edge_index[1], edge_index[0], None)}

    samples = local_samples(msg_rows, msg_cols, held_unit, n_items, base.SEED)
    pairs = torch.tensor([[s[0], s[1]] for s in samples], dtype=torch.long, device=device)
    labels = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)
    directions = torch.tensor([s[3] for s in samples], dtype=torch.float32, device=device)
    print(f"  local samples {len(samples)} "
          f"(connection {float(labels.float().mean()):.3f}, "
          f"direction-defined {float((directions != 0.5).float().mean()):.3f})", flush=True)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    eval_pos = held_pairs[np.array([(b, a) not in held_directed
                                    for a, b in held_pairs.tolist()])][:8000]
    eval_neg = []
    rng = np.random.default_rng(base.SEED)
    while len(eval_neg) < len(eval_pos):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit or (min(a, b), max(a, b)) in held_unit:
            continue
        eval_neg.append((a, b))
    eval_neg = np.array(eval_neg, dtype=np.int64)

    from sklearn.metrics import average_precision_score, roc_auc_score
    report = {"sampling": {"local_samples": int(len(samples)),
                           "connection_share": float(labels.float().mean()),
                           "direction_defined_share":
                               float((directions != 0.5).float().mean())},
              "protocol": {"encoder": "dual DiGAE (direction_arms), unweighted mean",
                           "steps": STEPS, "batch": base.BATCH, "seed": base.SEED,
                           "existence": "symmetric MLP on [x_A, x_B, x_A*x_B, |x_A-x_B|]",
                           "direction": "sigmoid((s_A.t_B - s_B.t_A)/2), ranking loss on "
                                        "one-way pairs",
                           "eval": "held-out one-way edges, direct and probe with null"},
              "arms": {}}
    for arm in ("A_single_inner_product", "B_decomposed"):
        torch.manual_seed(base.SEED)
        model = build_model(semantics.shape[1]).to(device)
        existence = torch.nn.Sequential(
            torch.nn.Linear(4 * semantics.shape[1], 256), torch.nn.GELU(),
            torch.nn.Linear(256, 1)).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + (list(existence.parameters())
                                        if arm == "B_decomposed" else []),
            lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(base.SEED)
        for step in range(STEPS):
            model.train()
            pick = torch.randint(0, len(pairs), (base.BATCH,), generator=generator)
            out = model(x, edges, n_items, ALPHA)
            left, right = pairs[pick, 0], pairs[pick, 1]
            if arm == "A_single_inner_product":
                logits = (out["s"][left] * out["t"][right]).sum(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[pick])
            else:
                fused = out["fused"]
                pair_feature = torch.cat([fused[left], fused[right],
                                          fused[left] * fused[right],
                                          (fused[left] - fused[right]).abs()], dim=-1)
                conn_logits = existence(pair_feature).squeeze(-1)
                margin = ((out["s"][left] * out["t"][right]).sum(-1)
                          - (out["s"][right] * out["t"][left]).sum(-1))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    conn_logits, labels[pick])
                defined = directions[pick] != 0.5
                if bool(defined.any()):
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        margin[defined] / 2.0, directions[pick][defined])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == STEPS - 1:
                print(f"    {arm} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, edges, n_items, ALPHA)
            s_np, t_np = out["s"].cpu().numpy(), out["t"].cpu().numpy()
            fused = out["fused"].cpu().numpy().astype(np.float32)
            if arm == "B_decomposed":
                def conn_score(pairs_np):
                    with torch.no_grad():
                        chosen = torch.from_numpy(pairs_np).to(device)
                        feat = torch.cat([
                            out["fused"][chosen[:, 0]], out["fused"][chosen[:, 1]],
                            out["fused"][chosen[:, 0]] * out["fused"][chosen[:, 1]],
                            (out["fused"][chosen[:, 0]]
                             - out["fused"][chosen[:, 1]]).abs()], dim=-1)
                        return existence(feat).squeeze(-1).cpu().numpy()
            else:
                def conn_score(pairs_np):
                    with torch.no_grad():
                        chosen = torch.from_numpy(pairs_np).to(device)
                        return (out["s"][chosen[:, 0]] * out["t"][chosen[:, 1]]
                                ).sum(-1).cpu().numpy()
        labels_auc = np.concatenate([np.ones(len(eval_pos)), np.zeros(len(eval_neg))])
        values = np.concatenate([conn_score(eval_pos), conn_score(eval_neg)])
        direct = float((conn_score(eval_pos) > conn_score(np.ascontiguousarray(eval_pos[:, ::-1]))).mean())
        probe = base.probe(fused, eval_pos, device, steps=2500)
        rng_null = np.random.default_rng(base.SEED + 31)
        shuffled = eval_pos.copy()
        flip = rng_null.random(len(shuffled)) < 0.5
        shuffled[flip] = shuffled[flip][:, ::-1]
        probe_null = base.probe(fused, shuffled, device, steps=2500)
        report["arms"][arm] = {
            "existence_auc": float(roc_auc_score(labels_auc, values)),
            "existence_ap": float(average_precision_score(labels_auc, values)),
            "direct_direction": direct,
            "probe_direction": probe["direction_accuracy"],
            "probe_null": probe_null["direction_accuracy"]}
        print(f"  {arm}: AUC {report['arms'][arm]['existence_auc']:.4f} "
              f"AP {report['arms'][arm]['existence_ap']:.4f} "
              f"direct {direct:.4f} probe {probe['direction_accuracy']:.4f} "
              f"(null {probe_null['direction_accuracy']:.4f})", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
