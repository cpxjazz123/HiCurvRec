#!/usr/bin/env python3
"""Constrain the quantised vector with the behaviour adjacency directly.

No teacher, no GNN: the batch adjacency of the real behaviour graph is the target, and the
only quantity that can explain it is q.

    A_hat = sigmoid(Q W Q^T)                     W in R^{32x32}, asymmetric by construction
    L_graph = weighted BCE over masked (i, j)    diagonal and held-out edges masked out
    L_total = L_RQVAE + lambda * L_graph

lambda is calibrated once from the ratio of encoder-gradient norms (the fixed lambda=1
earlier let an auxiliary term dominate by ~3000x), the codebooks stay frozen so the
geometry behind the exported SID is preserved, and every arm is evaluated with the same
protocol: held-out direction probe with a null control, held-out neighbour recall@5, SID
collision, and a semantic probe on q.

Arms: no graph loss; undirected adjacency; randomly oriented adjacency; directed adjacency.
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
sys.path.insert(0, str(HERE.parent / "inject_directed"))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))

from inject_directed import (ARM_DIR, RESULT_DIR, SEMANTICS, FULL_GRAPH, grad_norm,
                             load_model, model_config, q_straight_through, sha256)  # noqa: E402
import augment_gnn as base  # noqa: E402

OUT = HERE / "graph_loss.json"
STEPS = 1000
BATCH_ITEMS = 512
NEIGHBOURS_PER_SEED = 3
LAMBDA_FALLBACK = 1e-3


def batch_with_neighbours(out_table, anchors, rng, size: int):
    """A batch that contains real edges: seeds plus a few of their out-neighbours."""
    chosen = []
    while len(chosen) < size:
        seed = int(anchors[rng.integers(0, len(anchors))])
        chosen.append(seed)
        neighbours = out_table.get(seed, [])
        if neighbours:
            take = rng.choice(len(neighbours), min(NEIGHBOURS_PER_SEED, len(neighbours)),
                              replace=False)
            chosen.extend(int(neighbours[i]) for i in take)
    return np.array(chosen[:size], dtype=np.int64)


def adjacency_of(items: np.ndarray, out_table: dict, mask_held: set, mode: str, rng):
    """Target matrix, mask and weights for one batch, all derived from the real graph."""
    n = len(items)
    index = {int(item): position for position, item in enumerate(items.tolist())}
    target = np.zeros((n, n), dtype=np.float32)
    for a in items.tolist():
        for b in out_table.get(int(a), []):
            if b not in index or a == b:
                continue
            if (min(a, b), max(a, b)) in mask_held:
                continue
            i, j = index[int(a)], index[int(b)]
            if mode == "undirected":
                target[i, j] = target[j, i] = 1.0
            elif mode == "random":
                if rng.random() < 0.5:
                    target[i, j] = 1.0
                else:
                    target[j, i] = 1.0
            else:
                target[i, j] = 1.0
    mask = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)
    weight = np.where(target > 0, 1.0, 0.1).astype(np.float32)
    # the reverse of a true edge carries weight 1: it is the direction-bearing negative
    reverse = (target.T > 0) & (target == 0)
    weight = np.where(reverse, 1.0, weight).astype(np.float32)
    weight = weight * mask
    return target, mask, weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    out_table = {}
    for a, b in zip(msg_rows.tolist(), msg_cols.tolist()):
        out_table.setdefault(a, []).append(b)
    anchors = np.array([k for k, v in out_table.items() if v], dtype=np.int64)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]

    report = {"split": {"message_edges": int(len(msg_rows)),
                        "held_edges": int(len(held_rows)),
                        "held_one_way": int(len(one_way))},
              "protocol": {"loss": "weighted BCE on sigmoid(Q W Q^T), diagonal and held-out "
                                   "edges masked, positives and reverse non-edges weight 1, "
                                   "ordinary negatives 0.1",
                           "lambda": "calibrated from encoder-gradient norms",
                           "codebook": "frozen",
                           "steps": STEPS, "batch_items": BATCH_ITEMS, "seed": base.SEED,
                           "eval": "direction probe with null, neighbour recall@5, SID "
                                   "collision, semantic probe"},
              "arms": {}}
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    for arm in ("A_none", "B_undirected", "C_random", "D_directed"):
        torch.manual_seed(base.SEED)
        model = load_model().to(device)
        for layer in model.rq.vq_layers:
            layer.embed.requires_grad_(False)
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
        relation = torch.nn.Parameter(torch.eye(32, device=device) * 0.1)
        optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + [relation],
                                      lr=1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(base.SEED)
        lam = None
        trace = []
        for step in range(STEPS):
            model.train()
            batch = batch_with_neighbours(out_table, anchors, rng, BATCH_ITEMS)
            mode = ("directed" if arm == "D_directed" else
                    "undirected" if arm == "B_undirected" else "random")
            target, mask, weight = adjacency_of(batch, out_table, held_unit, mode, rng)
            chosen = torch.from_numpy(batch).to(device)
            recon, quant_parts, _, _ = model(items[chosen])
            loss_rq, _ = criterion(recon, items[chosen], quant_parts)
            loss_aux = torch.zeros((), device=device)
            if arm != "A_none":
                q, _ = q_straight_through(model, items[chosen])
                predicted = torch.sigmoid(q @ relation @ q.T)
                m = torch.from_numpy(mask).to(device)
                w = torch.from_numpy(weight).to(device)
                t = torch.from_numpy(target).to(device)
                bce = -(t * torch.log(predicted.clamp(min=1e-7))
                        + (1 - t) * torch.log((1 - predicted).clamp(min=1e-7)))
                loss_aux = (bce * m * w).sum() / (m * w).sum().clamp(min=1.0)
                if lam is None:
                    g_rq = grad_norm(list(model.encoder.parameters()), loss_rq)
                    g_aux = grad_norm(list(model.encoder.parameters()), loss_aux)
                    lam = float(g_rq / max(g_aux, 1e-12))
                    print(f"    {arm}: lambda {lam:.3e} "
                          f"(grad rq {g_rq:.2e}, aux {g_aux:.2e})", flush=True)
            loss = loss_rq + (lam if lam else LAMBDA_FALLBACK) * loss_aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == STEPS - 1:
                trace.append({"step": step, "rqvae": float(loss_rq.detach()),
                              "graph": float(loss_aux.detach())})
                print(f"    {arm} step {step}: rqvae {float(loss_rq):.5f} "
                      f"graph {float(loss_aux):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            q_np = q.cpu().numpy().astype(np.float32)
            codes = tokens.cpu().numpy().astype(np.int64)
            predicted_full = torch.sigmoid(
                torch.from_numpy(q_np).to(device) @ relation
                @ torch.from_numpy(q_np).to(device).T).cpu().numpy()
        probe = base.probe(q_np, one_way, device, steps=2500)
        rng_null = np.random.default_rng(base.SEED + 31)
        shuffled = one_way.copy()
        flip = rng_null.random(len(shuffled)) < 0.5
        shuffled[flip] = shuffled[flip][:, ::-1]
        probe_null = base.probe(q_np, shuffled, device, steps=2500)
        recall_hits, recall_total = 0, 0
        for a, b in one_way[:2000].tolist():
            scores = predicted_full[int(a)].copy()
            scores[int(a)] = -np.inf
            top = np.argpartition(-scores, 5)[:5]
            recall_hits += int(int(b) in top.tolist())
            recall_total += 1
        rng_sem = np.random.default_rng(base.SEED)
        order = rng_sem.permutation(len(q_np))
        cut = int(0.8 * len(order))
        train_idx, eval_idx = order[:cut], order[cut:]
        xc = q_np[train_idx] - q_np[train_idx].mean(0, keepdims=True)
        gram = xc.T @ xc
        lam_reg = 1e-3 * float(np.trace(gram)) / xc.shape[1]
        weights = np.linalg.solve(gram + lam_reg * np.eye(gram.shape[0]),
                                  xc.T @ (semantics[train_idx] - semantics[train_idx].mean(0)))
        predicted_sem = (q_np[eval_idx] - q_np[train_idx].mean(0)) @ weights \
            + semantics[train_idx].mean(0)
        sem_r2 = float(1.0 - np.mean((predicted_sem - semantics[eval_idx]) ** 2)
                       / np.mean((semantics[eval_idx] - semantics[eval_idx].mean(0)) ** 2))
        report["arms"][arm] = {
            "lambda": lam, "trace": trace,
            "q_shift": float(np.linalg.norm(q_np - (
                np.load(HERE.parent / "fidelity" / "fidelity_states.npz")["q"]))
                             / max(len(q_np), 1)),
            "direction_probe": probe["direction_accuracy"],
            "direction_probe_null": probe_null["direction_accuracy"],
            "neighbour_recall_at_5": recall_hits / max(recall_total, 1),
            "sid_collision": float(1.0 - len(np.unique(codes[:, :3], axis=0)) / len(codes)),
            "level1_codes_used": int(len(np.unique(codes[:, 0]))),
            "semantic_probe_r2": sem_r2}
        print(f"  {arm}: probe {probe['direction_accuracy']:.4f} "
              f"(null {probe_null['direction_accuracy']:.4f}) recall@5 "
              f"{report['arms'][arm]['neighbour_recall_at_5']:.4f} "
              f"collision {report['arms'][arm]['sid_collision']:.4f} "
              f"semR2 {sem_r2:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
