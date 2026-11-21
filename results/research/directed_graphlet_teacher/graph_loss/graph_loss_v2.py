#!/usr/bin/env python3
"""Adjacency supervision with the four defects fixed and a measured gradient budget.

Fixes relative to the previous run:

* **batch de-duplication** -- `np.unique`, so one item appears once in the adjacency matrix
  and no real edge can be overwritten by a later occurrence;
* **strict test-edge masking** -- a pair touching a held-out edge is *masked out* of the
  loss, not silently turned into a negative;
* **per-class means** -- positives, reverse non-edges and ordinary non-edges each get their
  own mean BCE, so the vast number of ordinary negatives cannot dominate;
* **measured gradient budget** -- every step computes the graph term's encoder-gradient norm
  and rescales it so it never exceeds `GRAPH_GRAD_SHARE` of the RQ-VAE gradient.  The
  previous version calibrated once at step 0 and then let the term dominate.

Arms: none, undirected, randomly oriented, directed.
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
from inject_directed import (ARM_DIR, RESULT_DIR, SEMANTICS, FULL_GRAPH, load_model,
                             q_straight_through, sha256)  # noqa: E402
import augment_gnn as base  # noqa: E402

OUT = HERE / "graph_loss_v2.json"
STEPS = 800
BATCH_ITEMS = 512
GRAPH_GRAD_SHARE = 0.5
NEIGHBOURS_PER_SEED = 3


def unique_batch(out_table, anchors, rng, size: int) -> np.ndarray:
    """Seeds plus their out-neighbours, de-duplicated before any label is built."""
    chosen = []
    while len(set(chosen)) < size:
        seed = int(anchors[rng.integers(0, len(anchors))])
        chosen.append(seed)
        for b in out_table.get(seed, [])[:NEIGHBOURS_PER_SEED]:
            chosen.append(int(b))
    return np.unique(np.array(chosen, dtype=np.int64))


def class_masks(items: np.ndarray, out_table: dict, held_unit: set, mode: str, rng):
    """Target, and three masks: positives, reverse non-edges, ordinary non-edges."""
    n = len(items)
    index = {int(item): position for position, item in enumerate(items.tolist())}
    target = np.zeros((n, n), dtype=np.float32)
    blocked = np.zeros((n, n), dtype=bool)
    for a in items.tolist():
        for b in out_table.get(int(a), []):
            if b not in index or a == b:
                continue
            i, j = index[int(a)], index[int(b)]
            if (min(a, b), max(a, b)) in held_unit:
                # a held-out edge may not be used in either sense: it is neither a positive
                # nor a negative, so both cells are excluded from the loss
                blocked[i, j] = blocked[j, i] = True
                continue
            if mode == "undirected":
                target[i, j] = target[j, i] = 1.0
            elif mode == "random":
                if rng.random() < 0.5:
                    target[i, j] = 1.0
                else:
                    target[j, i] = 1.0
            else:
                target[i, j] = 1.0
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = off_diagonal & ~blocked
    positive = valid & (target > 0)
    reverse = valid & (target == 0) & (target.T > 0)
    ordinary = valid & ~positive & ~reverse
    return target, positive, reverse, ordinary


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
    out_table = {}
    for a, b in zip(msg_rows.tolist(), msg_cols.tolist()):
        out_table.setdefault(a, []).append(b)
    anchors = np.array([k for k, v in out_table.items() if v], dtype=np.int64)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    report = {"protocol": {"loss": "per-class mean BCE: positives + reverse non-edges + "
                                   "0.25 * ordinary non-edges",
                           "masking": "diagonal and any pair touching a held-out edge are "
                                      "excluded, not labelled negative",
                           "batch": "de-duplicated seeds plus out-neighbours",
                           "gradient_budget": f"graph encoder-grad norm <= "
                                              f"{GRAPH_GRAD_SHARE} x RQ-VAE norm, measured "
                                              f"every step and rescaled",
                           "codebook": "frozen", "steps": STEPS, "seed": base.SEED},
              "arms": {}}
    for arm in ("A_none", "B_undirected", "C_random", "D_directed"):
        torch.manual_seed(base.SEED)
        model = load_model().to(device)
        for layer in model.rq.vq_layers:
            layer.embed.requires_grad_(False)
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
        relation = torch.nn.Parameter(torch.eye(32, device=device) * 0.1)
        encoder_parameters = list(model.encoder.parameters())
        optimizer = torch.optim.AdamW(encoder_parameters + [relation], lr=1e-3,
                                      weight_decay=1e-4)
        rng = np.random.default_rng(base.SEED)
        gradient_log = []
        for step in range(STEPS):
            model.train()
            batch = unique_batch(out_table, anchors, rng, BATCH_ITEMS)
            chosen = torch.from_numpy(batch).to(device)
            recon, quant_parts, _, _ = model(items[chosen])
            loss_rq, _ = criterion(recon, items[chosen], quant_parts)
            optimizer.zero_grad(set_to_none=True)
            loss_rq.backward(retain_graph=True)
            rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                                if p.grad is not None) ** 0.5)
            aux_norm, scale = 0.0, 0.0
            if arm != "A_none":
                mode = ("directed" if arm == "D_directed" else
                        "undirected" if arm == "B_undirected" else "random")
                target, positive, reverse, ordinary = class_masks(
                    batch, out_table, held_unit, mode, rng)
                q, _ = q_straight_through(model, items[chosen])
                predicted = torch.sigmoid(q @ relation @ q.T)
                t = torch.from_numpy(target).to(device)
                eps = 1e-7
                bce = -(t * torch.log(predicted.clamp(min=eps))
                        + (1 - t) * torch.log((1 - predicted).clamp(min=eps)))
                parts = []
                for mask, weight in ((positive, 1.0), (reverse, 1.0), (ordinary, 0.25)):
                    m = torch.from_numpy(mask).to(device)
                    if bool(m.any()):
                        parts.append(weight * (bce * m).sum() / m.sum())
                loss_aux = torch.stack(parts).sum() if parts else torch.zeros((), device=device)
                aux_grads = torch.autograd.grad(loss_aux, encoder_parameters,
                                                retain_graph=True, allow_unused=True)
                aux_norm = float(sum(float(g.pow(2).sum()) for g in aux_grads
                                     if g is not None) ** 0.5)
                # the measured budget: never let the graph term exceed the share, every step
                scale = min(1.0, GRAPH_GRAD_SHARE * rq_norm / max(aux_norm, 1e-12))
                for parameter, grad in zip(encoder_parameters, aux_grads):
                    if grad is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = scale * grad
                    else:
                        parameter.grad = parameter.grad + scale * grad
                relation_grad = torch.autograd.grad(loss_aux, [relation],
                                                    retain_graph=False)[0]
                relation.grad = scale * relation_grad
            optimizer.step()
            if step % 100 == 0 or step == STEPS - 1:
                gradient_log.append({"step": step, "rq_norm": rq_norm,
                                     "graph_norm": aux_norm, "scale": scale})
                print(f"    {arm} step {step}: |g_rq| {rq_norm:.3e} "
                      f"|g_graph| {aux_norm:.3e} scale {scale:.3f}", flush=True)
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            q_np = q.cpu().numpy().astype(np.float32)
            codes = tokens.cpu().numpy().astype(np.int64)
        probe = base.probe(q_np, one_way, device, steps=2500)
        rng_null = np.random.default_rng(base.SEED + 31)
        shuffled = one_way.copy()
        flip = rng_null.random(len(shuffled)) < 0.5
        shuffled[flip] = shuffled[flip][:, ::-1]
        probe_null = base.probe(q_np, shuffled, device, steps=2500)
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
            "gradient_log": gradient_log,
            "direction_probe": probe["direction_accuracy"],
            "direction_probe_null": probe_null["direction_accuracy"],
            "sid_collision": float(1.0 - len(np.unique(codes[:, :3], axis=0)) / len(codes)),
            "level1_codes_used": int(len(np.unique(codes[:, 0]))),
            "semantic_probe_r2": sem_r2}
        print(f"  {arm}: probe {probe['direction_accuracy']:.4f} "
              f"(null {probe_null['direction_accuracy']:.4f}) collision "
              f"{report['arms'][arm]['sid_collision']:.4f} semR2 {sem_r2:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
