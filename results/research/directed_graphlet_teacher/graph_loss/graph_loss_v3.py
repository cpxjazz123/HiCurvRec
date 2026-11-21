#!/usr/bin/env python3
"""Adjacency supervision, third pass: true masking, a free W, one batch sequence.

Fixes relative to ``graph_loss_v2.py``:

* **the mask now sees the held-out edges at all** -- the previous version looked for
  them inside ``out_table``, which is built from the *training* edges, so a held-out
  pair could never be found and silently became an ordinary negative.  Masking is now
  decided over every pair in the batch against the full graph's held set, and the
  sampler deliberately brings held partners into the batch so the mask has to fire.
* **the relation matrix trains freely** -- only the graph term's effect on the encoder
  is budgeted; ``W`` keeps its own (unscaled) gradient, since the whole point is to let
  it learn which asymmetric relation explains the edges.
* **one batch sequence for every arm**, and the random-orientation control fixes its
  orientation once rather than re-drawing it per batch, so the two are comparable.
* **structure fidelity is measured on the SID itself** -- direction accuracy and
  existence AUC of held-out edges read from the quantised vector, not from the loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inject_directed"))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
from inject_directed import (SEMANTICS, FULL_GRAPH, load_model,  # noqa: E402
                             q_straight_through)
import augment_gnn as base  # noqa: E402

OUT = HERE / "graph_loss_v3.json"
STEPS = 800
BATCH_ITEMS = 512
GRAPH_GRAD_SHARE = 0.5
TRAIN_NEIGHBOURS = 2
HELD_NEIGHBOURS = 1


def plan_batches(out_table, held_table, anchors, rng, size: int, steps: int) -> list:
    """One batch sequence, drawn once and reused by every arm.

    Each seed contributes two training out-neighbours and one held-out partner: the
    first keeps positives flowing, the second makes the mask do real work instead of
    never meeting a held pair.
    """
    batches = []
    for _ in range(steps):
        chosen = []
        while len(set(chosen)) < size:
            seed = int(anchors[rng.integers(0, len(anchors))])
            chosen.append(seed)
            chosen.extend(out_table.get(seed, [])[:TRAIN_NEIGHBOURS])
            chosen.extend(held_table.get(seed, [])[:HELD_NEIGHBOURS])
        batches.append(np.unique(np.array(chosen, dtype=np.int64)))
    return batches


def class_masks(items: np.ndarray, out_table: dict, held_partner: dict, mode: str,
                orientation: dict):
    """Target and three masks, with held-out pairs excluded in *both* directions."""
    n = len(items)
    index = {int(item): position for position, item in enumerate(items.tolist())}
    target = np.zeros((n, n), dtype=np.float32)
    blocked = np.zeros((n, n), dtype=bool)
    for position, a in enumerate(items.tolist()):
        for b in held_partner.get(a, ()):
            other = index.get(b)
            if other is not None:
                blocked[position, other] = blocked[other, position] = True
    for a in items.tolist():
        for b in out_table.get(a, []):
            if b not in index or a == b:
                continue
            i, j = index[int(a)], index[int(b)]
            if blocked[i, j]:
                continue
            if mode == "undirected":
                target[i, j] = target[j, i] = 1.0
            elif mode == "random":
                if orientation[(min(int(a), int(b)), max(int(a), int(b)))] > 0:
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
    return target, positive, reverse, ordinary, int(blocked.sum() // 2)


def fold_held(rows: np.ndarray, cols: np.ndarray) -> dict:
    table = {}
    for a, b in zip(rows.tolist(), cols.tolist()):
        table.setdefault(a, []).append(b)
        table.setdefault(b, []).append(a)
    return table


def existence_auc(x: np.ndarray, positives: np.ndarray, negatives: np.ndarray,
                  device, holdout: float = 0.3, steps: int = 1500) -> float:
    """Can the state tell a held-out edge from a pair with no edge at all?"""
    import torch.nn as nn
    rng = np.random.default_rng(base.SEED + 11)
    pairs = np.concatenate([positives, negatives], axis=0)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    items = np.unique(pairs.ravel())
    rng.shuffle(items)
    cut = int((1 - holdout) * len(items))
    train_items = set(items[:cut].tolist())
    eval_items = set(items[cut:].tolist())
    tr = np.array([i for i, (a, b) in enumerate(pairs.tolist())
                   if a in train_items and b in train_items], dtype=np.int64)
    ev = np.array([i for i, (a, b) in enumerate(pairs.tolist())
                   if a in eval_items and b in eval_items], dtype=np.int64)
    torch.manual_seed(base.SEED)

    def features(rows):
        a, b = pairs[rows][:, 0], pairs[rows][:, 1]
        return torch.from_numpy(np.concatenate(
            [x[a], x[b], x[a] * x[b], np.abs(x[a] - x[b])], -1).astype(np.float32)).to(device)

    head = nn.Sequential(nn.LayerNorm(4 * x.shape[1]), nn.Linear(4 * x.shape[1], 256),
                         nn.GELU(), nn.Linear(256, 1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr, xev = features(tr), features(ev)
    ytr = torch.from_numpy(labels[tr].astype(np.float32)).to(device).unsqueeze(-1)
    generator = torch.Generator(device="cpu").manual_seed(base.SEED)
    for _ in range(steps):
        pick = torch.randint(0, len(xtr), (min(2048, len(xtr)),), generator=generator)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(xtr[pick]),
                                                                    ytr[pick])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = head(xev).squeeze(-1).cpu().numpy()
    labels_ev = labels[ev]
    order = np.argsort(scores)  # ascending ranks; descending inverted the statistic
    ranked = labels_ev[order]
    positives_total = ranked.sum()
    negatives_total = len(ranked) - positives_total
    if positives_total == 0 or negatives_total == 0:
        return float("nan")
    ranks = np.arange(1, len(ranked) + 1)
    return float((ranks[ranked == 1].sum() - positives_total * (positives_total + 1) / 2)
                 / (positives_total * negatives_total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--qdir", default=str(HERE / "q"))
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    out_table = {}
    for a, b in zip(msg_rows.tolist(), msg_cols.tolist()):
        out_table.setdefault(a, []).append(b)
    held_table = fold_held(held_rows, held_cols)
    held_partner = held_table
    anchors = np.array([k for k, v in out_table.items() if v], dtype=np.int64)
    batches = plan_batches(out_table, held_table, anchors,
                           np.random.default_rng(base.SEED), BATCH_ITEMS, STEPS)
    orientation_rng = np.random.default_rng(base.SEED + 5)
    orientation = {unit: (1 if orientation_rng.random() < 0.5 else -1) for unit in train_unit}

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]
    all_train = set(map(tuple, zip(msg_rows.tolist(), msg_cols.tolist())))
    pairs_by_direction = {(min(a, b), max(a, b)) for a, b in all_train}
    # The negative keeps the positive's source and only replaces its target, so the probe
    # cannot separate positives from negatives by item degree -- the earlier global-random
    # negatives did exactly that and drove the AUC below chance.
    all_pairs = set(pairs_by_direction) | set(held_unit)
    negative_rng = np.random.default_rng(base.SEED + 13)
    negatives = []
    for a, _ in held_pairs.tolist():
        while True:
            c = int(negative_rng.integers(0, len(semantics)))
            if c == a or (min(a, c), max(a, c)) in all_pairs:
                continue
            negatives.append([a, c])
            break
    negatives = np.array(negatives, dtype=np.int64)
    from inject_directed import ARM_DIR
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    report = {"protocol": {
        "masking": "every batch pair that is a held-out edge (either direction) is "
                   "excluded from the loss; the sampler brings held partners into the "
                   "batch so the mask fires",
        "orientation_control": "one fixed orientation per train edge, shared by every "
                               "batch and drawn once",
        "batch_sequence": "drawn once and replayed identically for every arm",
        "gradient_budget": f"graph encoder-grad norm <= {GRAPH_GRAD_SHARE} x RQ-VAE norm; "
                           f"the relation matrix trains unscaled",
        "codebook": "frozen", "steps": STEPS, "seed": base.SEED},
        "arms": {}}
    for arm in ("A_none", "B_undirected", "C_random", "D_directed"):
        cached = Path(args.qdir) / f"{arm}.npy"
        codes = None
        gradient_log, blocked_log = [], []
        if args.reuse and cached.is_file():
            q_np = np.load(cached).astype(np.float32)
            print(f"  {arm}: reused {cached}", flush=True)
        else:
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
            gradient_log, blocked_log = [], []
            for step, batch in enumerate(batches):
                model.train()
                chosen = torch.from_numpy(batch).to(device)
                recon, quant_parts, _, _ = model(items[chosen])
                loss_rq, _ = criterion(recon, items[chosen], quant_parts)
                optimizer.zero_grad(set_to_none=True)
                loss_rq.backward(retain_graph=True)
                rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                                    if p.grad is not None) ** 0.5)
                aux_norm = scale = 0.0
                if arm != "A_none":
                    mode = ("directed" if arm == "D_directed" else
                            "undirected" if arm == "B_undirected" else "random")
                    target, positive, reverse, ordinary, blocked_pairs = class_masks(
                        batch, out_table, held_partner, mode, orientation)
                    blocked_log.append(blocked_pairs)
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
                    relation.grad = relation_grad  # W is not budgeted
                optimizer.step()
                if step % 200 == 0 or step == STEPS - 1:
                    gradient_log.append({"step": step, "rq_norm": rq_norm,
                                         "graph_norm": aux_norm, "scale": scale,
                                         "blocked_pairs": blocked_log[-1] if blocked_log else 0})
                    print(f"    {arm} step {step}: |g_rq| {rq_norm:.3e} "
                          f"|g_graph| {aux_norm:.3e} scale {scale:.3f} "
                          f"blocked {gradient_log[-1]['blocked_pairs']}", flush=True)
            if blocked_log and sum(blocked_log) == 0:
                raise RuntimeError(f"{arm}: the mask never fired -- the held pairs never "
                                   f"reached a batch, so masking would be vacuous")
            model.eval()
            Path(args.qdir).mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                q, tokens = q_straight_through(model, items)
                q_np = q.cpu().numpy().astype(np.float32)
                codes = tokens.cpu().numpy().astype(np.int64)
                np.save(Path(args.qdir) / f"{arm}.npy", q_np)
        probe = base.probe(q_np, one_way, device, steps=2500)
        rng_null = np.random.default_rng(base.SEED + 31)
        shuffled = one_way.copy()
        flip = rng_null.random(len(shuffled)) < 0.5
        shuffled[flip] = shuffled[flip][:, ::-1]
        probe_null = base.probe(q_np, shuffled, device, steps=2500)
        existence = existence_auc(q_np, held_pairs, negatives, device)
        null_positives = []
        while len(null_positives) < len(held_pairs):
            a, b = negative_rng.integers(0, len(semantics), 2)
            if a == b or (min(int(a), int(b)), max(int(a), int(b))) in pairs_by_direction:
                continue
            null_positives.append([int(a), int(b)])
        existence_null = existence_auc(q_np, np.array(null_positives, dtype=np.int64),
                                       negatives, device)
        rng_sem = np.random.default_rng(base.SEED)
        order = rng_sem.permutation(len(q_np))
        cut = int(0.8 * len(order))
        train_idx, eval_idx = order[:cut], order[cut:]
        xc = q_np[train_idx] - q_np[train_idx].mean(0, keepdims=True)
        lam = 1e-3 * float(np.trace(xc.T @ xc)) / xc.shape[1]
        weights = np.linalg.solve(xc.T @ xc + lam * np.eye(xc.shape[1]),
                                  xc.T @ (semantics[train_idx] - semantics[train_idx].mean(0)))
        predicted_sem = (q_np[eval_idx] - q_np[train_idx].mean(0)) @ weights \
            + semantics[train_idx].mean(0)
        sem_r2 = float(1.0 - np.mean((predicted_sem - semantics[eval_idx]) ** 2)
                       / np.mean((semantics[eval_idx] - semantics[eval_idx].mean(0)) ** 2))
        report["arms"][arm] = {
            "gradient_log": gradient_log,
            "masked_pairs_mean": float(np.mean(blocked_log)) if blocked_log else 0.0,
            "direction_probe": probe["direction_accuracy"],
            "direction_probe_null": probe_null["direction_accuracy"],
            "existence_auc": existence,
            "existence_auc_null": existence_null,
            "sid_collision": (float(1.0 - len(np.unique(codes[:, :3], axis=0)) / len(codes))
                              if codes is not None else None),
            "semantic_probe_r2": sem_r2}
        collision_text = (f"{report['arms'][arm]['sid_collision']:.4f}"
                          if report["arms"][arm]["sid_collision"] is not None else "cached")
        print(f"  {arm}: dir {probe['direction_accuracy']:.4f} "
              f"(null {probe_null['direction_accuracy']:.4f}) "
              f"exist {existence:.4f} (null {existence_null:.4f}) "
              f"collision {collision_text} semR2 {sem_r2:.3f} "
              f"masked {report['arms'][arm]['masked_pairs_mean']:.1f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
