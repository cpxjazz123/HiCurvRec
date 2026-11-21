#!/usr/bin/env python3
"""DiGAE-1L concatenation instead of an MLP fusion: x' = [s || t] in 768 dimensions.

One propagation layer, two 384-d halves, and two objectives:

    L = L_edge + lambda * L_semantic
    L_edge = BCE(s_v . t_u)          on positives, reverse hard negatives, random negatives
    L_sem  = MSE(decoder([s || t]), x)   keeps the item's semantics inside the pair

Arms (identical split, budget, seed, head, and evaluation pairs):

    A  raw 768-d semantics
    B  [s || t] trained on the edge loss only
    C  [s || t] trained on the edge loss plus the semantic reconstruction

Evaluated on held-out one-way edges with the same asymmetric head for every arm, plus two
semantic checks: cosine between the state and the original vector, and the fraction of the
original variance the state reconstructs.
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

OUT = HERE / "concat_diage.json"
STEPS = 1200
HEAD_STEPS = 1500
HALF = 384
LAMBDA_SEM = 1.0


def build_model(in_dim: int = 768, half: int = HALF):
    import torch.nn as nn

    class ConcatDiGAE1L(nn.Module):
        """One directed propagation layer, then the two halves are concatenated."""

        def __init__(self) -> None:
            super().__init__()
            self.source_in = nn.Linear(in_dim, half)
            self.target_in = nn.Linear(in_dim, half)
            self.ss = nn.Linear(half, half)
            self.st = nn.Linear(half, half)
            self.tt = nn.Linear(half, half)
            self.ts = nn.Linear(half, half)
            self.act = nn.ELU()
            self.decoder = nn.Linear(2 * half, in_dim)

        def forward(self, x, out_adj, in_adj, out_norm, in_norm):
            s0 = self.act(self.source_in(x))
            t0 = self.act(self.target_in(x))
            s = self.act(self.ss(s0) + self.st(torch.sparse.mm(out_adj, t0) / out_norm))
            t = self.act(self.tt(t0) + self.ts(torch.sparse.mm(in_adj, s0) / in_norm))
            return s, t

        def state(self, s, t):
            return torch.cat([s, t], dim=-1)

    return ConcatDiGAE1L()


def train_state(arm: str, x, out_adj, in_adj, out_norm, in_norm, triples, device):
    torch.manual_seed(base.SEED)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(base.SEED)
    trace = []
    for step in range(STEPS):
        model.train()
        loss_total = 0.0
        for pairs, label in triples:
            pick = torch.randint(0, len(pairs), (base.BATCH,), generator=generator)
            s, t = model(x, out_adj, in_adj, out_norm, in_norm)
            chosen = pairs[pick]
            logits = (s[chosen[:, 0]] * t[chosen[:, 1]]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.full_like(logits, label))
            if arm == "C_concat_plus_semantic":
                state = model.state(s, t)
                loss = loss + LAMBDA_SEM * torch.nn.functional.mse_loss(
                    model.decoder(state), x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss)
        if step % 400 == 0 or step == STEPS - 1:
            trace.append({"step": step, "loss": loss_total / len(triples)})
            print(f"    {arm} step {step}: loss {loss_total / len(triples):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        s, t = model(x, out_adj, in_adj, out_norm, in_norm)
    return (model.state(s, t).cpu().numpy().astype(np.float32),
            s.cpu().numpy().astype(np.float32), t.cpu().numpy().astype(np.float32), trace)


def train_head(state, triples, device):
    import torch.nn as nn
    torch.manual_seed(base.SEED)
    dim = state.shape[1]
    source = nn.Linear(dim, 64).to(device)
    target = nn.Linear(dim, 64).to(device)
    optimizer = torch.optim.AdamW(list(source.parameters()) + list(target.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(state).to(device)
    rng = np.random.default_rng(base.SEED)
    for _ in range(HEAD_STEPS):
        for pairs, label in triples:
            pick = torch.from_numpy(rng.integers(0, len(pairs), 256)).to(device)
            chosen = pairs[pick]
            logits = (source(x[chosen[:, 0]]) * target(x[chosen[:, 1]])).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.full_like(logits, label))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    source.eval(), target.eval()
    return source, target


def scores(state, source, target, pairs, device):
    with torch.no_grad():
        x = torch.from_numpy(state).to(device)
        chosen = torch.from_numpy(pairs).to(device)
        return (source(x[chosen[:, 0]]) * target(x[chosen[:, 1]])).sum(-1).cpu().numpy()


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
    out_adj, in_adj, out_deg, in_deg = base.sparse_adjacency(msg_rows, msg_cols, n_items, device)
    out_norm, in_norm = out_deg + 1.0, in_deg + 1.0
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    rng = np.random.default_rng(base.SEED)

    pos = np.stack([msg_rows, msg_cols], axis=1)
    pos = pos[rng.permutation(len(pos))[:80000]]
    rev = np.ascontiguousarray(pos[:, ::-1])[
        np.array([(int(b), int(a)) not in directed for a, b in pos.tolist()])]
    neg = []
    while len(neg) < len(rev):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        # a training negative must not be a training edge, and must not be a held-out edge
        # either: otherwise the sampler can present a test edge as an easy negative
        if a == b or (min(a, b), max(a, b)) in train_unit or (min(a, b), max(a, b)) in held_unit:
            continue
        neg.append((a, b))
    neg = np.array(neg, dtype=np.int64)
    triples = [(torch.from_numpy(pos).to(device), 1.0),
               (torch.from_numpy(rev).to(device), 0.0),
               (torch.from_numpy(neg).to(device), 0.0)]

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    # held-out edges are excluded from both training positives and training negatives; the
    # evaluation negatives additionally exclude train_unit and held_unit
    eval_pos = held_pairs[np.array([(b, a) not in held_directed
                                    for a, b in held_pairs.tolist()])][:8000]
    eval_rev = np.ascontiguousarray(eval_pos[:, ::-1])
    eval_neg = []
    while len(eval_neg) < len(eval_pos):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit or (min(a, b), max(a, b)) in held_unit:
            continue
        eval_neg.append((a, b))
    eval_neg = np.array(eval_neg, dtype=np.int64)

    from sklearn.metrics import average_precision_score, roc_auc_score
    report = {"split": {"message_edges": int(len(msg_rows)),
                        "held_edges": int(len(held_rows)),
                        "eval_positives": int(len(eval_pos)),
                        "training_positives": int(len(pos)),
                        "training_reverse_negatives": int(len(rev))},
              "protocol": {"architecture": "DiGAE-1L, two 384-d halves concatenated to 768",
                           "edge_loss": "BCE on positives, reverse hard negatives and random "
                                        "negatives",
                           "semantic_loss": "MSE(decoder([s||t]), x), weight 1.0 in arm C",
                           "steps": STEPS, "head_steps": HEAD_STEPS, "seed": base.SEED},
              "arms": {}}

    states = {"A_raw_semantics": semantics}
    halves = {}
    for arm in ("B_concat_edge_only", "C_concat_plus_semantic"):
        states[arm], s_np, t_np, trace = train_state(
            arm, x, out_adj, in_adj, out_norm, in_norm, triples, device)
        halves[arm] = (s_np, t_np)
        report["arms"].setdefault(arm, {})["trace"] = trace
        print(f"  trained {arm}", flush=True)

    for arm, state in states.items():
        labels_set = {}
        # A has no pair-specific vectors, so it is read through the same small head; B and C
        # are read BOTH ways -- directly from their trained 384-d halves (no extra
        # projection) and through the head -- so a loss caused by the head is visible.
        if arm in halves:
            s_np, t_np = halves[arm]
            direct = lambda pairs, s_np=s_np, t_np=t_np: (s_np[pairs[:, 0]]
                                                          * t_np[pairs[:, 1]]).sum(-1)
            pos_score = direct(eval_pos)
            neg_score = direct(eval_neg)
            rev_score = direct(eval_rev)
            labels = np.concatenate([np.ones(len(pos_score)),
                                     np.zeros(len(neg_score) + len(rev_score))])
            values = np.concatenate([pos_score, neg_score, rev_score])
            labels_set["diage_direct_no_head"] = {
                "roc_auc": float(roc_auc_score(labels, values)),
                "average_precision": float(average_precision_score(labels, values)),
                "direction_accuracy": float((pos_score > rev_score).mean())}
        source, target = train_head(state, triples, device)
        pos_score = scores(state, source, target, eval_pos, device)
        neg_score = scores(state, source, target, eval_neg, device)
        rev_score = scores(state, source, target, eval_rev, device)
        labels = np.concatenate([np.ones(len(pos_score)),
                                 np.zeros(len(neg_score) + len(rev_score))])
        values = np.concatenate([pos_score, neg_score, rev_score])
        rng_probe = np.random.default_rng(base.SEED)
        order = rng_probe.permutation(len(state))
        cut = int(0.8 * len(order))
        train_idx, eval_idx = order[:cut], order[cut:]
        xc = state[train_idx] - state[train_idx].mean(0, keepdims=True)
        gram = xc.T @ xc
        lam = 1e-3 * float(np.trace(gram)) / xc.shape[1]
        weights = np.linalg.solve(gram + lam * np.eye(gram.shape[0]),
                                  xc.T @ (semantics[train_idx] - semantics[train_idx].mean(0)))
        predicted = (state[eval_idx] - state[train_idx].mean(0)) @ weights \
            + semantics[train_idx].mean(0)
        variance = float(1.0 - np.mean((predicted - semantics[eval_idx]) ** 2)
                         / np.mean((semantics[eval_idx] - semantics[eval_idx].mean(0)) ** 2))
        labels_set["with_asymmetric_head"] = {
            "roc_auc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
            "direction_accuracy": float((pos_score > rev_score).mean())}
        entry = report["arms"].setdefault(arm, {})
        entry.update({"readouts": labels_set,
                      "semantic_linear_probe_r2": variance})
        print(f"  {arm}: " + " | ".join(
            f"{k} AUC {v['roc_auc']:.4f} AP {v['average_precision']:.4f} "
            f"dir {v['direction_accuracy']:.4f}" for k, v in labels_set.items())
              + f" | probe R2 {variance:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
