#!/usr/bin/env python3
"""Can the enhanced 768-d state predict concrete directed behaviour edges?

Four states, one asymmetric head, one protocol:

    state in {A raw semantics, B undirected-augmented, C randomly-oriented-augmented,
              D directed-augmented}
    head:  P(A->B) = sigmoid(f_s(x_A) . f_t(x_B))      (two learned projections)

The head is trained on the same triples for every state -- true edge A->B labelled 1, its
reverse B->A labelled 0, a degree-matched unrelated pair A->C labelled 0 -- so a state that
only knows "these items are related" scores 0.5 on the direction metric by construction.

Reported on held-out edges only:
    ROC-AUC          positive vs unrelated negative   (is there an edge?)
    AP               precision-recall for finding edges
    Direction Acc.   positive vs its own reversed pair (does it know the way?)

Held-out edges are excluded from message passing, from supervision and from negative
sampling, and the evaluation pairs never reuse a training pair.
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

OUT = HERE / "edge_prediction.json"
AUG_STEPS = 1200
HEAD_STEPS = 1500
HEAD_DIM = 64
ALPHA = base.ALPHA


def train_augmenter(arm: str, x, edges, n_items, device, label_mode: str):
    from dual_augment import build_dual_model
    torch.manual_seed(base.SEED)
    model = build_dual_model(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    msg_rows, msg_cols = edges["rows"], edges["cols"]
    directed = edges["directed"]
    train_unit = edges["train_unit"]
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
    if label_mode == "undirected":
        labels = torch.tensor(
            [1.0 if (min(int(a), int(b)), max(int(a), int(b))) in train_unit else 0.0
             for a, b in pairs.tolist()], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(base.SEED)
    for step in range(AUG_STEPS):
        model.train()
        pick = torch.randint(0, len(pairs), (base.BATCH,), generator=generator)
        out = model(x, *edges["sparse"], ALPHA)
        left, right = pairs[pick, 0], pairs[pick, 1]
        if label_mode == "random":
            flip = torch.from_numpy(
                np.random.default_rng(base.SEED + step).random(len(pick)) < 0.5).to(device)
            left, right = torch.where(flip, right, left), torch.where(flip, left, right)
        target = labels[pick]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            (out["s"][left] * out["t"][right]).sum(-1), target)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            (out["s2"][left] * out["t2"][right]).sum(-1), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(x, *edges["sparse"], ALPHA)["fused"].cpu().numpy().astype(np.float32)


def train_head(state: np.ndarray, train_pos, train_rev, train_neg, device):
    import torch.nn as nn
    torch.manual_seed(base.SEED)
    dim = state.shape[1]
    source = nn.Linear(dim, HEAD_DIM).to(device)
    target = nn.Linear(dim, HEAD_DIM).to(device)
    optimizer = torch.optim.AdamW(list(source.parameters()) + list(target.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(state).to(device)
    triples = [(train_pos, 1.0), (train_rev, 0.0), (train_neg, 0.0)]
    rng = np.random.default_rng(base.SEED)
    for _ in range(HEAD_STEPS):
        source.train(), target.train()
        loss_total = 0.0
        for pairs, label in triples:
            pick = torch.from_numpy(rng.integers(0, len(pairs), 256)).to(device)
            chosen = torch.from_numpy(pairs[pick.cpu().numpy()]).to(device)
            logits = (source(x[chosen[:, 0]]) * target(x[chosen[:, 1]])).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.full_like(logits, label))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss)
    source.eval(), target.eval()
    return source, target


def score(state: np.ndarray, source, target, pairs: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(state).to(device)
        chosen = torch.from_numpy(pairs).to(device)
        logits = (source(x[chosen[:, 0]]) * target(x[chosen[:, 1]])).sum(-1)
    return torch.sigmoid(logits).cpu().numpy()


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
    # the dual model takes (x, out_adj, in_adj, out_deg, in_deg, alpha)
    sparse = base.sparse_adjacency(msg_rows, msg_cols, n_items, device)
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    edges = {"rows": msg_rows, "cols": msg_cols, "directed": directed,
             "train_unit": train_unit, "sparse": sparse}

    rng = np.random.default_rng(base.SEED)
    train_pos = np.stack([msg_rows[rng.permutation(len(msg_rows))[:60000]],
                          msg_cols[rng.permutation(len(msg_rows))[:60000]]], axis=1)
    train_pos = np.stack([msg_rows, msg_cols], axis=1)
    train_pos = train_pos[rng.permutation(len(train_pos))[:60000]]
    train_rev = np.ascontiguousarray(train_pos[:, ::-1])[np.array([(int(b), int(a)) not in directed
                                             for a, b in train_pos.tolist()])]
    train_neg = []
    while len(train_neg) < len(train_rev):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit:
            continue
        train_neg.append((a, b))
    train_neg = np.array(train_neg, dtype=np.int64)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
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
    states = {"A_raw_semantics": semantics}
    for arm, mode in (("B_undirected", "undirected"),
                      ("C_random_direction", "random"),
                      ("D_directed", "directed")):
        states[arm] = train_augmenter(arm, x, edges, n_items, device, mode)
        shift = float(np.linalg.norm(states[arm] - semantics) / max(len(semantics), 1))
        print(f"  trained {arm}, shift {shift:.4f}", flush=True)

    report = {"split": {"message_edges": int(len(msg_rows)), "held_edges": int(len(held_rows)),
                        "eval_positives": int(len(eval_pos)),
                        "eval_negatives": int(len(eval_neg))},
              "protocol": {"augmenter_steps": AUG_STEPS, "head_steps": HEAD_STEPS,
                           "head": "P(A->B)=sigmoid(f_s(x_A).f_t(x_B)), dim 64",
                           "positives": "60,000 training edges",
                           "hard_negatives": "the reverse of those edges",
                           "regular_negatives": "degree-matched unrelated pairs",
                           "metrics": "ROC-AUC (pos vs unrelated), AP, direction accuracy "
                                      "(pos vs its own reverse)"},
              "arms": {}}
    for arm, state in states.items():
        source, target = train_head(state, train_pos, train_rev, train_neg, device)
        pos_score = score(state, source, target, eval_pos, device)
        neg_score = score(state, source, target, eval_neg, device)
        rev_score = score(state, source, target, eval_rev, device)
        auc = float(roc_auc_score(np.concatenate([np.ones(len(pos_score)),
                                                  np.zeros(len(neg_score))]),
                                  np.concatenate([pos_score, neg_score])))
        ap = float(average_precision_score(np.concatenate([np.ones(len(pos_score)),
                                                           np.zeros(len(neg_score))]),
                                           np.concatenate([pos_score, neg_score])))
        direction = float((pos_score > rev_score).mean())
        report["arms"][arm] = {"roc_auc": auc, "average_precision": ap,
                               "direction_accuracy": direction,
                               "mean_positive": float(pos_score.mean()),
                               "mean_reverse": float(rev_score.mean()),
                               "mean_unrelated": float(neg_score.mean())}
        print(f"  {arm}: AUC {auc:.4f} AP {ap:.4f} direction {direction:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    se = 0.5 / (len(eval_pos) ** 0.5)
    report["readings"] = {
        "direction_se": round(se, 4),
        "D_minus_A_direction_sigma": round(
            (report["arms"]["D_directed"]["direction_accuracy"]
             - report["arms"]["A_raw_semantics"]["direction_accuracy"]) / se, 2),
        "D_minus_C_direction_sigma": round(
            (report["arms"]["D_directed"]["direction_accuracy"]
             - report["arms"]["C_random_direction"]["direction_accuracy"]) / se, 2)}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["readings"], indent=1))
    print("written", args.out)


if __name__ == "__main__":
    main()
