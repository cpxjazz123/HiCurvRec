#!/usr/bin/env python3
"""Faithful DiGAE reproduction on the behaviour graph, before any 768-d fusion.

The paper keeps two representations per node and predicts edges directly from them:

    s^{l+1}_v = act(W_ss s_v + W_st * sum_{u in Out(v)} t_u / (d_out(v) + 1))
    t^{l+1}_u = act(W_tt t_u + W_ts * sum_{v in In(u)}  s_v / (d_in(u)  + 1))
    P(v -> u) = sigmoid(s^{l+1}_v . t^{l+1}_u)

This script implements exactly that (deterministic variant: the paper's KL term is a
regulariser, not a source of edge signal, and it is noted rather than silently dropped),
then compares two read-outs with identical supervision:

    A  the DiGAE source/target vectors scored directly
    B  the same vectors fused to 768 dimensions and read by a fresh asymmetric head

If A works and B does not, fusion is the problem; if neither works, the behaviour graph or
the task is.
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

OUT = HERE / "diage_repro.json"
STEPS = 1500
BATCH = base.BATCH
DIM = 128
LAYERS = 2


def build_diage(in_dim: int, dim: int = DIM, layers: int = LAYERS):
    import torch.nn as nn

    class DiGAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.source_in = nn.Linear(in_dim, dim)
            self.target_in = nn.Linear(in_dim, dim)
            self.layers = nn.ModuleList(
                nn.ModuleDict({"ss": nn.Linear(dim, dim), "st": nn.Linear(dim, dim),
                               "tt": nn.Linear(dim, dim), "ts": nn.Linear(dim, dim)})
                for _ in range(layers))
            self.act = nn.ELU()
            # a fusion head exists only for read-out B; it is not used by read-out A
            self.fuse = nn.Sequential(nn.Linear(2 * dim, 256), nn.GELU(),
                                      nn.Linear(256, in_dim), nn.Tanh())

        def forward(self, x, out_adj, in_adj, out_norm, in_norm):
            s = self.act(self.source_in(x))
            t = self.act(self.target_in(x))
            for layer in self.layers:
                # DiGAE's directed degree normalisation: divide by (degree + 1)
                s_msg = torch.sparse.mm(out_adj, t) / out_norm
                t_msg = torch.sparse.mm(in_adj, s) / in_norm
                s = self.act(layer["ss"](s) + layer["st"](s_msg))
                t = self.act(layer["tt"](t) + layer["ts"](t_msg))
            return s, t

    return DiGAE()


def head(in_dim: int, dim: int = 64):
    import torch.nn as nn
    return (nn.Linear(in_dim, dim), nn.Linear(in_dim, dim))


def score_head(state, source, target, pairs, device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(state).to(device)
        chosen = torch.from_numpy(pairs).to(device)
    with torch.no_grad():
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
    out_norm = out_deg + 1.0
    in_norm = in_deg + 1.0
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    rng = np.random.default_rng(base.SEED)

    pos = np.stack([msg_rows, msg_cols], axis=1)
    pos = pos[rng.permutation(len(pos))[:80000]]
    rev = np.ascontiguousarray(pos[:, ::-1])[
        np.array([(int(b), int(a)) not in directed for a, b in pos.tolist()])]
    neg = []
    while len(neg) < len(rev):
        a, b = int(rng.integers(0, n_items)), int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit:
            continue
        neg.append((a, b))
    neg = np.array(neg, dtype=np.int64)

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

    torch.manual_seed(base.SEED)
    model = build_diage(semantics.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(base.SEED)
    triples = [(torch.from_numpy(pos).to(device), 1.0),
               (torch.from_numpy(rev).to(device), 0.0),
               (torch.from_numpy(neg).to(device), 0.0)]
    for step in range(STEPS):
        model.train()
        loss_total = 0.0
        for pairs, label in triples:
            pick = torch.randint(0, len(pairs), (BATCH,), generator=generator)
            s, t = model(x, out_adj, in_adj, out_norm, in_norm)
            chosen = pairs[pick]
            logits = (s[chosen[:, 0]] * t[chosen[:, 1]]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.full_like(logits, label))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss)
        if step % 400 == 0 or step == STEPS - 1:
            print(f"    step {step}: loss {loss_total / 3:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        s, t = model(x, out_adj, in_adj, out_norm, in_norm)
        fused = model.fuse(torch.cat([s, t], dim=-1)).cpu().numpy().astype(np.float32)
        s_np, t_np = s.cpu().numpy(), t.cpu().numpy()

    from sklearn.metrics import average_precision_score, roc_auc_score

    def metrics(positive, negative, reverse) -> dict:
        labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
        values = np.concatenate([positive, negative])
        return {"roc_auc": float(roc_auc_score(labels, values)),
                "average_precision": float(average_precision_score(labels, values)),
                "direction_accuracy": float((positive > reverse).mean())}

    def diage_scores(pairs):
        chosen = torch.from_numpy(pairs).to(device)
        with torch.no_grad():
            return (s[chosen[:, 0]] * t[chosen[:, 1]]).sum(-1).cpu().numpy()

    # read-out A: DiGAE's own vectors, no fusion, no extra head
    a = metrics(diage_scores(eval_pos), diage_scores(eval_neg), diage_scores(eval_rev))

    # read-out B: the fused 768-d state with a fresh asymmetric head
    source, target = head(fused.shape[1])
    source, target = source.to(device), target.to(device)
    optimizer = torch.optim.AdamW(list(source.parameters()) + list(target.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    with torch.no_grad():
        xf = torch.from_numpy(fused).to(device)
    for _ in range(STEPS):
        source.train(), target.train()
        for pairs, label in triples:
            pick = torch.randint(0, len(pairs), (BATCH,), generator=generator)
            chosen = pairs[pick]
            logits = (source(xf[chosen[:, 0]]) * target(xf[chosen[:, 1]])).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.full_like(logits, label))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    source.eval(), target.eval()
    b = metrics(score_head(fused, source, target, eval_pos, device),
                score_head(fused, source, target, eval_neg, device),
                score_head(fused, source, target, eval_rev, device))

    report = {"split": {"message_edges": int(len(msg_rows)),
                        "held_edges": int(len(held_rows)),
                        "training_positives": int(len(pos)),
                        "training_reverses": int(len(rev)),
                        "eval_positives": int(len(eval_pos))},
              "protocol": {"propagation": "DiGAE: sum over Out with 1/(d_out+1), sum over In "
                                          "with 1/(d_in+1), ELU, 2 layers, dim 128",
                           "loss": "BCE on positives, reversed edges and random negatives",
                           "kl_term": "omitted (regulariser only; stated, not hidden)",
                           "steps": STEPS, "batch": BATCH, "seed": base.SEED},
              "readouts": {"A_diage_vectors": a, "B_fused_768_head": b}}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for key, value in report["readouts"].items():
        print(f"  {key}: AUC {value['roc_auc']:.4f} AP {value['average_precision']:.4f} "
              f"direction {value['direction_accuracy']:.4f}", flush=True)
    print("written", args.out)


if __name__ == "__main__":
    main()
