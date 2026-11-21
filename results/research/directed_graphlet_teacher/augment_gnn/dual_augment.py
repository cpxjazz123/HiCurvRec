#!/usr/bin/env python3
"""True DiGAE-style dual propagation: source and target representations stay separate.

The shared-hidden variant aggregates in/out neighbours into ONE 128-d vector and only
splits it at the output heads, which may wash out direction before any edge is scored.
This variant keeps two states per node from the first layer onwards, as DiGAE does:

    s^{l+1}_v = norm(s_v + W_ss s_v + W_st * mean_{u in Out(v)} t_u)
    t^{l+1}_u = norm(t_u + W_tt t_u + W_ts * mean_{v in In(u)} s_v)
    P(v->u)   = sigmoid(s_v . t_u)

and fuses the two into the 768-d item vector with a fusion net that is trained, because a
second edge head reads the fused vector.

Same data, split, budget and controls as `augment_gnn.py`, so the two variants are
directly comparable: directed / undirected / randomly-oriented / alpha=0.
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
TEACHER = HERE.parent
sys.path.insert(0, str(HERE))

import augment_gnn as base  # noqa: E402

OUT = HERE / "dual_augment.json"
HIDDEN = base.HIDDEN
LAYERS = base.LAYERS
STEPS = base.STEPS
BATCH = base.BATCH
ALPHA = base.ALPHA


def build_dual_model(in_dim: int, node_dim: int = HIDDEN):
    import torch.nn as nn

    class DualAugmenter(nn.Module):
        """Two states per node, each reading the other role's state of its neighbours."""

        def __init__(self) -> None:
            super().__init__()
            self.source_in = nn.Linear(in_dim, node_dim)
            self.target_in = nn.Linear(in_dim, node_dim)
            self.layers = nn.ModuleList(
                nn.ModuleDict({
                    "ss": nn.Linear(node_dim, node_dim),
                    "st": nn.Linear(node_dim, node_dim),
                    "tt": nn.Linear(node_dim, node_dim),
                    "ts": nn.Linear(node_dim, node_dim),
                    "norm_s": nn.LayerNorm(node_dim),
                    "norm_t": nn.LayerNorm(node_dim)}) for _ in range(LAYERS))
            self.fuse = nn.Sequential(nn.Linear(2 * node_dim, base.HIDDEN_FUSE), nn.GELU(),
                                      nn.Linear(base.HIDDEN_FUSE, in_dim), nn.Tanh())
            self.source_fused = nn.Linear(in_dim, node_dim)
            self.target_fused = nn.Linear(in_dim, node_dim)

        def propagate(self, x, out_adj, in_adj, out_deg, in_deg):
            s = self.source_in(x)
            t = self.target_in(x)
            for layer in self.layers:
                # a source reads its out-neighbours' TARGET states, and vice versa: the two
                # roles never share a vector, which is the whole difference from the shared
                # variant
                s_msg = torch.sparse.mm(out_adj, t) / out_deg
                t_msg = torch.sparse.mm(in_adj, s) / in_deg
                s = layer["norm_s"](s + layer["ss"](s) + layer["st"](s_msg))
                t = layer["norm_t"](t + layer["tt"](t) + layer["ts"](t_msg))
            return s, t

        def forward(self, x, out_adj, in_adj, out_deg, in_deg, alpha: float):
            s, t = self.propagate(x, out_adj, in_adj, out_deg, in_deg)
            fused = x + alpha * self.fuse(torch.cat([s, t], dim=-1))
            return {"s": s, "t": t, "fused": fused,
                    "s2": self.source_fused(fused), "t2": self.target_fused(fused)}

    return DualAugmenter()


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
    undirected_labels = torch.tensor(
        [1.0 if (min(int(a), int(b)), max(int(a), int(b))) in train_unit else 0.0
         for a, b in pairs.tolist()], dtype=torch.float32, device=device)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]

    report = {"inputs": base.__dict__.get("__doc__", "")[:0] or {},
              "variant": "dual (DiGAE-style source/target propagation)",
              "split": {"message_edges": int(len(msg_rows)),
                        "held_edges": int(len(held_rows)),
                        "held_one_way_for_probe": int(len(one_way))},
              "protocol": {"steps": STEPS, "batch": BATCH, "hidden": HIDDEN,
                           "layers": LAYERS, "alpha": ALPHA, "seed": base.SEED,
                           "propagation": "source reads out-neighbours' targets; "
                                          "target reads in-neighbours' sources",
                           "fuse_trained_by": "second edge head on the fused vector",
                           "controls": "undirected labels, randomised orientation, alpha=0"},
              "arms": {}}
    for arm in ("C_directed", "B_undirected", "D_random_direction", "E_alpha_zero"):
        torch.manual_seed(base.SEED)
        model = build_dual_model(semantics.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(base.SEED)
        alpha = 0.0 if arm == "E_alpha_zero" else ALPHA
        target_labels = undirected_labels if arm == "B_undirected" else labels
        for step in range(STEPS):
            model.train()
            pick = torch.randint(0, len(pairs), (BATCH,), generator=generator)
            out = model(x, out_adj, in_adj, out_deg, in_deg, alpha)
            left, right = pairs[pick, 0], pairs[pick, 1]
            if arm == "D_random_direction":
                flip = torch.from_numpy(
                    np.random.default_rng(base.SEED + step).random(len(pick)) < 0.5).to(device)
                left, right = torch.where(flip, right, left), torch.where(flip, left, right)
            target = target_labels[pick]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                (out["s"][left] * out["t"][right]).sum(-1), target)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                (out["s2"][left] * out["t2"][right]).sum(-1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == STEPS - 1:
                print(f"    dual {arm} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, out_adj, in_adj, out_deg, in_deg, alpha)
            fused = out["fused"].cpu().numpy().astype(np.float32)
            pair_t = torch.from_numpy(one_way).to(device)
            forward_score = (out["s"][pair_t[:, 0]] * out["t"][pair_t[:, 1]]).sum(-1)
            backward_score = (out["s"][pair_t[:, 1]] * out["t"][pair_t[:, 0]]).sum(-1)
            head_dir = float((forward_score > backward_score).float().mean())
        probe = base.probe(fused, one_way, device)
        report["arms"][arm] = {"edge_head_direction": head_dir,
                               "shift_l2": float(np.linalg.norm(fused - semantics)
                                                 / max(len(fused), 1)),
                               "probe": probe}
        print(f"  dual {arm}: head {head_dir:.4f} shift "
              f"{report['arms'][arm]['shift_l2']:.4f} probe "
              f"{probe['direction_accuracy']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    se = 0.5 / math.sqrt(report["arms"]["C_directed"]["probe"]["pairs_eval"])
    directed = report["arms"]["C_directed"]["probe"]["direction_accuracy"]
    random_arm = report["arms"]["D_random_direction"]["probe"]["direction_accuracy"]
    report["readings"] = {"se": round(se, 4),
                          "directed_minus_random_sigma": round((directed - random_arm) / se, 2)}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
