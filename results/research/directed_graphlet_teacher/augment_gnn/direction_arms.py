#!/usr/bin/env python3
"""Five arms on the dual DiGAE: direction loss, edge weights, attention, role features.

The comparison the plan asks for, on one protocol (same split, seed, budget, probe):

  A  dual directed, mean aggregation, BCE only
  B  A + direction ranking loss  -log sigmoid(s_A.t_B - s_B.t_A) on one-way edges
  C  B + behaviour-count edge weights in the aggregation
  D  C + attention over the neighbours (segment softmax on learned edge scores)
  E  D + neighbour role profile (5 family proportions) appended to the node features

Everything else is held fixed, and the probe is the same seeded, item-isolated direction
head as in `augment_gnn.py`.
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

ROLES = TEACHER / "teacher" / "structure" / "v3" / "structure_targets_v3.npz"
OUT = HERE / "direction_arms.json"
HIDDEN = base.HIDDEN
STEPS = base.STEPS
BATCH = base.BATCH
ALPHA = base.ALPHA
DIR_LAMBDA = 1.0


def build_model(in_dim: int, node_dim: int = HIDDEN, attention: bool = False):
    import torch.nn as nn

    class Arm(nn.Module):
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
                    "norm_t": nn.LayerNorm(node_dim)}) for _ in range(base.LAYERS))
            self.attention = attention
            if attention:
                self.edge_score = nn.Sequential(nn.Linear(2 * node_dim, node_dim), nn.GELU(),
                                                nn.Linear(node_dim, 1))
            self.fuse = nn.Sequential(nn.Linear(2 * node_dim, base.HIDDEN_FUSE), nn.GELU(),
                                      nn.Linear(base.HIDDEN_FUSE, in_dim), nn.Tanh())
            self.source_fused = nn.Linear(in_dim, node_dim)
            self.target_fused = nn.Linear(in_dim, node_dim)

        def aggregate(self, rows, cols, weight, state, node_count):
            """Neighbour mean (optionally weighted, optionally attention-weighted).

            `rows` are the receiving nodes, `cols` the neighbours they read, matching the
            message `state[cols]`.
            """
            if not self.attention:
                if weight is None:
                    return torch.sparse.mm(
                        torch.sparse_coo_tensor(
                            torch.stack([rows, cols]),
                            torch.ones(len(rows), device=rows.device),
                            (node_count, node_count)).coalesce(), state) / \
                        torch.bincount(rows, minlength=node_count).clamp(min=1).unsqueeze(-1)
                return torch.sparse.mm(
                    torch.sparse_coo_tensor(torch.stack([rows, cols]), weight,
                                            (node_count, node_count)).coalesce(), state) / \
                    torch.zeros(node_count, device=rows.device).index_add(
                        0, rows, weight).clamp(min=1e-6).unsqueeze(-1)
            # attention: softmax over each receiver's neighbourhood
            scores = self.edge_score(torch.cat([state[rows], state[cols]], dim=-1)).squeeze(-1)
            scale = scores.max()
            exponent = torch.exp(scores - scale)
            denominator = torch.zeros(node_count, device=rows.device).index_add(
                0, rows, exponent).clamp(min=1e-12)
            alpha = (exponent / denominator[rows]).unsqueeze(-1)
            return torch.zeros_like(state).index_add(0, rows, alpha * state[cols])

        def forward(self, x, edges, node_count, alpha: float):
            s = self.source_in(x)
            t = self.target_in(x)
            out_rows, out_cols, out_w = edges["out"]
            in_rows, in_cols, in_w = edges["in"]
            for layer in self.layers:
                s_msg = self.aggregate(out_rows, out_cols, out_w, t, node_count)
                t_msg = self.aggregate(in_rows, in_cols, in_w, s, node_count)
                s = layer["norm_s"](s + layer["ss"](s) + layer["st"](s_msg))
                t = layer["norm_t"](t + layer["tt"](t) + layer["ts"](t_msg))
            fused = x + alpha * self.fuse(torch.cat([s, t], dim=-1))
            return {"s": s, "t": t, "fused": fused,
                    "s2": self.source_fused(fused), "t2": self.target_fused(fused)}

    return Arm()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(base.FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    weights = graph["weight"].astype(np.float32)
    n_items = int(graph["nodes"])
    semantics = np.load(base.SEMANTICS).astype(np.float32)
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    keep = np.array([(min(a, b), max(a, b)) in train_unit
                     for a, b in zip(rows.tolist(), cols.tolist())])
    msg_w = weights[keep].astype(np.float32)
    edge_index = torch.from_numpy(np.stack([msg_rows, msg_cols]).astype(np.int64)).to(device)
    edge_weight = torch.from_numpy(msg_w).to(device)
    edges = {"out": (edge_index[0], edge_index[1], edge_weight),
             "in": (edge_index[1], edge_index[0], edge_weight)}

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

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]

    roles = np.load(ROLES)["neighbour_family"].astype(np.float32)
    features = {"plain": torch.from_numpy(semantics).to(device),
                "roles": torch.from_numpy(
                    np.concatenate([semantics, roles], axis=1)).to(device)}

    plan = [("A_dual_bce", dict(dir_loss=False, weights=False, attention=False, roles=False)),
            ("B_plus_direction_loss", dict(dir_loss=True, weights=False, attention=False, roles=False)),
            ("C_plus_edge_weights", dict(dir_loss=True, weights=True, attention=False, roles=False)),
            ("D_plus_attention", dict(dir_loss=True, weights=True, attention=True, roles=False)),
            ("E_plus_role_features", dict(dir_loss=True, weights=True, attention=True, roles=True))]
    report = {"split": {"message_edges": int(len(msg_rows)), "held_edges": int(len(held_rows)),
                        "held_one_way_for_probe": int(len(one_way)),
                        "weighted_message_edges_share": float((msg_w > 1).mean())},
              "protocol": {"steps": STEPS, "batch": BATCH, "seed": base.SEED, "alpha": ALPHA,
                           "dir_lambda": DIR_LAMBDA,
                           "direction_loss": "-log sigmoid(s_A.t_B - s_B.t_A) on one-way edges",
                           "probe": "seeded, item-isolated direction head"},
              "arms": {}}
    for name, flags in plan:
        torch.manual_seed(base.SEED)
        x = features["roles" if flags["roles"] else "plain"]
        model = build_model(x.shape[1], attention=flags.get("attention", False)).to(device)
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
            if flags["dir_loss"]:
                # only true one-way edges carry a direction to rank; for a mutual or absent
                # pair the contrast (s_A.t_B - s_B.t_A) has no defined sign
                one_way_rows = (target > 0.5) & torch.from_numpy(
                    np.array([(int(b), int(a)) not in train_unit
                              for a, b in pairs[pick].tolist()])).to(device)
                if bool(one_way_rows.any()):
                    margin = forward_score[one_way_rows] - backward_score[one_way_rows]
                    loss = loss - DIR_LAMBDA * torch.log_softmax(
                        torch.stack([torch.zeros_like(margin), margin], dim=-1), -1)[:, 1].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == STEPS - 1:
                print(f"    {name} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, edges, n_items, ALPHA)
            fused = out["fused"][:, :semantics.shape[1]].cpu().numpy().astype(np.float32)
            pair_t = torch.from_numpy(one_way).to(device)
            head = float((((out["s"][pair_t[:, 0]] * out["t"][pair_t[:, 1]]).sum(-1))
                          > ((out["s"][pair_t[:, 1]] * out["t"][pair_t[:, 0]]).sum(-1))
                          ).float().mean())
        probe = base.probe(fused, one_way, device)
        report["arms"][name] = {"edge_head_direction": head,
                                "shift_l2": float(np.linalg.norm(fused - semantics)
                                                  / max(len(fused), 1)),
                                "probe": probe}
        print(f"  {name}: head {head:.4f} shift "
              f"{report['arms'][name]['shift_l2']:.4f} probe "
              f"{probe['direction_accuracy']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    se = 0.5 / math.sqrt(report["arms"]["A_dual_bce"]["probe"]["pairs_eval"])
    report["readings"] = {
        "se": round(se, 4),
        "B_minus_A_sigma": round((report["arms"]["B_plus_direction_loss"]["probe"]["direction_accuracy"]
                                  - report["arms"]["A_dual_bce"]["probe"]["direction_accuracy"]) / se, 2),
        "E_minus_B_sigma": round((report["arms"]["E_plus_role_features"]["probe"]["direction_accuracy"]
                                  - report["arms"]["B_plus_direction_loss"]["probe"]["direction_accuracy"]) / se, 2)}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
