"""Issue #241 stage 2: the pair bank and the topological relation strength S_G.

S_G is the Bhattacharyya coefficient between the teacher's decoded context
distributions, averaged over the directions that carry evidence:

    S_out(i, j) = sum_k sqrt(P_out(i, k) * P_out(j, k))
    S_in(i, j)  = sum_k sqrt(P_in(i, k)  * P_in(j, k))

The distributions are over the real item vocabulary, so the score measures shared
purchase context, not role similarity -- the issue's explicit requirement.  A
direction is "valid" when the item has at least one neighbour in that direction in
the ego graph; if neither direction is valid the pair is skipped and counted.

Five groups, balanced by centre item, each frozen before any training:

    direct          a visible transition i -> j
    shared_out      common out-neighbour, no direct edge either way
    shared_in       common in-neighbour, no direct edge either way
    two_hop         i -> x -> j in the ego graph, no direct edge
    degree_matched  similar (in, out) degree, no known relation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from teacher import DEVICE, Teacher, ego_sample, load_graph  # noqa: E402

GRAPH = HERE / "graph"
OUT = HERE / "pairs"
SEED = 241005
PER_GROUP = 12000


def load_teacher():
    checkpoint = torch.load(HERE / "teacher" / "teacher.pt", map_location="cpu")
    model = Teacher(checkpoint["n_items"]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, int(checkpoint["n_items"])


def decode_all(model, out_csr, in_csr, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """P_out and P_in for every item, as sqrt-probabilities (what Bhattacharyya needs)."""
    rng = np.random.default_rng(SEED)
    sqrt_out = np.zeros((n_items, n_items), dtype=np.float32)
    sqrt_in = np.zeros((n_items, n_items), dtype=np.float32)
    batch = 512
    with torch.no_grad():
        for start in range(0, n_items, batch):
            roots = np.arange(start, min(start + batch, n_items))
            samples = [ego_sample(int(r), out_csr, in_csr, rng) for r in roots]
            vectors = model.encode(samples, out_csr, in_csr)
            p_out = torch.softmax(model.head_out(vectors), dim=-1).cpu().numpy()
            p_in = torch.softmax(model.head_in(vectors), dim=-1).cpu().numpy()
            sqrt_out[roots] = np.sqrt(p_out)
            sqrt_in[roots] = np.sqrt(p_in)
            if start % 4096 == 0:
                print(f"[decode] {start}/{n_items}", flush=True)
    return sqrt_out, sqrt_in


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out_csr, in_csr = load_graph()
    n_items = int(out_csr.shape[0])
    model, _ = load_teacher()

    print("[pairs] decoding context distributions", flush=True)
    sqrt_out, sqrt_in = decode_all(model, out_csr, in_csr, n_items)
    np.save(OUT / "sqrt_p_out.npy", sqrt_out)
    np.save(OUT / "sqrt_p_in.npy", sqrt_in)

    out_deg = np.diff(out_csr.indptr)
    in_deg = np.diff(in_csr.indptr)
    has_out = out_deg > 0
    has_in = in_deg > 0
    pairs = set()
    coo = out_csr.tocoo()
    edges = np.stack([coo.row, coo.col], axis=1)
    edge_set = set(map(tuple, edges))
    undirected = {(int(a), int(b)) for a, b in edge_set} | {(int(b), int(a)) for a, b in edge_set}

    # --- direct ---------------------------------------------------------
    pick = rng.choice(len(edges), size=min(PER_GROUP, len(edges)), replace=False)
    direct = edges[pick]

    # --- shared out-neighbour, no direct edge ---------------------------
    two_hop = (out_csr @ out_csr).tocsr()
    shared_out, shared_in, two_hop_pairs = [], [], []
    centres = rng.permutation(n_items)
    for centre in centres:
        if len(shared_out) >= PER_GROUP:
            break
        row_i = out_csr.getrow(int(centre)).indices
        if len(row_i) == 0:
            continue
        partner = int(rng.choice(row_i))
        j_row = out_csr.getrow(partner).indices
        if len(j_row) == 0:
            continue
        k = int(rng.choice(j_row))
        i_row = set(int(x) for x in row_i)
        if k == int(centre) or (int(centre), k) in undirected:
            continue
        if (int(centre), k) in edge_set:
            continue
        shared_out.append((int(centre), k))
        if len(shared_in) < PER_GROUP:
            in_row = in_csr.getrow(int(centre)).indices
            if len(in_row):
                p2 = int(rng.choice(in_row))
                preds = in_csr.getrow(p2).indices
                if len(preds):
                    q = int(rng.choice(preds))
                    if q != int(centre) and (int(centre), q) not in undirected:
                        shared_in.append((int(centre), q))
        if len(two_hop_pairs) < PER_GROUP:
            reach = two_hop.getrow(int(centre)).indices
            for cand in reach:
                if int(cand) != int(centre) and (int(centre), int(cand)) not in undirected:
                    two_hop_pairs.append((int(centre), int(cand)))
                    break

    # --- degree matched -------------------------------------------------
    degree_matched = []
    by_degree = {}
    for node in range(n_items):
        by_degree.setdefault((int(in_deg[node]), int(out_deg[node])), []).append(node)
    keys = list(by_degree)
    while len(degree_matched) < PER_GROUP and keys:
        key = keys[int(rng.integers(0, len(keys)))]
        bucket = by_degree[key]
        if len(bucket) < 2:
            continue
        a, b = rng.choice(bucket, size=2, replace=False)
        if (int(a), int(b)) in undirected:
            continue
        degree_matched.append((int(a), int(b)))

    bank = {
        "direct": np.array(direct, dtype=np.int64),
        "shared_out": np.array(shared_out, dtype=np.int64),
        "shared_in": np.array(shared_in, dtype=np.int64),
        "two_hop": np.array(two_hop_pairs, dtype=np.int64),
        "degree_matched": np.array(degree_matched, dtype=np.int64),
    }

    records = {}
    summary = {}
    for group, array in bank.items():
        if len(array) == 0:
            records[group] = np.zeros((0, 5), dtype=np.float64)
            summary[group] = {"pairs": 0}
            continue
        i, j = array[:, 0], array[:, 1]
        s_out = np.einsum("nd,nd->n", sqrt_out[i], sqrt_out[j])
        s_in = np.einsum("nd,nd->n", sqrt_in[i], sqrt_in[j])
        valid_out = has_out[i] & has_out[j]
        valid_in = has_in[i] & has_in[j]
        valid_any = valid_out | valid_in
        total = np.where(valid_out & valid_in, 0.5 * (s_out + s_in),
                         np.where(valid_out, s_out, np.where(valid_in, s_in, 0.0)))
        records[group] = np.stack([i, j, s_out, s_in, total], axis=1)
        summary[group] = {
            "pairs": int(len(array)),
            "skipped_no_evidence": int((~valid_any).sum()),
            "both_directions_valid": int((valid_out & valid_in).sum()),
            "out_only": int((valid_out & ~valid_in).sum()),
            "in_only": int((~valid_out & valid_in).sum()),
            "s_g_mean": float(total[valid_any].mean()) if valid_any.any() else None,
            "s_g_p10": float(np.quantile(total[valid_any], 0.10)) if valid_any.any() else None,
            "s_g_median": float(np.median(total[valid_any])) if valid_any.any() else None,
            "s_g_p90": float(np.quantile(total[valid_any], 0.90)) if valid_any.any() else None,
            "abs_direction_gap_mean": float(np.abs(s_out - s_in)[valid_out & valid_in].mean())
            if (valid_out & valid_in).any() else None,
        }
        print(group, json.dumps(summary[group]), flush=True)

    for group, array in records.items():
        np.save(OUT / f"bank_{group}.npy", array)
    overlap = {}
    groups = list(bank)
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            left = {tuple(x) for x in bank[groups[a]].tolist()}
            right = {tuple(x) for x in bank[groups[b]].tolist()}
            overlap[f"{groups[a]}__{groups[b]}"] = len(left & right)
    metadata = {
        "schema": "genrec.topo_pairs.issue241.v1",
        "seed": SEED,
        "per_group_target": PER_GROUP,
        "groups": {k: int(len(v)) for k, v in bank.items()},
        "summary": summary,
        "overlap": overlap,
        "validity": {
            "items_with_out_evidence": int(has_out.sum()),
            "items_with_in_evidence": int(has_in.sum()),
            "items_isolated": int((~has_out & ~has_in).sum()),
        },
        "score": "S_G = mean of Bhattacharyya coefficients over the directions with evidence; skipped when neither direction has evidence",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
