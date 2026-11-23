"""Issue #242 step 6: pair relation strength S_G = (S_geo + C_ctx) / 2.

S_geo is the teacher's mixed-geometry similarity, exp(-D_geo / tau) with tau the median
of the finite positive distances seen during training.  C_ctx is the Bhattacharyya
overlap of the *visible graph's* directed transition distributions:

    P_out(i, k) = A_ik / sum_k A_ik,          P_in(i, k) = A_ki / sum_k A_ki
    C_out = sum_k sqrt(P_out(i,k) P_out(j,k)), C_in likewise
    C_ctx = mean over the directions that carry evidence

A direction with no mass is missing evidence, not a negative: it is dropped from the
mean, and a pair with neither direction keeps only S_geo (marked).  Nothing here reads
validation or test interactions.
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

from teacher import (CURVATURES, DEVICE, MixedGeometryTeacher, ego_sample, edge_index,
                     load_all)  # noqa: E402

GRAPH = HERE / "graph"
OUT = HERE / "pairs"
SEED = 242002
PER_GROUP = 12000


def load_teacher():
    blob = torch.load(HERE / "teacher" / "teacher.pt", map_location="cpu")
    out_csr, in_csr, embeddings, structural = load_all()
    model = MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, out_csr, in_csr, embeddings, structural


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    model, out_csr, in_csr, embeddings, structural = load_teacher()
    n = embeddings.shape[0]
    semantic = torch.tensor(embeddings, device=DEVICE)
    structural_t = torch.tensor(structural, device=DEVICE)
    full_x = torch.cat([semantic, structural_t], dim=1)
    edges = edge_index(out_csr)

    with torch.no_grad():
        expert_emb = model.embeddings(full_x, edges)
        descriptors = []
        batch = 512
        for start in range(0, n, batch):
            roots = np.arange(start, min(start + batch, n))
            descriptors.append(model.descriptor.encode_two_scales(
                [int(r) for r in roots], out_csr, in_csr, rng, embeddings, structural))
            if start % 8192 == 0:
                print(f"[descriptor] {start}/{n}", flush=True)
        descriptors = torch.cat(descriptors, dim=0)
        gate = model.weights_for(descriptors)

    # ---- pair bank (same groups the issue names) -----------------------
    out_deg = np.diff(out_csr.indptr)
    in_deg = np.diff(in_csr.indptr)
    coo = out_csr.tocoo()
    edges_np = np.stack([coo.row, coo.col], axis=1)
    edge_set = {tuple(int(v) for v in row) for row in edges_np}
    undirected = edge_set | {(b, a) for a, b in edge_set}

    direct = edges_np[np.random.default_rng(SEED).choice(
        len(edges_np), size=min(PER_GROUP, len(edges_np)), replace=False)]

    two_hop = (out_csr @ out_csr).tocsr()
    shared_out, shared_in, two_hop_pairs, degree_matched = [], [], [], []
    for centre in rng.permutation(n):
        if len(shared_out) >= PER_GROUP and len(shared_in) >= PER_GROUP:
            break
        out_row = out_csr.getrow(int(centre)).indices
        if len(out_row) and len(shared_out) < PER_GROUP:
            j = int(rng.choice(out_row))
            j_row = out_csr.getrow(j).indices
            if len(j_row):
                k = int(rng.choice(j_row))
                if k != int(centre) and (int(centre), k) not in undirected:
                    shared_out.append((int(centre), k))
        in_row = in_csr.getrow(int(centre)).indices
        if len(in_row) and len(shared_in) < PER_GROUP:
            j = int(rng.choice(in_row))
            j_row = in_csr.getrow(j).indices
            if len(j_row):
                k = int(rng.choice(j_row))
                if k != int(centre) and (int(centre), k) not in undirected:
                    shared_in.append((int(centre), k))
        if len(two_hop_pairs) < PER_GROUP:
            reach = two_hop.getrow(int(centre)).indices
            for cand in reach:
                if int(cand) != int(centre) and (int(centre), int(cand)) not in undirected:
                    two_hop_pairs.append((int(centre), int(cand)))
                    break
    buckets = {}
    for node in range(n):
        buckets.setdefault((int(in_deg[node]), int(out_deg[node])), []).append(node)
    keys = [k for k, v in buckets.items() if len(v) >= 2]
    while len(degree_matched) < PER_GROUP and keys:
        bucket = buckets[keys[int(rng.integers(0, len(keys)))]]
        a, b = rng.choice(bucket, size=2, replace=False)
        if (int(a), int(b)) in undirected:
            continue
        degree_matched.append((int(a), int(b)))

    groups = {
        "direct": np.array(direct, dtype=np.int64),
        "shared_out": np.array(shared_out, dtype=np.int64),
        "shared_in": np.array(shared_in, dtype=np.int64),
        "two_hop": np.array(two_hop_pairs, dtype=np.int64),
        "degree_matched": np.array(degree_matched, dtype=np.int64),
    }

    # ---- context strength from the visible graph only -------------------
    def row_dist(csr, node):
        start, end = csr.indptr[node], csr.indptr[node + 1]
        cols = csr.indices[start:end]
        vals = csr.data[start:end].astype(np.float64)
        if len(cols) == 0:
            return None, None
        return cols, vals / vals.sum()

    tau_samples = []
    summaries = {}
    records = {}
    for group, array in groups.items():
        if len(array) == 0:
            records[group] = np.zeros((0, 6), dtype=np.float64)
            summaries[group] = {"pairs": 0}
            continue
        i_idx = torch.tensor(array[:, 0], device=DEVICE)
        j_idx = torch.tensor(array[:, 1], device=DEVICE)
        with torch.no_grad():
            d_geo, per_expert, pair_w = model.distance(
                expert_emb, gate[i_idx], gate[j_idx], i_idx, j_idx)
        d_geo_np = d_geo.detach().cpu().numpy()
        s_out = np.zeros(len(array))
        s_in = np.zeros(len(array))
        valid_out = np.zeros(len(array), dtype=bool)
        valid_in = np.zeros(len(array), dtype=bool)
        for k, (a, b) in enumerate(array):
            cols, p = row_dist(out_csr, int(a))
            if cols is not None:
                cols_b, q = row_dist(out_csr, int(b))
                if cols_b is not None:
                    common, ia, ib = np.intersect1d(cols, cols_b, return_indices=True)
                    s_out[k] = float(np.sqrt(p[ia] * q[ib]).sum())
                    valid_out[k] = True
            cols, p = row_dist(in_csr, int(a))
            if cols is not None:
                cols_b, q = row_dist(in_csr, int(b))
                if cols_b is not None:
                    common, ia, ib = np.intersect1d(cols, cols_b, return_indices=True)
                    s_in[k] = float(np.sqrt(p[ia] * q[ib]).sum())
                    valid_in[k] = True
        valid_any = valid_out | valid_in
        c_ctx = np.where(valid_out & valid_in, 0.5 * (s_out + s_in),
                         np.where(valid_out, s_out, np.where(valid_in, s_in, np.nan)))
        tau_samples.append(d_geo_np[np.isfinite(d_geo_np) & (d_geo_np > 0)])
        records[group] = np.stack([
            array[:, 0], array[:, 1], d_geo_np, s_out, s_in,
            np.where(valid_any, c_ctx, np.nan),
        ], axis=1)
        summaries[group] = {
            "pairs": int(len(array)),
            "both_directions": int((valid_out & valid_in).sum()),
            "neither_direction": int((~valid_any).sum()),
            "d_geo_mean": float(d_geo_np.mean()),
            "d_geo_median": float(np.median(d_geo_np)),
        }

    tau_all = np.concatenate(tau_samples) if tau_samples else np.array([1.0])
    tau = float(np.median(tau_all[tau_all > 0])) if (tau_all > 0).any() else 1.0
    print("tau", tau, flush=True)

    for group, array in records.items():
        if len(array) == 0:
            np.save(OUT / f"bank_{group}.npy", array)
            continue
        s_geo = np.exp(-array[:, 2] / tau)
        c_ctx = array[:, 5]
        has_ctx = np.isfinite(c_ctx)
        s_g = np.where(has_ctx, 0.5 * (s_geo + np.nan_to_num(c_ctx)), s_geo)
        array = np.concatenate([array, s_geo[:, None], s_g[:, None]], axis=1)
        np.save(OUT / f"bank_{group}.npy", array)
        summaries[group].update({
            "s_geo_mean": float(s_geo.mean()),
            "c_ctx_mean": float(np.nanmean(c_ctx)) if has_ctx.any() else None,
            "s_g_mean": float(s_g.mean()),
            "s_g_p10": float(np.quantile(s_g, 0.10)),
            "s_g_median": float(np.median(s_g)),
            "s_g_p90": float(np.quantile(s_g, 0.90)),
            "geometry_only_pairs": int((~has_ctx).sum()),
        })
        print(group, json.dumps(summaries[group]), flush=True)

    torch.save({"gate": gate.detach().cpu()}, OUT / "gate_weights.pt")
    metadata = {
        "schema": "genrec.topo_pairs.issue242.v1",
        "seed": SEED,
        "tau": tau,
        "tau_rule": "median of finite positive D_geo over the frozen bank",
        "s_g": "0.5*(S_geo + C_ctx) when a direction has evidence, else S_geo alone (marked)",
        "groups": {k: int(len(v)) for k, v in groups.items()},
        "summary": summaries,
        "curvatures": list(CURVATURES),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
