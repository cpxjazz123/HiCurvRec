"""Issue #244 §5: freeze the pair bank and its soft target S_G.

    S_geo(i,j) = exp(-D_geo(i,j) / tau)      tau = median of the finite positive distances
                                              over the frozen training candidates
    C_out      = sum_k sqrt(P_out(i,k) P_out(j,k))
    C_in       = sum_k sqrt(P_in(i,k)  P_in(j,k))
    C_ctx      = mean over the directions that carry evidence
    S_G        = (S_geo + C_ctx) / 2 when at least one direction has evidence,
                 S_geo alone otherwise (marked, never imputed as a negative)

Groups follow ``pairs.py``: direct, shared successor, shared predecessor, two hop, and a
degree-matched candidate group that is explicitly NOT a verified negative.  Nothing here
touches validation or test interactions.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pairs as bank  # noqa: E402
import teacher as T  # noqa: E402

GRAPH = HERE.parent / "issue242" / "graph"
OUT = HERE / "pairs"
SEED = 243401
PER_GROUP = 12000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(csr, node: int):
    start, end = csr.indptr[node], csr.indptr[node + 1]
    cols = csr.indices[start:end]
    values = csr.data[start:end].astype(np.float64)
    if len(cols) == 0:
        return None, None
    return cols, values / values.sum()


def bhattacharyya(left, right) -> float | None:
    if left[0] is None or right[0] is None:
        return None
    cols_l, p = left
    cols_r, q = right
    common, il, ir = np.intersect1d(cols_l, cols_r, return_indices=True)
    if len(common) == 0:
        return 0.0
    return float(np.sqrt(p[il] * q[ir]).sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out_csr, in_csr, embeddings, structural = T.load_all()
    n = embeddings.shape[0]
    blob = torch.load(HERE / "teacher" / "teacher.pt", map_location="cpu")
    model = T.MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(T.DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    full_x = torch.cat([torch.tensor(embeddings, device=T.DEVICE),
                        torch.tensor(structural, device=T.DEVICE)], dim=1)
    edges = T.edge_index(out_csr)
    with torch.no_grad():
        expert_emb = model.embeddings(full_x, edges)

    groups = {
        "direct": bank.direct_pairs(out_csr, rng, PER_GROUP),
        "shared_out": bank.shared_successor_pairs(out_csr, n, rng, PER_GROUP),
        "shared_in": bank.shared_predecessor_pairs(in_csr, n, rng, PER_GROUP),
        "two_hop": bank.two_hop_pairs(out_csr, n, rng, PER_GROUP),
        "degree_matched": bank.degree_matched_pairs(out_csr, in_csr, n, rng, PER_GROUP),
    }
    summary = {}
    for name, produced in groups.items():
        summary[name] = bank.group_stats(produced)
        if len(produced):
            bank.verify_group(out_csr, in_csr, produced, name)

    # one teacher pass over every unique node in the bank
    all_pairs = np.concatenate([g for g in groups.values() if len(g)], axis=0)
    unique = np.unique(all_pairs)
    descriptors = model.descriptor.encode_two_scales(
        [int(x) for x in unique], out_csr, in_csr, rng, embeddings, structural)
    with torch.no_grad():
        gate = model.weights_for(descriptors)
    position = {int(node): k for k, node in enumerate(unique)}

    rows, scores, s_geo_all, c_ctx_all, group_ids = [], [], [], [], []
    for group_index, (name, produced) in enumerate(groups.items()):
        if len(produced) == 0:
            continue
        index_i = torch.tensor(produced[:, 0], device=T.DEVICE)
        index_j = torch.tensor(produced[:, 1], device=T.DEVICE)
        weight_i = gate[torch.tensor([position[int(x)] for x in produced[:, 0]], device=T.DEVICE)]
        weight_j = gate[torch.tensor([position[int(x)] for x in produced[:, 1]], device=T.DEVICE)]
        with torch.no_grad():
            d_geo, per_expert, pair_weight = model.distance(
                expert_emb, weight_i, weight_j, index_i, index_j)
        rows.append(produced)
        s_geo_all.append(d_geo.detach().cpu().numpy())
        group_ids.append(np.full(len(produced), group_index, dtype=np.int64))
    d_geo_all = np.concatenate(s_geo_all)
    finite = d_geo_all[np.isfinite(d_geo_all) & (d_geo_all > 0)]
    tau = float(np.median(finite)) if finite.size else 1.0

    c_ctx_all = np.full(len(np.concatenate(rows)), np.nan)
    cursor = 0
    distribution_cache = {}
    for name, produced in groups.items():
        if len(produced) == 0:
            continue
        for k, (i, j) in enumerate(produced):
            i, j = int(i), int(j)
            for node in (i, j):
                if node not in distribution_cache:
                    distribution_cache[node] = (distribution(out_csr, node), distribution(in_csr, node))
            (out_i, in_i), (out_j, in_j) = distribution_cache[i], distribution_cache[j]
            c_out = bhattacharyya(out_i, out_j)
            c_in = bhattacharyya(in_i, in_j)
            parts = [x for x in (c_out, c_in) if x is not None]
            if parts:
                c_ctx_all[cursor + k] = float(np.mean(parts))
        cursor += len(produced)

    s_geo = np.exp(-d_geo_all / tau)
    has_ctx = np.isfinite(c_ctx_all)
    scores = np.where(has_ctx, 0.5 * (s_geo + np.nan_to_num(c_ctx_all)), s_geo)

    payload = {
        "pairs": np.concatenate(rows).astype(np.int64),
        "scores": scores.astype(np.float32),
        "s_geo": s_geo.astype(np.float32),
        "c_ctx": c_ctx_all.astype(np.float32),
        "groups": np.concatenate(group_ids).astype(np.int64),
        "group_names": np.array(list(groups)),
    }
    path = OUT / "bank.npz"
    np.savez_compressed(path, **payload)

    metadata = {
        "schema": "genrec.topo_bank.issue244.v1",
        "seed": SEED,
        "tau": tau,
        "tau_rule": "median of finite positive D_geo over the frozen candidates",
        "target": "S_G = 0.5*(S_geo + C_ctx) when a direction has evidence, else S_geo (marked)",
        "groups": summary,
        "pairs": int(len(payload["pairs"])),
        "geometry_only_pairs": int((~has_ctx).sum()),
        "score_support": {
            "min": float(scores.min()), "median": float(np.median(scores)), "max": float(scores.max()),
        },
        "hashes": {
            "bank.npz": sha256(path),
            "teacher.pt": sha256(HERE / "teacher" / "teacher.pt"),
            "graph": sha256(GRAPH / "metadata.json"),
            "features": sha256(T.FEATURES),
        },
        "curvatures": list(T.CURVATURES),
        "expert_alpha": list(T.EXPERT_ALPHA),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
