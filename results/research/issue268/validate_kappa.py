"""Issue #268 stage C: independent validation of the fitted per-triple curvature.

Three questions, each answered on data the fit did not use to choose a per-row
value:

1. does the fit transfer to an independent reference subset (seed 269)?
2. does it survive a full-reference audit (all component references)?
3. is the per-triple value distinguishable from one global curvature?

A per-triple mechanism is only meaningful if 1-2 hold and 3 says the per-triple
values beat the single global value by more than subset noise.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.graph_infra.assets import load_sparse

ISSUE = REPO / "results/research/issue268"
ART = ISSUE / "artifacts"
CACHE = REPO / "results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/kappa_cache"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
Z_REF = ART / "z_ref.npy"

SAMPLE_ROWS = 20000
AUDIT_ROWS = 512
AUDIT_REFERENCES = 256
SEED_FIT = 268
SEED_AUDIT = 269


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def squared_torch(x, y, k):
    xy = (x - y).square().sum(-1)
    x2 = x.square().sum(-1)
    y2 = y.square().sum(-1)
    magnitude = k.abs().clamp_min(torch.finfo(x.dtype).tiny)
    denominator = (1.0 + k * x2) * (1.0 + k * y2)
    if bool((denominator <= 0).any()):
        raise ValueError("coordinate outside the frozen curvature domain")
    t2 = (magnitude * xy / denominator).clamp_min(0.0)
    sphere = torch.asin(t2.clamp(max=1.0 - 1e-7).sqrt()).square() / magnitude
    hyperbolic = torch.asinh(t2.sqrt()).square() / magnitude
    values = torch.where(k > 0, sphere, torch.where(k < 0, hyperbolic, xy))
    return 4.0 * values


def references(graph, seed, size):
    count, labels = connected_components(graph, directed=False)
    rng = np.random.default_rng(seed)
    blocks = []
    for component in range(count):
        members = np.flatnonzero(labels == component)
        if len(members) < 2:
            continue
        blocks.append(np.sort(rng.choice(members, min(size, len(members)), replace=False)))
    return np.concatenate(blocks), labels


def main() -> None:
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(Z_REF).astype(np.float64)
    graph = load_sparse(GRAPH).tocsr().astype(np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    coo = graph.tocoo()
    upper = coo.row < coo.col
    edge_keys = set((coo.row[upper].astype(np.int64) * len(z)
                     + coo.col[upper].astype(np.int64)).tolist())
    lengths = np.linalg.norm(z[coo.row[upper]] - z[coo.col[upper]], axis=1)
    scale = float(np.median(lengths))
    manifest = json.loads((CACHE / "manifest.json").read_text())
    grid = np.asarray(manifest["grid"], dtype=np.float64)
    if scale != manifest["scale"]:
        raise ValueError("scale differs from the frozen cache")
    kappa = np.load(CACHE / "kappa.npy", mmap_mode="r")
    if len(kappa) != int(np.load(OFFSETS, mmap_mode="r")[-1]):
        raise ValueError("kappa rows do not match the target table")
    if sha256(CACHE / "kappa.npy") != manifest["outputs"]["kappa_sha256"]:
        raise ValueError("kappa cache changed after the fit")

    refs, labels = references(graph, SEED_AUDIT, AUDIT_REFERENCES)
    gdist = shortest_path(graph, unweighted=True, directed=False, indices=refs).T
    usable = np.isfinite(gdist) & (gdist > 0)
    counts = usable.sum(1).astype(np.float32)
    w = torch.from_numpy((z / (2.0 * scale)).astype(np.float32)).to(device)
    wr = w[torch.from_numpy(refs).to(device)]
    gval = torch.from_numpy(np.where(usable, gdist, 1.0).astype(np.float32)).to(device)
    usable_t = torch.from_numpy(usable).to(device)
    nodes = torch.arange(len(z), device=device)
    f_all = np.empty((len(grid), len(z)), dtype=np.float32)
    with torch.no_grad():
        for gindex, k in enumerate(grid):
            kk = torch.as_tensor(float(k), dtype=torch.float32, device=device)
            total = torch.zeros(len(z), device=device)
            for start in range(0, len(z), 512):
                ids = nodes[start:start + 512]
                d2 = squared_torch(w[ids][:, None, :].expand(-1, len(refs), -1), wr[None], kk)
                ratio = d2.clamp_min(0).sqrt() / gval[ids]
                terms = torch.where(usable_t[ids], (ratio - 1.0).square(),
                                    torch.zeros_like(ratio))
                total[ids] = terms.sum(1)
            f_all[gindex] = total.cpu().numpy()

    targets = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    rng = np.random.default_rng(SEED_FIT + 3)
    rows = np.sort(rng.choice(len(kappa), SAMPLE_ROWS, replace=False))
    centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int64)
    b = np.asarray(targets["b"][rows], dtype=np.int64)
    c = np.asarray(targets["c"][rows], dtype=np.int64)
    low = np.minimum(b, c) * len(z) + np.maximum(b, c)
    g_bc = np.fromiter((1.0 if key in edge_keys else 2.0 for key in low.tolist()),
                       dtype=np.float32, count=len(low))
    denom = counts[centres] + counts[b] + counts[c] + 1.0
    kappa_sample = np.asarray(kappa[rows], dtype=np.float64)
    xb = w[torch.from_numpy(b.astype(np.int32)).to(device)]
    xc = w[torch.from_numpy(c.astype(np.int32)).to(device)]
    xy = (xb - xc).square().sum(-1).cpu().numpy().astype(np.float64)
    x2 = xb.square().sum(-1).cpu().numpy().astype(np.float64)
    y2 = xc.square().sum(-1).cpu().numpy().astype(np.float64)

    def pairwise_q(kk, x2v, y2v, xyv):
        if kk == 0:
            return 4.0 * xyv
        magnitude = abs(kk)
        t2 = magnitude * xyv / ((1.0 + kk * x2v) * (1.0 + kk * y2v))
        if kk > 0:
            return 4.0 * np.arcsin(np.sqrt(np.clip(t2, 0.0, 1.0 - 1e-7))) ** 2 / magnitude
        return 4.0 * np.arcsinh(np.sqrt(np.clip(t2, 0.0, None))) ** 2 / magnitude

    def j_for(kappas):
        """Exact J for a per-row curvature vector on the independent subset."""
        values = np.empty(len(rows), dtype=np.float64)
        for kk in np.unique(kappas):
            mask = kappas == kk
            index = int(np.argmin(np.abs(grid - kk)))
            if abs(grid[index] - kk) > 1e-6:
                raise ValueError('kappa value off the frozen grid: %r' % (kk,))
            q = pairwise_q(float(kk), x2[mask], y2[mask], xy[mask])
            values[mask] = (f_all[index][centres[mask]] + f_all[index][b[mask]]
                            + f_all[index][c[mask]] + q) / denom[mask]
        return values

    j_zero = j_for(np.zeros(len(rows)))
    j_triple = j_for(kappa_sample)
    global_k = float(np.median(kappa_sample))
    j_global = j_for(np.full(len(rows), global_k))
    j_floor = j_for(np.full(len(rows), float(grid[0])))

    # Full-reference audit: all component references and the exact g>0 mask.
    audit_rng = np.random.default_rng(SEED_FIT + 4)
    audit_rows = np.sort(audit_rng.choice(len(kappa), AUDIT_ROWS, replace=False))
    audit_centres = (np.searchsorted(offsets, audit_rows, side="right") - 1).astype(np.int64)
    ab = np.asarray(targets["b"][audit_rows], dtype=np.int64)
    ac = np.asarray(targets["c"][audit_rows], dtype=np.int64)
    audit_kappa = np.asarray(kappa[audit_rows], dtype=np.float64)
    audit_global = np.full(len(audit_rows), global_k)
    unique_nodes = np.unique(np.concatenate((audit_centres, ab, ac)))
    full_distance = shortest_path(graph, unweighted=True, directed=False,
                                  indices=unique_nodes)
    position = {int(node): index for index, node in enumerate(unique_nodes)}
    component_labels = labels[audit_centres]
    all_refs = np.flatnonzero(labels == component_labels[0])

    def full_j(kappas):
        totals = np.zeros(len(audit_rows), dtype=np.float64)
        denominators = np.zeros(len(audit_rows), dtype=np.float64)
        with torch.no_grad():
            ref_t = torch.from_numpy(all_refs).to(device)
            ref_w = w[ref_t]
            for role in (audit_centres, ab, ac):
                g_role = full_distance[[position[int(node)] for node in role]]
                usable_role = np.isfinite(g_role) & (g_role > 0)
                g_safe = np.where(usable_role, g_role, 1.0).astype(np.float32)
                for start in range(0, len(audit_rows), 32):
                    ids = np.arange(start, min(start + 32, len(audit_rows)))
                    kk = torch.from_numpy(kappas[ids].astype(np.float32)).to(device)[:, None]
                    node_t = torch.from_numpy(role[ids].astype(np.int64)).to(device)
                    d2 = squared_torch(w[node_t][:, None, :].expand(-1, len(all_refs), -1),
                                       ref_w[None], kk)
                    ratio = d2.clamp_min(0).sqrt() / torch.from_numpy(
                        g_safe[ids][:, all_refs]).to(device)
                    mask = torch.from_numpy(usable_role[ids][:, all_refs]).to(device)
                    terms = torch.where(mask, (ratio - 1.0).square(),
                                        torch.zeros_like(ratio))
                    totals[ids] += terms.sum(1).cpu().numpy()
                    denominators[ids] += usable_role[ids][:, all_refs].sum(1)
            pair_g = np.fromiter(
                (1.0 if (min(int(u), int(v)) * len(z) + max(int(u), int(v))) in edge_keys
                 else 2.0 for u, v in zip(ab, ac)), dtype=np.float32, count=len(ab))
            pair = squared_torch(w[torch.from_numpy(ab.astype(np.int64)).to(device)],
                                 w[torch.from_numpy(ac.astype(np.int64)).to(device)],
                                 torch.from_numpy(kappas.astype(np.float32)).to(device))
            totals += ((pair.clamp_min(0).sqrt()
                        / torch.from_numpy(pair_g).to(device) - 1.0).square()
                       .cpu().numpy())
            denominators += 1.0
        return totals / denominators

    full_zero = full_j(np.zeros(len(audit_rows)))
    full_triple = full_j(audit_kappa)
    full_global = full_j(audit_global)

    improvement = (j_zero - j_triple) / j_zero
    summary = {
        "schema": "genrec.issue268.transfer/1",
        "rows": len(rows), "audit_rows": len(audit_rows),
        "references": int(len(refs)), "audit_references": int(len(all_refs)),
        "scale": scale,
        "subset": {
            "j_zero_mean": float(j_zero.mean()),
            "j_triple_mean": float(j_triple.mean()),
            "j_global_mean": float(j_global.mean()),
            "j_floor_mean": float(j_floor.mean()),
            "improvement_mean": float(improvement.mean()),
            "improvement_median": float(np.median(improvement)),
            "improved_fraction": float((j_triple < j_zero - 1e-12).mean()),
            "global_kappa": global_k,
            "global_improvement_mean": float(((j_zero - j_global) / j_zero).mean()),
            "triple_minus_global_mean": float((j_triple - j_global).mean()),
            "triple_minus_global_median": float(np.median(j_triple - j_global)),
            "triple_beats_global_fraction": float((j_triple < j_global - 1e-12).mean()),
        },
        "full_reference_audit": {
            "j_zero_mean": float(full_zero.mean()),
            "j_triple_mean": float(full_triple.mean()),
            "j_global_mean": float(full_global.mean()),
            "improvement_mean": float(((full_zero - full_triple) / full_zero).mean()),
            "global_improvement_mean": float(((full_zero - full_global) / full_zero).mean()),
            "triple_minus_global_mean": float((full_triple - full_global).mean()),
            "triple_beats_global_fraction": float(
                (full_triple < full_global - 1e-12).mean()),
        },
        "kappa": {
            "mean": float(np.mean(kappa_sample)), "std": float(np.std(kappa_sample)),
            "min": float(kappa_sample.min()), "max": float(kappa_sample.max()),
            "at_floor_fraction": float((kappa_sample <= grid[0] + 1e-9).mean()),
            "at_ceiling_fraction": float((kappa_sample >= grid[-1] - 1e-9).mean()),
            "distinct_values": int(np.unique(kappa_sample).size),
        },
        "cache_sha256": {name: sha256(CACHE / name) for name in
                         ("kappa.npy", "j_zero.npy", "j_fit.npy", "manifest.json")},
        "runtime_s": time.time() - started,
    }
    (ISSUE / "analysis").mkdir(parents=True, exist_ok=True)
    (ISSUE / "analysis/TRANSFER_VALIDATION.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
