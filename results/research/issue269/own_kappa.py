"""Issue #269 own-kappa diagnostic on the trained hard q and encoder z.

The training criterion itself: fixed kappa_t and S*, the A.1 moment formula on a
deterministic 20,000-row sample with the independent reference set (seed 269).
Reports prediction error/ranking against the frozen graph target and the same
rows' distance distortion J(kappa_t) / J(0), all read-only.
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.graph_infra.assets import load_sparse

OUT = ROOT / "results/research/issue269/analysis"
RAW = ROOT / "results/research/issue269/raw"
CACHE = ROOT / "results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284"
GRAPH = ROOT / "assets/train/visible_undirected.npz"
TARGETS = ROOT / "assets/v8_runtime_layout/targets.npy"
OFFSETS = ROOT / "assets/v8_runtime_layout/target_offsets.npy"
ROLES = ROOT / "stage2_RQ-VAE/_archive/TIGER_topology_residual_l2_rq_issue263/role_table.json"
SCALE = 0.06828367045803868
SEED = 269
SAMPLE_ROWS = 20000
REFERENCE_SIZE = 256


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def squared(x, y, k):
    diff = x - y
    xy = (diff * diff).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    kk = k if torch.is_tensor(k) else torch.as_tensor(k, dtype=x.dtype, device=x.device)
    kk = kk.to(x.dtype)
    magnitude = kk.abs().clamp(min=1e-30)
    denominator = (1.0 + kk * x2) * (1.0 + kk * y2)
    if bool((denominator <= 0).any()):
        raise ValueError("coordinate outside the frozen curvature domain")
    t2 = (magnitude * xy / denominator).clamp(min=0.0)
    root = t2.clamp(max=1.0 - 1e-7).sqrt()
    sphere = torch.asin(root).square() / magnitude
    hyperbolic = torch.asinh(root).square() / magnitude
    return 4.0 * torch.where(kk > 0, sphere, torch.where(kk < 0, hyperbolic, xy))


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(ROOT / "results/research/issue268/artifacts/z_ref.npy").astype(np.float64)
    graph = load_sparse(GRAPH).tocsr().astype(np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    coo = graph.tocoo()
    upper = coo.row < coo.col
    edge_keys = set((coo.row[upper].astype(np.int64) * len(z)
                     + coo.col[upper].astype(np.int64)).tolist())
    _, labels = connected_components(graph, directed=False)
    rng = np.random.default_rng(SEED)
    blocks = []
    for component in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == component)
        if len(members) >= 2:
            blocks.append(np.sort(rng.choice(members, min(REFERENCE_SIZE, len(members)), replace=False)))
    refs = np.concatenate(blocks)
    gdist = shortest_path(graph, unweighted=True, directed=False, indices=refs).T
    usable = np.isfinite(gdist) & (gdist > 0)
    gval = np.where(usable, gdist, 1.0).astype(np.float32)
    counts = usable.sum(1).astype(np.float32)

    targets = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    pick = np.random.default_rng(SEED + 1)
    rows = np.sort(pick.choice(int(offsets[-1]), SAMPLE_ROWS, replace=False))
    rows = rows[np.isfinite(np.asarray(targets["target"][rows]))]
    centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int64)
    b = np.asarray(targets["b"][rows], dtype=np.int64)
    c = np.asarray(targets["c"][rows], dtype=np.int64)
    target = np.asarray(targets["target"][rows], dtype=np.float64)
    kappa = np.load(CACHE / "kappa.npy", mmap_mode="r")
    kappa_sample = np.asarray(kappa[rows], dtype=np.float64)
    pair_g = np.fromiter(
        (1.0 if (min(int(u), int(v)) * len(z) + max(int(u), int(v))) in edge_keys else 2.0
         for u, v in zip(b, c)), dtype=np.float32, count=len(b))
    denominator = counts[centres] + counts[b] + counts[c] + 1.0
    usable_t = torch.from_numpy(usable).to(device)
    gval_t = torch.from_numpy(gval).to(device)
    roles = None
    if ROLES.is_file():
        roles = np.asarray(json.loads(ROLES.read_text())["table"], dtype=np.int64)

    summary = {"schema": "genrec.issue269.own_kappa/1", "scale": SCALE,
               "rows": int(len(rows)), "references": int(len(refs)),
               "cache_sha256": sha(CACHE / "kappa.npy"),
               "targets_sha256": sha(TARGETS), "graph_sha256": sha(GRAPH)}
    for space in ("q", "z"):
        blob = np.load(RAW / f"{space}_TIGER_calibrated_kappa_rq.npz", allow_pickle=False)
        values = blob[space].astype(np.float64) / (2.0 * SCALE)
        values_t = torch.from_numpy(values.astype(np.float32)).to(device)
        wr = values_t[torch.from_numpy(refs).to(device)]

        predicted = np.empty(len(rows), dtype=np.float64)
        j_fit = np.empty(len(rows), dtype=np.float64)
        j_zero = np.empty(len(rows), dtype=np.float64)
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            tri = np.stack((centres[start:stop], b[start:stop], c[start:stop]), axis=1)
            k_t = torch.from_numpy(kappa_sample[start:stop].astype(np.float32)).to(device)
            mom = []
            for column in range(3):
                node = torch.from_numpy(tri[:, column].astype(np.int64)).to(device)
                d2 = squared(values_t[node][:, None, :].expand(-1, len(refs), -1),
                             wr[None], k_t[:, None])
                ratio = d2.clamp_min(0).sqrt() / gval_t[node]
                mom.append(torch.where(usable_t[node], (ratio - 1.0).square(),
                                       torch.zeros_like(ratio)).sum(1))
            pair = squared(values_t[torch.from_numpy(b[start:stop].astype(np.int64)).to(device)],
                           values_t[torch.from_numpy(c[start:stop].astype(np.int64)).to(device)], k_t)
            distance = (pair.clamp_min(0).sqrt()
                        / torch.from_numpy(pair_g[start:stop]).to(device) - 1.0).square()
            predicted[start:stop] = (mom[0] - 0.5 * (mom[1] + mom[2]) + 0.25 * pair).cpu().numpy() / denominator[start:stop]
            j_fit[start:stop] = (mom[0].add(mom[1]).add(mom[2]).add(distance) / torch.from_numpy(denominator[start:stop]).to(device)).cpu().numpy()
            # kappa = 0 control on the same coordinates
            zero = torch.zeros_like(k_t)
            mom0 = []
            for column in range(3):
                node = torch.from_numpy(tri[:, column].astype(np.int64)).to(device)
                d2 = squared(values_t[node][:, None, :].expand(-1, len(refs), -1), wr[None], zero[:, None])
                ratio = d2.clamp_min(0).sqrt() / gval_t[node]
                mom0.append(torch.where(usable_t[node], (ratio - 1.0).square(),
                                        torch.zeros_like(ratio)).sum(1))
            pair0 = squared(values_t[torch.from_numpy(b[start:stop].astype(np.int64)).to(device)],
                            values_t[torch.from_numpy(c[start:stop].astype(np.int64)).to(device)], zero)
            distance0 = (pair0.clamp_min(0).sqrt()
                         / torch.from_numpy(pair_g[start:stop]).to(device) - 1.0).square()
            j_zero[start:stop] = (mom0[0].add(mom0[1]).add(mom0[2]).add(distance0)
                                  / torch.from_numpy(denominator[start:stop]).to(device)).cpu().numpy()
        truth_sign = np.sign(target)
        pred_sign = np.sign(predicted)
        entry = {
            "pearson": float(pearsonr(target, predicted).statistic),
            "spearman": float(spearmanr(target, predicted).statistic),
            "mae": float(np.mean(np.abs(predicted - target))),
            "rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
            "sign_accuracy": float((truth_sign == pred_sign).mean()),
            "prediction_mean": float(predicted.mean()),
            "target_mean": float(target.mean()),
            "mean_j_fit": float(j_fit.mean()),
            "mean_j_zero": float(j_zero.mean()),
            "j_improvement_mean": float((j_zero.mean() - j_fit.mean()) / j_zero.mean()),
        }
        if roles is not None:
            entry["role_strata"] = {
                str(int(role)): {
                    "count": int((roles[centres] == role).sum()),
                    "mae": float(np.mean(np.abs(predicted[roles[centres] == role]
                                                 - target[roles[centres] == role])))
                    if (roles[centres] == role).any() else None,
                }
                for role in np.unique(roles[centres])}
        summary[space] = entry
        print(space, json.dumps({k: v for k, v in entry.items() if k != "role_strata"}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "OWN_KAPPA.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT / "OWN_KAPPA.json")


if __name__ == "__main__":
    main()
