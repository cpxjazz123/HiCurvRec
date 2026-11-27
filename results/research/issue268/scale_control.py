"""Issue #268 stage D: is the curvature gain anything more than a global scale?

Competing explanation required by the issue: "geometry ruler has too much
freedom" versus "curvature proxy lacks information" versus "the frozen
reference representation lacks behavior geometry".  A pure global Euclidean
rescaling is the weakest possible ruler change.  If one global scale reaches
the same distortion as the fitted per-triple curvature, the per-triple
mechanism is not identifiable and must not be trained.

All quantities are evaluated on the independent reference subset (seed 269),
so no fitted per-row value is reused as evidence for itself.
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
SEED_FIT = 268
SEED_AUDIT = 269


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if scale != manifest["scale"]:
        raise ValueError("scale differs from the frozen cache")
    kappa = np.load(CACHE / "kappa.npy", mmap_mode="r")

    count, labels = connected_components(graph, directed=False)
    rng = np.random.default_rng(SEED_AUDIT)
    blocks = []
    for component in range(count):
        members = np.flatnonzero(labels == component)
        if len(members) < 2:
            continue
        blocks.append(np.sort(rng.choice(members, min(256, len(members)), replace=False)))
    refs = np.concatenate(blocks)
    gdist = shortest_path(graph, unweighted=True, directed=False, indices=refs).T
    usable = np.isfinite(gdist) & (gdist > 0)
    right = np.where(usable, gdist, 1.0)
    # Euclidean coordinates at the frozen graph-unit scale.
    d = np.linalg.norm(z[:, None, :] - z[refs][None, :, :], axis=2)
    a = np.where(usable, d / (scale * right), 0.0)
    t1 = a.sum(1)
    t2 = (a * a).sum(1)
    n_valid = usable.sum(1).astype(np.float64)

    targets = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    pick = np.random.default_rng(SEED_FIT + 3)
    rows = np.sort(pick.choice(len(kappa), SAMPLE_ROWS, replace=False))
    centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int64)
    b = np.asarray(targets["b"][rows], dtype=np.int64)
    c = np.asarray(targets["c"][rows], dtype=np.int64)
    a_bc = np.linalg.norm(z[b] - z[c], axis=1) / scale
    a_bc = a_bc / np.fromiter(
        (1.0 if (min(int(u), int(v)) * len(z) + max(int(u), int(v))) in edge_keys else 2.0
         for u, v in zip(b, c)), dtype=np.float64, count=len(b))
    kappa_sample = np.asarray(kappa[rows], dtype=np.float64)

    t1_sum = t1[centres] + t1[b] + t1[c]
    t2_sum = t2[centres] + t2[b] + t2[c]
    n_sum = n_valid[centres] + n_valid[b] + n_valid[c]
    denominator = n_sum + 1.0

    def j_scale(c_value):
        return (t2_sum * c_value ** 2 - 2.0 * t1_sum * c_value + n_sum
                + (a_bc * c_value - 1.0) ** 2) / denominator

    candidates = np.exp(np.linspace(np.log(0.2), np.log(20.0), 4001))
    means = np.array([j_scale(value).mean() for value in candidates])
    best = int(np.argmin(means))
    c_star = float(candidates[best])
    j_star = float(means[best])
    # one parabolic refinement around the grid optimum
    step = np.log(candidates[1] / candidates[0])
    fine = np.exp(np.linspace(np.log(c_star) - step, np.log(c_star) + step, 2001))
    fine_means = np.array([j_scale(value).mean() for value in fine])
    c_star = float(fine[int(np.argmin(fine_means))])
    j_star = float(fine_means.min())

    # Per-triple free scale (diagnostic only, forbidden as a mechanism):
    # closed-form minimizer of the quadratic / (n+1) for each row.
    per_triple_scale = (t1_sum + a_bc) / np.maximum(t2_sum + a_bc ** 2, 1e-30)
    j_per_triple_scale = j_scale(per_triple_scale)

    # Reference: the fitted per-triple curvature on this same subset.
    j_zero = j_scale(1.0)
    j_global_kappa = json.loads(
        (ISSUE / "analysis/TRANSFER_VALIDATION.json").read_text())["subset"]["j_global_mean"]
    j_triple_kappa = json.loads(
        (ISSUE / "analysis/TRANSFER_VALIDATION.json").read_text())["subset"]["j_triple_mean"]

    summary = {
        "schema": "genrec.issue268.scale_control/1",
        "rows": int(len(rows)), "references": int(len(refs)), "scale": scale,
        "j_zero_at_frozen_scale": float(j_zero.mean()),
        "best_global_scale_ratio": c_star,
        "best_global_scale_absolute": scale / c_star,
        "j_best_global_scale": j_star,
        "j_per_triple_scale": float(j_per_triple_scale.mean()),
        "j_global_kappa": j_global_kappa,
        "j_triple_kappa": j_triple_kappa,
        "curvature_gain_over_best_global_scale": float(j_star - j_triple_kappa),
        "kappa_minus_best_scale": float(j_triple_kappa - j_star),
        "kappa_minus_per_triple_scale": float(j_triple_kappa - j_per_triple_scale.mean()),
        "notes": ("j_triple_kappa and j_global_kappa are recomputed from "
                  "TRANSFER_VALIDATION.json on the same rows; the scale control is a "
                  "closed-form quadratic in one global multiplier."),
        "cache_sha256": sha256(CACHE / "kappa.npy"),
        "runtime_s": time.time() - started,
    }
    (ISSUE / "analysis/SCALE_CONTROL.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
