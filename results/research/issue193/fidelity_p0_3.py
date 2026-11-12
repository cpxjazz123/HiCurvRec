"""P0-3 of issue #193: is the SID faithful to the #181 teacher, and does the triangle-count defect move that answer?

The teacher is the arm's own ``T = (P + P^2) / 2``, built twice -- once with the shipped edge weights and once with the corrected triangle count -- so
the fidelity table says both what #181 actually achieved and whether the defect was material to it.  The SID comes from the arm's shipped
``item_sids.json``; nothing is retrained.

Metrics per teacher:
  * Pearson and Spearman between the teacher's row weight and the SID's prefix agreement, over sampled anchors and their teacher support;
  * KL(P_row || Q_row) with Q built from the SID's negative distance on the same support;
  * layered means: average teacher weight at each shared-prefix depth (0, 1, 2, 3 levels shared).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARM_DIR = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq"
ANCHORS = 512
MAX_SUPPORT = 4096
SEED = 42


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def train_edges(frame: pd.DataFrame):
    edges = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        edges.add((int(history[-1]), int(target)))
    return sorted(edges)


def degrees_of(edges):
    counts: dict[int, int] = {}
    for i, j in edges:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    return counts


def teacher_weights(coo, n_items, af3_module, with_triangles: bool) -> torch.Tensor:
    degrees = degrees_of(coo)
    if with_triangles:
        neighbours = af3_module.neighbour_sets(coo)
        values = af3_module.af3_values(coo, degrees, neighbours)
    else:
        values = af3_module.af3_values(coo, degrees)
    q05, q95 = af3_module.quantiles(values)
    span = max(q95 - q05, 1e-12)
    rows, cols, vals = [], [], []
    for (i, j), value in values.items():
        u = min(max((value - q05) / span, 0.0), 1.0)
        w = 1.0 + u
        rows.extend((i, j)); cols.extend((j, i)); vals.extend((w, w))
    index = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.tensor(vals, dtype=torch.float64), (n_items, n_items)).coalesce()


def teacher_matrix(w: torch.Tensor) -> torch.Tensor:
    dense = w.to_dense()
    degree = dense.sum(1)
    inv = torch.where(degree > 0, 1.0 / torch.sqrt(degree.clamp_min(1e-30)), torch.zeros_like(degree))
    p = dense * inv.unsqueeze(1) * inv.unsqueeze(0)
    t = 0.5 * (p + p @ p)
    return 0.5 * (t + t.t())


def fiducial(teacher: torch.Tensor, sids: np.ndarray, n_items: int) -> dict:
    rng = np.random.default_rng(SEED)
    anchors = rng.choice(n_items, size=min(ANCHORS, n_items), replace=False)
    anchors = [int(a) for a in anchors if float(teacher[a].abs().sum()) > 0]

    xs, ys, ys_w = [], [], []
    kl_terms, layer_sums, layer_counts = [], np.zeros(4), np.zeros(4)

    for anchor in anchors:
        row = teacher[anchor]
        support = torch.nonzero(row, as_tuple=False).reshape(-1)
        if support.numel() == 0:
            continue
        if support.numel() > MAX_SUPPORT:
            keep = torch.topk(row[support].abs(), MAX_SUPPORT).indices
            support = support[keep]
        weights = row[support].to(torch.float64).numpy()
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            continue
        ids = support.numpy()
        shared = (sids[anchor][None, :] == sids[ids]).sum(axis=1).astype(np.float64)
        agreement = (sids[anchor][None, :] == sids[ids]).mean(axis=1)
        xs.append(weights)
        ys.append(shared)
        ys_w.append(agreement)
        for depth in range(4):
            mask = shared == depth
            if mask.any():
                layer_sums[depth] += float(weights[mask].sum())
                layer_counts[depth] += int(mask.sum())
        # KL(p || q) with q a softmax over the SID agreement on the same support
        p = weights / weights.sum()
        logits = shared * 1.0
        q = np.exp(logits - logits.max())
        q = q / q.sum()
        kl_terms.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    yw = np.concatenate(ys_w) if ys_w else np.array([])

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3:
            return float("nan")
        ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64)
        rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    return {
        "anchors_used": len(kl_terms),
        "pairs": int(x.size),
        "pearson_weight_vs_shared_levels": float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan"),
        "pearson_weight_vs_agreement": float(np.corrcoef(x, yw)[0, 1]) if x.size > 2 else float("nan"),
        "spearman_weight_vs_shared_levels": spearman(x, y),
        "kl_p_teacher_q_sid_mean": float(np.mean(kl_terms)) if kl_terms else float("nan"),
        "mean_teacher_weight_by_shared_levels": {str(d): float(layer_sums[d] / layer_counts[d]) if layer_counts[d] else None
                                                for d in range(4)},
        "support_share_by_shared_levels": {str(d): float(layer_counts[d] / layer_counts.sum()) if layer_counts.sum() else None
                                           for d in range(4)},
    }


def main() -> int:
    sys.path.insert(0, str(ARM_DIR))
    arm = load("arm181", ARM_DIR / "train_rqvae.py")
    shipped = load("af3_shipped", ARM_DIR / "model" / "af3_teacher.py")
    fixed = load("af3_fixed", HERE / "af3_teacher_fixed.py")

    frame = pd.read_parquet(arm.TRAIN_FILE)
    coo = train_edges(frame)
    n_items = int(max(max(edge) for edge in coo)) + 1

    exported = json.loads((RESULT_DIR / "item_sids.json").read_text())
    names = sorted(exported, key=lambda key: int(key))
    sids = np.zeros((n_items, 3), dtype=np.int64)
    for key in names:
        row = int(key)
        if row < n_items:
            sids[row] = exported[key][:3]

    report = {
        "arm": "TIGER_af3_structdistill_rq",
        "items": n_items,
        "edges": len(coo),
        "trained_collision_from_checkpoint": 0.204498312116159,
        "shipped_teacher": fiducial(teacher_matrix(teacher_weights(coo, n_items, shipped, False)), sids, n_items),
        "fixed_teacher": fiducial(teacher_matrix(teacher_weights(coo, n_items, fixed, True)), sids, n_items),
    }
    (HERE / "p0_3_fidelity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
