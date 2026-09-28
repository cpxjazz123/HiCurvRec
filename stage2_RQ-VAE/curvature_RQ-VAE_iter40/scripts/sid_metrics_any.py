"""FORGE-style SID metrics for stage2 output.

按 CLAUDE.md §2 钦点: 接受 SID 路径 + 标签位置参数 (Project Rules §1 例外).
与 stage2_RQ-VAE/.../modules/sid_quality.py 解析口径一致 (HitRate@K=50
+ 3-token Gini + per-layer Gini).

Usage (位置参数, Project Rules §1 例外):
    python3 scripts/sid_metrics_any.py <sid_npy_path> <label>

Prints 4 lines in canonical format:
    HitRate @ K=50: <float>
    Gini (full SID occupancy): <float>
    layer 0 Gini= <float>
    layer 1 Gini= <float>
    layer 2 Gini= <float>
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np


def hitrate_k50(sids: np.ndarray, embeddings: np.ndarray) -> float:
    """近似 RecBole TIGER HitRate@K=50:
       用 item_emb cosine kNN, top-50 内非自身 SID 邻居的占比.
       与 baseline_metrics.md 的 HR 0.4472 对齐.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    emb_norm = embeddings / norms
    sim = emb_norm @ emb_norm.T
    np.fill_diagonal(sim, -np.inf)
    K = min(50, n - 1)
    top_idx = np.argpartition(-sim, kth=K, axis=1)[:, :K]
    hit = 0
    for i in range(n):
        neigh_sids = sids[top_idx[i]]
        own = sids[i]
        same = np.all(neigh_sids == own, axis=1).sum()
        hit += (K - same) / K
    return float(hit / n)


def gini(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    n = counts.size
    if n == 0 or counts.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(
        (2.0 * np.sum(idx * counts) - (n + 1) * counts.sum()) / (n * counts.sum())
    )


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: python3 scripts/sid_metrics_any.py <sid_npy_path> <label>",
            file=sys.stderr,
        )
        return 2

    sid_path = Path(sys.argv[1])
    label = sys.argv[2]
    if not sid_path.is_file():
        print(f"sid file not found: {sid_path}", file=sys.stderr)
        return 3

    sids = np.load(sid_path)
    if sids.ndim != 2 or sids.shape[1] < 3:
        print(f"unexpected SID shape {sids.shape}", file=sys.stderr)
        return 4

    tok3 = sids[:, :3]
    full_counts = Counter(map(tuple, tok3.tolist()))
    full_gini = gini(np.array(list(full_counts.values()), dtype=np.int64))

    per_layer = []
    for layer_idx in range(3):
        layer_vals = tok3[:, layer_idx]
        c = Counter(layer_vals.tolist())
        per_layer.append(gini(np.array(list(c.values()), dtype=np.int64)))

    # Find item embedding for HitRate computation
    emb_path = None
    for candidate in (
        sid_path.parent / "item_emb.npy",
        sid_path.parent.parent / "item_emb.npy",
        Path("/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy"),
    ):
        if candidate.is_file():
            emb_path = candidate
            break

    if emb_path is not None:
        emb = np.load(emb_path).astype(np.float32)
        hr = hitrate_k50(tok3.astype(np.int64), emb)
    else:
        hr = float("nan")

    print(f"[sid_metrics {label}] file={sid_path}")
    print(f"HitRate @ K=50: {hr:.6f}")
    print(f"Gini (full SID occupancy): {full_gini:.6f}")
    for layer_idx, g in enumerate(per_layer):
        print(f"layer {layer_idx} Gini= {g:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
