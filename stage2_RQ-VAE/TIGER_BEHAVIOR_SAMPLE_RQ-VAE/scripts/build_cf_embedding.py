"""Build the frozen item-side collaborative-filtering representation for E2.

LETTER's original recipe scores the quantised representation against a per-item CF
table it loads from disk. That table was never produced in this repository -- the
hard-coded path in LETTER_RQ-VAE resolves to nothing -- so E2 has to build its own
before it can test whether a genuinely independent collaborative signal can guide
the codebooks. This is that builder.

The representation is the textbook item-based one, computed from TRAIN interactions
only so that no validation or test signal can leak into Stage 2:

    user-item incidence X (users x items, binarised)
      -> item-item co-occurrence C = X' X
      -> positive pointwise mutual information
      -> truncated SVD to CF_DIM components
      -> L2-normalised rows

Index space: rows are RAW item ids, 0..n_items-1, matching ``sentence_t5.npy`` and
``item_sids.json``. That is also the space the trainer indexes when it builds a
batch with ``all_embeddings[ids]``, which is what the CF loss will index.

Everything is hard-coded per CLAUDE.md section 1 and deterministic: the SVD is
seeded and the incidence matrix is built by pandas explode plus sparse
accumulation, not by a per-user Python loop.

Writes one file and prints its summary. No report file is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TRAIN_FILE = ROOT / "results/stage0_build_parquet/train.parquet"
EMBEDDINGS_PATH = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
OUTPUT_PATH = (
    ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/cf_item_emb.npy"
)

CF_DIM = 64
SVD_SEED = 42
SVD_ITER = 7


def incidence(frame: pd.DataFrame, n_items: int) -> csr_matrix:
    """Binarised user x item incidence over history items and the target."""
    history = frame["seen_history"].explode()
    # An empty history explodes to NaN; those users still contribute their target.
    present = history.notna()
    history = history[present]
    rows = history.index.to_numpy(dtype=np.int64)
    cols = history.to_numpy(dtype=np.int64)
    rows = np.concatenate((rows, np.arange(len(frame), dtype=rows.dtype)))
    cols = np.concatenate((cols, frame["target"].to_numpy()))
    counts = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(frame), n_items),
    )
    counts.sum_duplicates()
    counts.data[:] = 1.0
    return counts


def ppmi(co_occurrence: csr_matrix) -> csr_matrix:
    """Positive pointwise mutual information with the self term dropped."""
    matrix = co_occurrence.tocoo()
    total = float(matrix.data.sum())
    row_sums = np.asarray(co_occurrence.sum(axis=1)).ravel().astype(np.float64)
    row = matrix.row
    col = matrix.col
    value = matrix.data.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((value * total) / (row_sums[row] * row_sums[col]))
    keep = (row != col) & np.isfinite(pmi) & (pmi > 0.0)
    return coo_matrix(
        (pmi[keep].astype(np.float32), (row[keep], col[keep])),
        shape=co_occurrence.shape,
    ).tocsr()


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    n_items = int(embeddings.shape[0])
    frame = pd.read_parquet(TRAIN_FILE, columns=["seen_history", "target"])

    observed = incidence(frame, n_items)
    co_occurrence = (observed.T @ observed).tocsr()
    weighted = ppmi(co_occurrence)

    svd = TruncatedSVD(n_components=CF_DIM, random_state=SVD_SEED, n_iter=SVD_ITER)
    components = svd.fit_transform(weighted).astype(np.float32)
    norms = np.linalg.norm(components, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    components = components / norms

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, components)

    covered = int((np.asarray(observed.sum(axis=0)).ravel() > 0).sum())
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "shape": list(components.shape),
                "dtype": str(components.dtype),
                "users": int(observed.shape[0]),
                "items": n_items,
                "items_with_interactions": covered,
                "items_without_interactions": n_items - covered,
                "incidence_nnz": int(observed.nnz),
                "cooccurrence_nnz": int(co_occurrence.nnz),
                "ppmi_nnz": int(weighted.nnz),
                "explained_variance_ratio_sum": float(
                    svd.explained_variance_ratio_.sum()
                ),
                "cf_dim": CF_DIM,
                "svd_seed": SVD_SEED,
                "row_norm_min": float(np.linalg.norm(components, axis=1).min()),
                "row_norm_max": float(np.linalg.norm(components, axis=1).max()),
                "zero_rows": int((np.linalg.norm(components, axis=1) == 0.0).sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
