"""Is the interaction graph tree-like, and can the text embedding reflect it?

Three questions, one probe, on the TRAINING interactions and the SAME item
quadruples so the geometries are comparable.

Q1  Is the behaviour distance tree-like? For a tree metric the two largest of the
    three pairing sums are equal. The relative deviation
    ``(S_max - S_mid) / S_max`` is zero exactly when that holds, and it is
    scale free, which matters because the geometries live at very different
    distance scales.

Q2  Does the text embedding carry the behavioural structure? The rank
    correlation between the text distance and the behaviour distance over item
    pairs, and the canonical correlations between the text space and the
    behavioural space (the truncated SVD of the PPMI co-occurrence).

Q3  Is the hyperbolic geometry better than the Euclidean one at keeping that
    relation? The probe cannot answer it -- the two trained geometries differ in
    more than the geometry -- but it measures what each preserves, so the trained
    models can be compared against the same yardstick.

The geometries:

    behaviour      the cosine distance between the PPMI-weighted co-occurrence
                   profiles of the items, from the training interactions only
    behaviour_raw  the same without the PPMI weighting, as a second view
    hyperbolic     the frozen Stage 2 behaviour reference, in the curvature-1 ball
    text           the Stage 1 sentence-T5 embedding, Euclidean
    null           the text embedding with the items permuted: the distance
                   distribution is preserved, the structure is destroyed

The null is what makes the four-point numbers readable: a geometry that is not
tree-like should not look different from its own permuted copy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "behaviour_structure_probe.json"
PARQUET = REPO / "results" / "stage0_build_parquet"
SEED = 42
QUADRUPLES = 20000
PAIRS = 200000
N_ITEMS = 24587


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cooccurrence() -> csr_matrix:
    import pandas as pd

    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    rows: list[int] = []
    columns: list[int] = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for first in sequence:
            for second in sequence:
                if first != second:
                    rows.append(first)
                    columns.append(second)
    matrix = csr_matrix((np.ones(len(rows), dtype=np.float32),
                          (np.asarray(rows), np.asarray(columns))),
                         shape=(N_ITEMS, N_ITEMS))
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def ppmi(matrix: csr_matrix) -> csr_matrix:
    """The positive pointwise mutual information of the co-occurrence."""
    total = matrix.sum()
    row = np.asarray(matrix.sum(axis=1)).ravel()
    column = np.asarray(matrix.sum(axis=0)).ravel()
    coo = matrix.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log((coo.data * total) / np.maximum(row[coo.row] * column[coo.col], 1e-12))
    value = np.maximum(value, 0.0)
    return csr_matrix((value.astype(np.float32), (coo.row, coo.col)), shape=matrix.shape)


def relative_delta(sums: np.ndarray) -> np.ndarray:
    """``(S_max - S_mid) / S_max``: zero exactly when the four-point condition holds."""
    ordered = np.sort(sums, axis=0)
    largest = np.maximum(ordered[2], 1e-12)
    return (ordered[2] - ordered[1]) / largest


def pairing_sums(d_ab, d_cd, d_ac, d_bd, d_ad, d_bc) -> np.ndarray:
    return np.stack([d_ab + d_cd, d_ac + d_bd, d_ad + d_bc], axis=0)


def canonical_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The canonical correlations of two centred spaces, from their bases."""
    u_left, _, _ = np.linalg.svd(left, full_matrices=False)
    u_right, _, _ = np.linalg.svd(right, full_matrices=False)
    overlap = u_left.T @ u_right
    return np.linalg.svd(overlap, compute_uv=False)


def main() -> int:
    from tools.research.science import sid_metric_bench as bench

    module = _load("four_point", HERE / "metrics_four_point_svd.py")
    context = bench.load_context()
    target = np.asarray(context["target"], dtype=np.float64)
    text = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(np.float64)

    print("building the co-occurrence from the training interactions", flush=True)
    counts = cooccurrence()
    weighted = ppmi(counts)

    def norms_of(matrix):
        return np.maximum(np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()), 1e-12)

    ppmi_norm = norms_of(weighted)
    raw_norm = norms_of(counts)

    def profile_distance(matrix, norms, left, right):
        dot = np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()
        return 1.0 - dot / (norms[left] * norms[right])

    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, N_ITEMS, (QUADRUPLES, 4))
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    unit_text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    permuted = rng.permutation(N_ITEMS)
    unit_null = unit_text[permuted]

    def hyperbolic_distance(left, right):
        return module.hyperbolic_distance(target[left], target[right], 1.0)

    def text_distance(left, right):
        return np.linalg.norm(unit_text[left] - unit_text[right], axis=1)

    def null_distance(left, right):
        return np.linalg.norm(unit_null[left] - unit_null[right], axis=1)

    geometries = {
        "behaviour_ppmi": lambda l, r: profile_distance(weighted, ppmi_norm, l, r),
        "behaviour_raw": lambda l, r: profile_distance(counts, raw_norm, l, r),
        "hyperbolic": hyperbolic_distance,
        "text": text_distance,
        "null_permuted_text": null_distance,
    }
    report = {"quadruples": QUADRUPLES, "four_point": {}, "pairwise": {}}

    for name, distance in geometries.items():
        sums = pairing_sums(distance(a, b), distance(c, d), distance(a, c),
                            distance(b, d), distance(a, d), distance(b, c))
        delta = relative_delta(sums)
        report["four_point"][name] = {
            "median_relative_delta": float(np.median(delta)),
            "p10": float(np.percentile(delta, 10)),
            "p90": float(np.percentile(delta, 90)),
            "share_below_0.01": float(np.mean(delta < 0.01)),
            "share_below_0.05": float(np.mean(delta < 0.05)),
        }
        print("%-20s median rel. delta %.4f | share < 0.01: %.3f" % (
            name, report["four_point"][name]["median_relative_delta"],
            report["four_point"][name]["share_below_0.01"]), flush=True)

    left = rng.integers(0, N_ITEMS, PAIRS)
    right = rng.integers(0, N_ITEMS, PAIRS)
    keep = left != right
    left, right = left[keep], right[keep]
    behaviour_distance = profile_distance(weighted, ppmi_norm, left, right)
    for name, values in (("text", text_distance(left, right)),
                         ("hyperbolic", hyperbolic_distance(left, right)),
                         ("null_permuted_text", null_distance(left, right)),
                         ("behaviour_raw", profile_distance(counts, raw_norm, left, right))):
        rho, p = stats.spearmanr(values, behaviour_distance)
        report["pairwise"][name] = {"spearman_vs_behaviour_ppmi": round(float(rho), 6),
                                    "p_value": float(p)}
        print("%-20s spearman with the behaviour distance: %+.4f" % (name, rho), flush=True)

    # the canonical correlations between the text space and the behavioural space
    u, s, _ = svds(weighted.astype(np.float64), k=64, random_state=SEED)
    behaviour = u * s
    text_centre = text - text.mean(axis=0)
    behaviour_centre = behaviour - behaviour.mean(axis=0)
    correlations = canonical_correlations(text_centre, behaviour_centre)
    report["canonical_correlations"] = {
        "top": [round(float(value), 6) for value in correlations[:8]],
        "mean": round(float(correlations.mean()), 6),
        "total_explained_fraction": round(float((correlations ** 2).sum() / len(correlations)), 6),
    }
    print("canonical correlations (text vs behaviour):",
          report["canonical_correlations"]["top"], flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
