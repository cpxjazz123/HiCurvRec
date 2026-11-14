"""AF3 curvature -> structural potential, computed once from the TRAIN split.

The whole file is one mapping, written exactly as `ideas/AF3_curvature_drive_idea/math.json`
states it:

    kappa_ij = 4 - d_i - d_j + 3 * T_ij          (formula `curvature`)
    s_i      = mean_{j in N(i)} kappa_ij          (formula `structural_potential`, d_i = 0 -> mean)
    s_i     <- (s_i - mu_s) / max(sigma_s, eps)   (formula `standardisation`, sigma_s = 0 -> 0)
    H_delta(x) = 0.5 x^2 if |x| <= delta else delta (|x| - 0.5 delta)   (formula `huber`, delta = 1)

The graph is the frozen behaviour graph of the training split, built with the repository's
existing convention (`analysis_sid_recall/graph_shape.py`): one undirected edge for every
pair of consecutive items inside a user sequence, sorted and deduplicated, self-loops
dropped. The sequence of a row is `history ++ [target]`, which is the same follow-pair
convention the #136 AF3 measurement used (`tools/research/science/community_curvature/data.py`
reads `history` and `target` from the split file). Degrees and common-neighbour counts are
taken on that BINARY, undirected, simple graph, so the two sides speak about the same number.

Nothing here trains: the table is computed once and frozen, and every value is a numpy
`float64` until the caller asks for a tensor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from pathlib import Path

#: constants of the mechanism, copied from math.json constants
DELTA = 1.0
EPS = 1e-8
KAPPA_OFFSET = 4.0
TRIANGLE_WEIGHT = 3.0


def repository_root() -> Path:
    """The repository root, found by the `stage2_RQ-VAE` marker (no parent counting)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


def train_parquet() -> Path:
    """The frozen training split the co-occurrence graph is read from."""
    return repository_root() / "results" / "stage0_build_parquet" / "train.parquet"


# --------------------------------------------------------------------------------------
# graph: pairs of consecutive items in a user sequence, undirected, deduplicated
# --------------------------------------------------------------------------------------

def sequences(frame: pd.DataFrame) -> list[np.ndarray]:
    """`history ++ [target]` per row, as integer arrays (rows too short are skipped)."""
    columns = set(frame.columns)
    if "history" not in columns:
        raise ValueError("the split frame has no `history` column")
    has_target = "target" in columns
    result: list[np.ndarray] = []
    for position, history in enumerate(frame["history"]):
        if history is None or len(history) == 0:
            continue
        items = [int(item) for item in history]
        if has_target:
            target = frame["target"].iloc[position]
            if target is not None and not (isinstance(target, float) and np.isnan(target)):
                items.append(int(target))
        if len(items) >= 2:
            result.append(np.asarray(items, dtype=np.int64))
    return result


def edge_arrays(items: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Every undirected consecutive pair once, sorted, self-loops dropped."""
    chunks = []
    for sequence in items:
        if len(sequence) < 2:
            continue
        chunks.append(np.stack([sequence[:-1], sequence[1:]], axis=1))
    if not chunks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def adjacency(rows: np.ndarray, cols: np.ndarray, num_items: int) -> sparse.csr_matrix:
    """Binary, symmetric, simple adjacency over the item id space."""
    if len(rows) == 0:
        return sparse.csr_matrix((num_items, num_items), dtype=np.float64)
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_items, num_items),
    ).tocsr()
    matrix = matrix + matrix.T
    matrix.data[:] = 1.0
    return matrix


def edge_curvatures(
    matrix: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """`kappa_ij = 4 - d_i - d_j + 3 * T_ij` on every given edge (formula `curvature`)."""
    if len(rows) == 0:
        return np.zeros(0, dtype=np.float64)
    degrees = np.asarray(matrix.sum(1)).ravel()
    common = np.asarray((matrix @ matrix)[rows, cols]).ravel()
    return (KAPPA_OFFSET - degrees[rows] - degrees[cols]
            + TRIANGLE_WEIGHT * common).astype(np.float64)


def structural_potentials(
    matrix: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray, curvature: np.ndarray
) -> np.ndarray:
    """`s_i = mean_{j in N(i)} kappa_ij`, degree 0 -> the graph mean (formula `structural_potential`)."""
    num_items = matrix.shape[0]
    totals = np.zeros(num_items, dtype=np.float64)
    counts = np.zeros(num_items, dtype=np.float64)
    np.add.at(totals, rows, curvature)
    np.add.at(totals, cols, curvature)
    np.add.at(counts, rows, 1.0)
    np.add.at(counts, cols, 1.0)
    potentials = np.full(num_items, np.nan, dtype=np.float64)
    has_edges = counts > 0
    potentials[has_edges] = totals[has_edges] / counts[has_edges]
    if not has_edges.any():
        return np.zeros(num_items, dtype=np.float64)
    graph_mean = float(np.mean(potentials[has_edges]))
    potentials[~has_edges] = graph_mean
    return potentials


def standardise(potentials: np.ndarray, eps: float = EPS) -> np.ndarray:
    """`(s - mu_s) / max(sigma_s, eps)`, and 0 when `sigma_s == 0` (formula `standardisation`).

    The standard deviation is the POPULATION one (`ddof=0`): math.json leaves the
    population/sample choice open and asks for one fixed convention, this is it.
    """
    values = np.asarray(potentials, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    mean = float(values.mean())
    sigma = float(values.std(ddof=0))
    if sigma == 0.0:
        return np.zeros_like(values)
    return (values - mean) / max(sigma, eps)


def huber(x, delta: float = DELTA):
    """`H_delta(x)`: `0.5 x^2` for `|x| <= delta`, else `delta (|x| - 0.5 delta)`."""
    import torch

    magnitude = torch.abs(x) if torch.is_tensor(x) else np.abs(x)
    if torch.is_tensor(x):
        quadratic = 0.5 * x * x
        linear = delta * (magnitude - 0.5 * delta)
        return torch.where(magnitude <= delta, quadratic, linear)
    return np.where(magnitude <= delta, 0.5 * x * x, delta * (magnitude - 0.5 * delta))


# --------------------------------------------------------------------------------------
# lambda calibration (formula `lambda_calibration`)
# --------------------------------------------------------------------------------------

def calibrate_lambda(
    distances_to_assigned: np.ndarray,
    potentials: np.ndarray,
    tokens: np.ndarray,
    assigned_potentials: np.ndarray,
    eps: float = EPS,
) -> tuple[float, float]:
    """`den = median_i |s_i - g_token(i)|`; `lambda = median_i d / den`, 0 if `den <= eps`.

    `assigned_potentials[i]` is `g_{token(i)}`, so the caller does not have to know how the
    codeword potentials were grouped. Returns `(den_l, lambda_l)`, both recorded in metrics.
    """
    residual = np.abs(np.asarray(potentials, dtype=np.float64)
                      - np.asarray(assigned_potentials, dtype=np.float64))
    distance = np.asarray(distances_to_assigned, dtype=np.float64)
    if residual.size == 0 or distance.size == 0:
        return 0.0, 0.0
    denominator = float(np.median(residual))
    if denominator <= eps:
        return denominator, 0.0
    return denominator, float(np.median(distance)) / denominator


# --------------------------------------------------------------------------------------
# the whole table
# --------------------------------------------------------------------------------------

def build(
    frame: pd.DataFrame | None = None,
    *,
    num_items: int | None = None,
    eps: float = EPS,
) -> dict:
    """Curvature, potential and the standardised potential over the TRAIN item id space.

    `num_items` is the embedding row count (ids are assumed dense, as in `train_rqvae.py`);
    when it is `None` the largest id seen in the split defines the size.
    """
    if frame is None:
        frame = pd.read_parquet(train_parquet())
    items = sequences(frame)
    rows, cols = edge_arrays(items)
    if num_items is None:
        if len(rows) == 0:
            num_items = 0
        else:
            num_items = int(max(rows.max(), cols.max())) + 1
    matrix = adjacency(rows, cols, int(num_items))
    curvature = edge_curvatures(matrix, rows, cols)
    raw = structural_potentials(matrix, rows, cols, curvature)
    standardised = standardise(raw, eps=eps)
    degrees = np.asarray(matrix.sum(1)).ravel() if num_items else np.zeros(0)
    return {
        "items": items,
        "edges": (rows, cols),
        "adjacency": matrix,
        "degrees": degrees,
        "curvature": curvature,
        "raw_potential": raw,
        "potential": standardised,
        "num_items": int(num_items),
        "metrics": {
            "graph_edges": int(len(rows)),
            "graph_nodes_with_edges": int((degrees > 0).sum()) if len(degrees) else 0,
            "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
            "mean_curvature": float(curvature.mean()) if len(curvature) else 0.0,
            "potential_mean": float(raw.mean()) if raw.size else 0.0,
            "potential_sigma": float(raw.std(ddof=0)) if raw.size else 0.0,
            "standardised_absmax": float(np.abs(standardised).max()) if standardised.size else 0.0,
        },
    }


def potential_tensor(table: dict):
    """The frozen standardised potential as a `float32` tensor (one value per item id)."""
    import torch

    return torch.as_tensor(table["potential"], dtype=torch.float32)
