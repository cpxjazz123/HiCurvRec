"""AF3 curvature-driven component share on a fixed product manifold.

The mechanism is one per-item scalar: the rank ``u_i`` of the AF3 structural
potential (neighbourhood mean of ``kappa_ij = 4 - d_i - d_j + 3 T_ij``). It says
*how much* of the two hyperbolic factors a codeword spends on the high-curvature
ball, and it acts on the CODEWORD side only:

    p_i    = u_i                                (share, in (0, 1))
    s_hi   = sqrt(p_i),  s_lo = sqrt(1 - p_i)
    v_k_hi = expmap0(sqrt(p_i) * tau_k^h1, c_hi)
    v_k_lo = expmap0(sqrt(1-p_i) * tau_k^h2, c_lo)
    x_hi   = expmap0(x^h1, c_hi),  x_lo = expmap0(x^h2, c_lo)   (item side: no scaling)

    D_ik = ||x_E - tau_k^E||^2
           + SCALE_HYP * (d_{c_hi}(x_hi, v_k_hi)^2 + d_{c_lo}(x_lo, v_k_lo)^2)

The product manifold is ``H^8(c_hi) x H^8(c_lo) x R^16`` with both curvatures
FIXED; AF3 only moves the share ``p_i``, never a curvature. The two ball operators
are implemented here from `tools/research/geometry/poincare_contract.py`
(``expmap0`` at :85, squared geodesic distance at :141/:153), including the
``c == 0`` branch, the ``||v|| -> 0`` tanhc limit and the ``(1-BALL_EPS)/sqrt(c)``
projection. Nothing is imported from a sibling arm.

The AF3 table (train split only, frozen before training) is built by the six
functions vendored verbatim from
``ideas/AF3_curvature_factor_mass_idea/reference_af3_potential.py`` and cached,
atomically, at ``eval/af3_factor_mass_p.npz``.

Assignments, the residual recursion, the quantiser and the losses are untouched:
the only thing that changes is the numeric value of ``D`` that they are fed, so
formulas `assignment`, `untouched_quantization` and `surfaces_excluded` keep their
baseline meaning (``model/layers.py`` is byte-identical to the frozen baseline).
Formulas implemented here: `af3_potential`, `share_map`, `latent_split`,
`energy_split`, `product_cost`, `euclidean_degeneracy`, `curvature_is_fixed_ladder`,
`no_pair_signal` (the cost carries no (i, j) term), `dedup_archive` (share lives on
the codeword side, item side unscaled) and `surfaces_excluded`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch

NAME = "euclidean"
DESCRIPTION = (
    "Euclidean on the 16-d block plus two geodesic squared distances on "
    "H^8(c_hi) and H^8(c_lo), codeword scaled per item by sqrt(p_i)/sqrt(1-p_i)"
)

#: The mechanism this module implements (read by the registry / tests).
MECHANISM = "af3_factor_mass"
AF3_FACTOR_MASS = True

# --------------------------------------------------------------------------------------
# frozen constants (math.json constants, not knobs)
# --------------------------------------------------------------------------------------

#: Euclidean factor dimension (the first 16 columns).
D_E = 16
#: High-curvature hyperbolic factor dimension (columns 16..24).
D_H1 = 8
#: Low-curvature hyperbolic factor dimension (columns 24..32).
D_H2 = 8
#: Total codeword dimension, identical to the frozen baseline codebook.
D_TOTAL = D_E + D_H1 + D_H2
#: Curvature of the high-curvature ball (radius 1/sqrt(c) = 0.5).
C_HI = 4.0
#: Curvature of the low-curvature ball (radius 1/sqrt(c) = 2).
C_LO = 0.25
#: Fixed scale shared by both hyperbolic factors; d_c^2 -> 4|x-y|^2 as c -> 0.
SCALE_HYP = 0.25
#: The constant share used by the ``behaviour_off`` wiring check.
SHARE_REF = 0.5

#: Lower-case aliases of the math.json constant names (read-only mirrors).
c_hi = C_HI
c_lo = C_LO

#: The frozen Sinkhorn temperature / iteration count from config/train_config.json.
#: The quantiser owns their use; they are recorded here so a test can compare them.
SK_EPSILON = 0.003
SK_ITERS = 50

#: How close to the boundary a projected point may come (poincare_contract.py).
BALL_EPS = 1e-6
#: The clip on the ``artanh`` argument (poincare_contract.py).
ARTANH_EPS = 1e-7

#: Where the frozen per-item share table is cached (relative to the training cwd).
P_TABLE_CACHE = Path("eval") / "af3_factor_mass_p.npz"

# --------------------------------------------------------------------------------------
# wiring switches (tests flip them; training never does)
# --------------------------------------------------------------------------------------

#: ``behaviour_off``: p_i is the constant SHARE_REF; the geometry is still there.
BEHAVIOUR_OFF = False
#: ``geometry_off``: return the baseline 32-d expansion, element for element.
GEOMETRY_OFF = False


def switches() -> dict:
    """The two wiring switches, for a test or a readout (both are off in training)."""
    return {"behaviour_off": bool(BEHAVIOUR_OFF), "geometry_off": bool(GEOMETRY_OFF)}


def set_switches(*, behaviour_off=None, geometry_off=None) -> dict:
    """Set the switches by name (``None`` leaves one alone) and return the new state."""
    global BEHAVIOUR_OFF, GEOMETRY_OFF
    if behaviour_off is not None:
        BEHAVIOUR_OFF = bool(behaviour_off)
    if geometry_off is not None:
        GEOMETRY_OFF = bool(geometry_off)
    return switches()


# --------------------------------------------------------------------------------------
# AF3 table: vendored from ideas/AF3_curvature_factor_mass_idea/reference_af3_potential.py
# (the six functions below are byte-for-byte copies; behaviour must not move)
# --------------------------------------------------------------------------------------

#: constants of the AF3 table, copied from the reference
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


# --------------------------------------------------------------------------------------
# per-item share p_i = u_i = (rank(kappa_bar_i) - 0.5) / N
# --------------------------------------------------------------------------------------

def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ascending ranks 1..N with ties taking their average rank."""
    values = np.asarray(values, dtype=np.float64)
    count = values.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def shares_from_potentials(potentials: np.ndarray) -> np.ndarray:
    """`u_i = (rank(kappa_bar_i) - 0.5) / N`, in (0, 1) (formula `share_map`)."""
    potentials = np.asarray(potentials, dtype=np.float64)
    count = potentials.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    ranks = average_ranks(potentials)
    shares = (ranks - 0.5) / float(count)
    if not np.isfinite(shares).all():
        raise ValueError("the AF3 share table contains non-finite values")
    return shares


def _split_digest(path: Path) -> str:
    """SHA-256 of the split file, so a cache built from another split is refused."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_p_table(frame: pd.DataFrame | None = None, *, digest: str | None = None) -> dict:
    """Build the frozen table: `kappa_bar_i`, its rank share `p_i`, and its digest.

    Formula `af3_potential`: `kappa_ij = 4 - d_i - d_j + 3 T_ij`, then the
    neighbourhood mean, then the global rank. Built from the TRAIN split only.
    """
    source = train_parquet()
    if frame is None:
        if not source.is_file():
            raise FileNotFoundError("the training split is missing: " + str(source))
        frame = pd.read_parquet(source)
    if digest is None:
        digest = _split_digest(source) if source.is_file() else ""
    items = sequences(frame)
    rows, cols = edge_arrays(items)
    if len(rows) == 0:
        num_items = 0
    else:
        num_items = int(max(rows.max(), cols.max())) + 1
    matrix = adjacency(rows, cols, num_items)
    curvature = edge_curvatures(matrix, rows, cols)
    kappa_bar = structural_potentials(matrix, rows, cols, curvature)
    shares = shares_from_potentials(kappa_bar)
    degrees = np.asarray(matrix.sum(1)).ravel() if num_items else np.zeros(0)
    return {
        "item_ids": np.arange(num_items, dtype=np.int64),
        "p": shares,
        "kappa_bar": np.asarray(kappa_bar, dtype=np.float64),
        "degrees": np.asarray(degrees, dtype=np.float64),
        "num_items": int(num_items),
        "digest": str(digest),
        "metrics": {
            "graph_edges": int(len(rows)),
            "graph_nodes_with_edges": int((degrees > 0).sum()) if len(degrees) else 0,
            "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
            "mean_curvature": float(curvature.mean()) if len(curvature) else 0.0,
            "share_mean": float(shares.mean()) if shares.size else 0.0,
            "share_min": float(shares.min()) if shares.size else 0.0,
            "share_max": float(shares.max()) if shares.size else 0.0,
        },
    }


def write_p_table(table: dict, path: Path | None = None) -> Path:
    """Write the table atomically: a `.tmp` file, then `os.replace`."""
    target = Path(path) if path is not None else Path(P_TABLE_CACHE)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            item_ids=np.asarray(table["item_ids"], dtype=np.int64),
            p=np.asarray(table["p"], dtype=np.float64),
            kappa_bar=np.asarray(table["kappa_bar"], dtype=np.float64),
            degrees=np.asarray(table["degrees"], dtype=np.float64),
            num_items=np.int64(table["num_items"]),
            digest=np.asarray(str(table["digest"])),
        )
    os.replace(temporary, target)
    return target


def read_p_table(path: Path | None = None) -> dict:
    """Load a cached table and verify its digest and item coverage (fail loud)."""
    target = Path(path) if path is not None else Path(P_TABLE_CACHE)
    with np.load(target, allow_pickle=False) as payload:
        item_ids = np.asarray(payload["item_ids"], dtype=np.int64)
        shares = np.asarray(payload["p"], dtype=np.float64)
        kappa_bar = np.asarray(payload["kappa_bar"], dtype=np.float64)
        degrees = np.asarray(payload["degrees"], dtype=np.float64)
        num_items = int(payload["num_items"])
        digest = str(payload["digest"])
    source = train_parquet()
    if not source.is_file():
        raise FileNotFoundError("the training split is missing: " + str(source))
    current = _split_digest(source)
    if digest != current:
        raise ValueError(
            "the cached AF3 table at %s was built from another split "
            "(digest %s, current %s)" % (target, digest, current))
    if num_items != shares.size or item_ids.size != shares.size:
        raise ValueError(
            "the cached AF3 table at %s has inconsistent coverage "
            "(num_items=%d, item_ids=%d, p=%d)"
            % (target, num_items, item_ids.size, shares.size))
    if not np.isfinite(shares).all() or shares.min() < 0.0 or shares.max() > 1.0:
        raise ValueError("the cached AF3 share table is not in [0, 1]: " + str(target))
    return {
        "item_ids": item_ids,
        "p": shares,
        "kappa_bar": kappa_bar,
        "degrees": degrees,
        "num_items": num_items,
        "digest": digest,
    }


#: Process-local memo of the frozen table, so the split is hashed/read once.
_TABLE: dict | None = None


def table(path: Path | None = None) -> dict:
    """The frozen table: read the cache when it validates, build and cache it otherwise."""
    global _TABLE
    target = Path(path) if path is not None else Path(P_TABLE_CACHE)
    if _TABLE is not None and _TABLE.get("_path") == str(target):
        return _TABLE
    if target.is_file():
        loaded = read_p_table(target)
    else:
        built = build_p_table()
        write_p_table(built, target)
        loaded = dict(built)
    loaded["_path"] = str(target)
    _TABLE = loaded
    return _TABLE


def shares_for(item_ids) -> np.ndarray:
    """The frozen share `p_i` for these item ids; missing or out-of-range ids raise."""
    if item_ids is None:
        raise ValueError(
            "af3_factor_mass needs the `item_id` field; the host published none")
    if torch.is_tensor(item_ids):
        ids = item_ids.detach().to("cpu", torch.long).numpy()
    else:
        ids = np.asarray(item_ids, dtype=np.int64)
    ids = np.atleast_1d(ids)
    frozen = table()
    shares = frozen["p"]
    if ids.size and (ids.min() < 0 or ids.max() >= shares.size):
        raise ValueError(
            "item id out of range for the AF3 share table: [%d, %d] vs size %d"
            % (ids.min(), ids.max(), shares.size))
    return shares[ids]


def readouts() -> dict:
    """Diagnostics for the frozen table and the current switch state."""
    frozen = table()
    shares = frozen["p"]
    percentiles = {}
    if shares.size:
        for name, q in (("p05", 5.0), ("p50", 50.0), ("p95", 95.0)):
            percentiles[name] = float(np.percentile(shares, q))
    rank_correlation = 0.0
    kappa_bar = frozen["kappa_bar"]
    if shares.size > 1 and np.std(kappa_bar) > 0.0 and np.std(shares) > 0.0:
        rank_correlation = float(np.corrcoef(average_ranks(kappa_bar),
                                             average_ranks(shares))[0, 1])
    return {
        "share_min": float(shares.min()) if shares.size else 0.0,
        "share_max": float(shares.max()) if shares.size else 0.0,
        "share_mean": float(shares.mean()) if shares.size else 0.0,
        "scaled_hi_min": float(np.sqrt(shares).min()) if shares.size else 0.0,
        "scaled_lo_min": float(np.sqrt(1.0 - shares).min()) if shares.size else 0.0,
        "p_percentiles": percentiles,
        "u_p_rank_corr": rank_correlation,
        "share_source": "SHARE_REF" if BEHAVIOUR_OFF else "af3_potential_rank",
        "num_items": int(frozen["num_items"]),
        "table_digest": str(frozen.get("digest", "")),
        "switches": switches(),
    }


# --------------------------------------------------------------------------------------
# ball operators: poincare_contract.py:85 expmap0, :141/:153 squared distance
# --------------------------------------------------------------------------------------

def _as(value, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def tanhc(x: torch.Tensor) -> torch.Tensor:
    """`tanh(x)/x` with the limit 1 at `x = 0` (explicit, keeping the sign)."""
    safe = torch.where(x.abs() < 1e-8, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.tanh(safe) / safe)


def project(x: torch.Tensor, c) -> torch.Tensor:
    """Keep a point inside the ball of the TRUE radius `(1-eps)/sqrt(c)`."""
    c = _as(c, x)
    if float(c) == 0.0:
        return x
    limit = (1.0 - BALL_EPS) / torch.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > limit, limit / norm.clamp_min(1e-30),
                        torch.ones_like(norm))
    return x * scale


def expmap0(v: torch.Tensor, c) -> torch.Tensor:
    """`tanh(sqrt(c)|v|)/(sqrt(c)|v|) * v`; the identity at `c = 0`."""
    c = _as(c, v)
    if float(c) == 0.0:
        return v.clone()
    root = torch.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True)
    return project(tanhc(root * norm) * v, c)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """`x (+) y` in the ball of curvature `-c`; the Euclidean sum at `c = 0`."""
    c = _as(c, x)
    if float(c) == 0.0:
        return x + y
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(1e-30)


def squared_distance(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """`((2/sqrt(c)) artanh(sqrt(c)|(-x)(+)y|))^2`; `4|x-y|^2` at `c = 0`."""
    c = _as(c, x)
    if float(c) == 0.0:
        return 4.0 * (x - y).pow(2).sum(-1)
    root = torch.sqrt(c)
    norm = mobius_add(-x, y, c).norm(dim=-1)
    angle = torch.atanh((root * norm).clamp(max=1 - ARTANH_EPS))
    return ((2.0 / root) * angle).pow(2)


# --------------------------------------------------------------------------------------
# the distance itself
# --------------------------------------------------------------------------------------

def euclidean(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The frozen baseline cost, `(B, D)` against `(K, D)`, squared Euclidean.

    This is the geometry_off geometry, exposed as its own registered name so the
    baseline stays callable (and the two branches stay distinguishable in the registry).
    """
    return _euclidean(latent, codebook)


def _euclidean(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The frozen baseline expansion, element for element."""
    return (
        latent.pow(2).sum(1, keepdim=True)
        - 2 * latent @ codebook.t()
        + codebook.pow(2).sum(1, keepdim=True).t()
    )


#: Rows per block in the (nonlinear) hyperbolic factors; the cost is row-additive,
#: so blocking changes only memory, never a value.
ROW_CHUNK = 4096


def _product_cost(latent: torch.Tensor, codebook: torch.Tensor,
                  share: torch.Tensor) -> torch.Tensor:
    """`(B, 32)` x `(K, 32)` -> `(B, K)` on `R^16 x H^8(c_hi) x H^8(c_lo)`.

    Formulas `latent_split`, `energy_split` and `product_cost`: the latent is cut
    ``(16 | 8 | 8)``, only the codeword blocks are scaled (by ``sqrt(p)`` and
    ``sqrt(1-p)``), and the three factor costs are added with the frozen
    `SCALE_HYP` on the two hyperbolic ones.
    """
    x_e = latent[:, :D_E]
    x_h1 = latent[:, D_E:D_E + D_H1]
    x_h2 = latent[:, D_E + D_H1:]
    t_e = codebook[:, :D_E]
    t_h1 = codebook[:, D_E:D_E + D_H1]
    t_h2 = codebook[:, D_E + D_H1:]

    # Euclidean factor: same expanded form as the baseline (no broadcast blow-up).
    cost = (x_e.pow(2).sum(1, keepdim=True)
            - 2 * x_e @ t_e.t()
            + t_e.pow(2).sum(1, keepdim=True).t())

    # Hyperbolic factors: the item side is fixed, the codeword side is scaled per item.
    x_hi = expmap0(x_h1, C_HI).unsqueeze(1)
    x_lo = expmap0(x_h2, C_LO).unsqueeze(1)
    v_hi = expmap0(share * t_h1.unsqueeze(0), C_HI)
    v_lo = expmap0(torch.sqrt(1.0 - share) * t_h2.unsqueeze(0), C_LO)

    return cost + SCALE_HYP * (
        squared_distance(x_hi, v_hi, C_HI) + squared_distance(x_lo, v_lo, C_LO)
    )


def distances(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """`(B, 32)` against `(K, 32)` -> `(B, K)`, the product-manifold cost.

    The rows of `latent` are aligned with the `item_id` field the host published;
    the AF3 share `p_i` of row `i` scales the codeword's two hyperbolic blocks.
    """
    if GEOMETRY_OFF:
        # formula `euclidean_degeneracy`: the explicit baseline branch, not a tiny c.
        if latent.shape[-1] != D_TOTAL or codebook.shape[-1] != D_TOTAL:
            raise ValueError(
                "the baseline cost needs %d dimensions, got %s / %s"
                % (D_TOTAL, tuple(latent.shape), tuple(codebook.shape)))
        return _euclidean(latent, codebook)

    if latent.dim() != 2 or codebook.dim() != 2:
        raise ValueError("distances expects (B, D) against (K, D)")
    if latent.shape[-1] != D_TOTAL or codebook.shape[-1] != D_TOTAL:
        raise ValueError(
            "the product manifold is %d-dimensional (16|8|8), got %s / %s"
            % (D_TOTAL, tuple(latent.shape), tuple(codebook.shape)))

    import runtime_context

    shares_np = shares_for(runtime_context.get("item_id"))
    if shares_np.size != latent.shape[0]:
        raise ValueError(
            "the AF3 share table gave %d rows for %d latents"
            % (shares_np.size, latent.shape[0]))
    share = torch.as_tensor(shares_np, dtype=latent.dtype, device=latent.device)
    if BEHAVIOUR_OFF:
        share = torch.full_like(share, SHARE_REF)
    share = share.view(-1, 1, 1)

    total = latent.shape[0]
    if total <= ROW_CHUNK:
        return _product_cost(latent, codebook, share)
    blocks = [_product_cost(latent[start:start + ROW_CHUNK], codebook,
                            share[start:start + ROW_CHUNK])
              for start in range(0, total, ROW_CHUNK)]
    return torch.cat(blocks, dim=0)


# --------------------------------------------------------------------------------------
# wiring diagnostics (called by the tests; nothing here runs during training)
# --------------------------------------------------------------------------------------

def geometry_off_matches_baseline(latent: torch.Tensor, codebook: torch.Tensor,
                                  *, atol: float = 1e-6, rtol: float = 0.0) -> dict:
    """`geometry_off` against the frozen 32-d expansion, element for element."""
    baseline = _euclidean(latent, codebook)
    previous = GEOMETRY_OFF
    try:
        set_switches(geometry_off=True)
        branch = distances(latent, codebook)
    finally:
        set_switches(geometry_off=previous)
    delta = (branch - baseline).abs()
    return {
        "max_abs_diff": float(delta.max()) if delta.numel() else 0.0,
        "atol": float(atol),
        "rtol": float(rtol),
        "ok": bool(torch.allclose(branch, baseline, rtol=rtol, atol=atol)),
    }


def ball_boundary_readout(points: torch.Tensor, c) -> dict:
    """How close `expmap0(points, c)` sits to the ball boundary, as a ratio.

    `ratio = ||expmap0(v, c)|| / ((1 - BALL_EPS) / sqrt(c))`, so 1.0 means the
    point was projected onto the boundary (saturated) and small values mean the
    factor still moves with its input.
    """
    limit = (1.0 - BALL_EPS) / float(np.sqrt(float(c)))
    mapped = expmap0(points, c).norm(dim=-1).div(limit)
    values = mapped.detach().to("cpu", torch.float64).flatten().numpy()
    if values.size == 0:
        return {"limit": float(limit), "quantiles": {}, "clipped_fraction": 0.0}
    quantiles = {name: float(np.percentile(values, q))
                 for name, q in (("p05", 5.0), ("p50", 50.0), ("p95", 95.0))}
    return {
        "limit": float(limit),
        "quantiles": quantiles,
        "clipped_fraction": float((values >= 1.0 - 1e-6).mean()),
    }


def saturation_readout(latent: torch.Tensor, codebook: torch.Tensor,
                       share: float = SHARE_REF) -> dict:
    """Boundary margins of both factors, for the constant-curvature check.

    `c_hi = 4.0` (radius 0.5) is fixed by math.json; the readout exists so the
    measured `||g|| / radius` quantiles and the clipped fraction are reported
    instead of assumed. A factor whose clipped fraction is ~1 has stopped moving
    with `p_i` and the constant would have to be revisited by the host.
    """
    x_h1 = latent[:, D_E:D_E + D_H1]
    x_h2 = latent[:, D_E + D_H1:]
    t_h1 = codebook[:, D_E:D_E + D_H1]
    t_h2 = codebook[:, D_E + D_H1:]
    scale = torch.as_tensor(float(share), dtype=latent.dtype, device=latent.device)
    return {
        "c_hi": float(C_HI),
        "c_lo": float(C_LO),
        "share": float(share),
        "item_hi": ball_boundary_readout(x_h1, C_HI),
        "item_lo": ball_boundary_readout(x_h2, C_LO),
        "codeword_hi": ball_boundary_readout(scale * t_h1, C_HI),
        "codeword_lo": ball_boundary_readout(torch.sqrt(1.0 - scale) * t_h2, C_LO),
    }


def distance_depends_on_p(latent: torch.Tensor, codebook: torch.Tensor,
                          share_a: float, share_b: float) -> dict:
    """The cost must move when only the per-item share moves."""
    rows = latent.shape[0]
    columns = codebook.shape[0]
    first = torch.full((rows, 1, 1), float(share_a), dtype=latent.dtype,
                       device=latent.device)
    second = torch.full((rows, 1, 1), float(share_b), dtype=latent.dtype,
                        device=latent.device)
    delta = (_product_cost(latent, codebook, first)
             - _product_cost(latent, codebook, second)).abs()
    return {
        "rows": int(rows),
        "codebook": int(columns),
        "share_a": float(share_a),
        "share_b": float(share_b),
        "max_abs_diff": float(delta.max()) if delta.numel() else 0.0,
        "changed": bool(delta.numel() > 0 and float(delta.max()) > 0.0),
    }
