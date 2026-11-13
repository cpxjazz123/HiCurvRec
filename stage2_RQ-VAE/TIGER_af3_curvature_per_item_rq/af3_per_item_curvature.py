"""Frozen per-item curvature table: AF3 structural potential -> one curvature per item.

Computed ONCE before training, then frozen. Exactly what
`ideas/AF3_curvature_per_item_idea/math.json` states:

    kappa_ij  = 4 - d_i - d_j + 3 * T_ij                    (formula `af3_potential`)
    kappa_bar_i = mean_{j in N(i)} kappa_ij, degree 0 -> graph mean
    u_i       = (rank(kappa_bar_i) - 0.5) / N                (GLOBAL ascending rank,
                                                              ties -> average rank)
    c_i       = c_min * (c_max / c_min)^{u_i}                (formula `curvature_map`, strictly
                                                              monotone, continuous, no threshold)
    R_i       = 1 / sqrt(c_i)

Three rules the numbers depend on:

* the graph and the potentials come from the SIBLING arm's `af3_potential.py` (seven
  functions: `repository_root`, `train_parquet`, `sequences`, `edge_arrays`, `adjacency`,
  `edge_curvatures`, `structural_potentials`). `standardise`, `huber`, `calibrate_lambda`,
  `build` and `potential_tensor` from that file are NOT called: this arm does no
  standardisation (the rank map is invariant to a strictly monotone transform), no Huber
  and no bias;
* the rank is computed on the WHOLE table, not inside a batch, so an item gets the same
  `c_i` in every batch and the result does not depend on batch composition;
* an item missing from the table (unknown / cold-start key) takes `u_i = 0.5`, i.e.
  `c_i = 1`; the miss count and the tie count (how many items took part in a tie) are
  reported.

This file fails loud when the cross-arm import is broken: it never keeps a second copy
of the AF3 graph code, because two implementations would silently disagree about what
`kappa_bar_i` means.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

#: constants of the mechanism (math.json `constants`); fixed, not scanned
C_MIN = 0.25
C_MAX = 4.0
D_E = 16
D_H = 16
SCALE_HYP = 0.25
#: `u_i = 0.5 -> c_i = 1`, the value `behaviour_off` collapses to
U_OFF = 0.5

#: the sibling arm whose frozen AF3 implementation this table composes
SIBLING_ARM = "TIGER_af3_curvature_drive_rq"
SIBLING_FILE = "af3_potential.py"
#: the seven functions this arm is allowed to use from it
USED_FUNCTIONS = ("repository_root", "train_parquet", "sequences", "edge_arrays",
                  "adjacency", "edge_curvatures", "structural_potentials")
#: the five this arm must not call (kept here so the intent is checkable)
UNUSED_FUNCTIONS = ("standardise", "huber", "calibrate_lambda", "build", "potential_tensor")


def repository_root() -> Path:
    """The repository root, found by the `stage2_RQ-VAE` marker (no parent counting)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


def sibling_path() -> Path:
    return repository_root() / "stage2_RQ-VAE" / SIBLING_ARM / SIBLING_FILE


def _load_sibling():
    """Load the sibling arm's frozen AF3 module, or stop with the exact expected path."""
    path = sibling_path()
    if not path.is_file():
        raise RuntimeError(
            "the AF3 structural potential lives in the sibling arm; expected "
            + "/".join(("stage2_RQ-VAE", SIBLING_ARM, SIBLING_FILE))
            + " below the repository root, found nothing at "
            + str(path)
            + " -- the cross-arm import must not be replaced by a local copy, the two "
              "would silently diverge"
        )
    spec = importlib.util.spec_from_file_location("_af3_potential_sibling", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery guard
        raise RuntimeError("cannot load the sibling AF3 module from " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in USED_FUNCTIONS if not hasattr(module, name)]
    if missing:  # pragma: no cover - fails only if the sibling arm changes shape
        raise RuntimeError("the sibling AF3 module is missing %s (expected %s in %s)"
                           % (", ".join(missing), SIBLING_FILE, path))
    return module


def global_rank_u(potentials, *, missing=None) -> tuple[np.ndarray, int, int]:
    """`u_i = (rank - 0.5) / N` over the whole table; ties take the average rank.

    Returns `(u, tie_count, miss_count)`. `missing` marks the rows that are not in the
    table at all: they take `u = 0.5` and are excluded from the ranking, so they cannot
    shift anybody else's rank. `tie_count` is how many items took part in a tie (an item
    that shares its potential with another one), not the number of tied groups.
    """
    values = np.asarray(potentials, dtype=np.float64).reshape(-1)
    if missing is None:
        missing = np.zeros(values.shape[0], dtype=bool)
    else:
        missing = np.asarray(missing, dtype=bool).reshape(-1)
    u = np.full(values.shape[0], U_OFF, dtype=np.float64)
    present = ~missing
    count = int(present.sum())
    if count == 0:
        return u, 0, int(missing.sum())
    present_values = values[present]
    # one vectorised rank: `rank(method="average")` is the average-rank tie rule
    ranks = pd.Series(present_values).rank(method="average").to_numpy(dtype=np.float64)
    u[present] = (ranks - 0.5) / float(count)
    _, counts = np.unique(present_values, return_counts=True)
    ties = int(counts[counts > 1].sum())
    return u, ties, int(missing.sum())


def curvature_of_u(u, *, behaviour_off: bool = False) -> np.ndarray:
    """`c_i = c_min * (c_max/c_min)^{u_i}`; `behaviour_off` pins `u_i = 0.5`."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if behaviour_off:
        u = np.full(u.shape[0], U_OFF, dtype=np.float64)
    return C_MIN * np.power(C_MAX / C_MIN, u)


def build(frame: pd.DataFrame | None = None, *, num_items: int | None = None,
          behaviour_off: bool = False) -> dict:
    """The frozen table plus the diagnostics the trainer logs once.

    `num_items` is the embedding row count (item ids are dense, as in `train_rqvae.py`).
    Ids `>= num_items` cannot exist; rows that carry no potential keep the graph mean,
    so the only "missing key" case is a row outside the table, which is reported as 0
    unless the caller says otherwise.
    """
    potential = _load_sibling()
    if frame is None:
        frame = pd.read_parquet(potential.train_parquet())
    items = potential.sequences(frame)
    rows, cols = potential.edge_arrays(items)
    if num_items is None:
        num_items = 0 if len(rows) == 0 else int(max(int(rows.max()), int(cols.max()))) + 1
    num_items = int(num_items)
    matrix = potential.adjacency(rows, cols, num_items)
    edge_kappa = potential.edge_curvatures(matrix, rows, cols)
    kappa_bar = potential.structural_potentials(matrix, rows, cols, edge_kappa)

    missing = np.zeros(num_items, dtype=bool)  # ids 0..num_items-1 all exist
    u, ties, misses = global_rank_u(kappa_bar, missing=missing)
    c = curvature_of_u(u, behaviour_off=behaviour_off)
    radius = 1.0 / np.sqrt(c)
    #: the projection limit the contract enforces, `(1 - BALL_EPS) / sqrt(c)`
    from model.poincare_contract import BALL_EPS  # noqa: PLC0415 - import-time side effect avoided
    radius_limit = (1.0 - BALL_EPS) / np.sqrt(c)

    def quantiles(values) -> dict:
        if values.size == 0:
            return {}
        points = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        return {key: float(value) for key, value in
                zip(("min", "p25", "p50", "p75", "max"), points)}

    return {
        "kappa_bar": np.asarray(kappa_bar, dtype=np.float64),
        "u": u,
        "curvature": c,
        "radius": radius,
        "num_items": num_items,
        "behaviour_off": bool(behaviour_off),
        "metrics": {
            "items": num_items,
            "graph_edges": int(len(rows)),
            "ties": ties,
            "misses": misses,
            "curvature_min": float(c.min()) if c.size else 0.0,
            "curvature_max": float(c.max()) if c.size else 0.0,
            "curvature_variance": float(c.var(ddof=0)) if c.size else 0.0,
            "curvature_quantiles": quantiles(c),
            "radius_quantiles": quantiles(radius),
            "radius_limit_quantiles": quantiles(radius_limit),
            "log_curvature_variance": float(np.log(c).var(ddof=0)) if c.size else 0.0,
            "c_min": C_MIN, "c_max": C_MAX,
            "D_E": D_E, "D_H": D_H, "SCALE_HYP": SCALE_HYP,
        },
    }


def curvature_tensor(table: dict):
    """The frozen curvature table as a `float32` tensor (one value per item row)."""
    import torch

    return torch.as_tensor(table["curvature"], dtype=torch.float32)
