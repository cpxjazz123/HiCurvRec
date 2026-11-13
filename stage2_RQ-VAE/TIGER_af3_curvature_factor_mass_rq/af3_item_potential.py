"""AF3 structural potential -> the full-table rank `u_i`, computed once and frozen.

One mapping, written exactly as `ideas/AF3_curvature_factor_mass_idea/math.json`
states it (formula `af3_potential`):

    kappa_ij    = 4 - d_i - d_j + 3 * T_ij           (sibling arm, edge_curvatures)
    kappa_bar_i = mean_{j in Adj(i)} kappa_ij        (sibling arm, structural_potentials)
    rank_i      = average rank of kappa_bar_i over the WHOLE table, ascending
    u_i         = (rank_i - 0.5) / N
    u_miss      = (rank_of_graph_mean - 0.5) / N      (degree 0 / missing key)

The AF3 construction is NOT copied here: `repository_root`, `train_parquet`,
`sequences`, `edge_arrays`, `adjacency`, `edge_curvatures` and
`structural_potentials` are imported from the sibling arm
`stage2_RQ-VAE/TIGER_af3_curvature_drive_rq/af3_potential.py` and used as they are.
`standardise`, `huber`, `calibrate_lambda`, `build` and `potential_tensor` are not
called: the rank acts on the UNSTANDARDISED neighbourhood mean, and a rank is
invariant under a strictly monotone transform anyway.

This file stops at `u_i`. The `u -> p` map lives in exactly one place,
`factor_share_geometry.share`, and nowhere else.

Output: the single artefact `eval/af3_item_potential.npz` with the fields
`item_ids`, `potential`, `u`, `rank`, `is_missing`, `rank_of_graph_mean` and the
`digest` (`sha256`) computed over those arrays. Training and export read that one
file and never recompute the table.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

#: The sibling arm whose AF3 implementation is reused verbatim.
SIBLING_ARM = "TIGER_af3_curvature_drive_rq"
SIBLING_FILE = "af3_potential.py"
#: The functions this file is allowed to call on the sibling module.
REUSED = ("repository_root", "train_parquet", "sequences", "edge_arrays",
          "adjacency", "edge_curvatures", "structural_potentials")
#: The embedding table that fixes the item-id space (the trainer's own EMBEDDING_FILE).
EMBEDDING_RELATIVE = ("stage1_GeneEmbedding", "output", "sentence_t5.npy")
#: The one artefact this file writes, relative to the candidate tree.
OUTPUT_DIRECTORY = "eval"
OUTPUT_NAME = "af3_item_potential.npz"
#: The fields of the artefact, in the order the digest walks them.
FIELDS = ("item_ids", "potential", "u", "rank", "is_missing", "rank_of_graph_mean")

_SIBLING = None


# --------------------------------------------------------------------------------------
# locating and importing the sibling arm
# --------------------------------------------------------------------------------------

def repository_root() -> Path:
    """The repository root, found by the `stage2_RQ-VAE` marker (no parent counting)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


def sibling_path() -> Path:
    """The full path of the sibling arm's AF3 file."""
    return repository_root() / "stage2_RQ-VAE" / SIBLING_ARM / SIBLING_FILE


def sibling():
    """Import the sibling arm's `af3_potential` module (cached, fail loud)."""
    global _SIBLING
    if _SIBLING is not None:
        return _SIBLING
    path = sibling_path()
    if not path.is_file():
        raise RuntimeError("the sibling AF3 implementation is missing: " + str(path))
    try:
        spec = importlib.util.spec_from_file_location(
            "af3_potential_" + SIBLING_ARM.replace("-", "_"), path)
        if spec is None or spec.loader is None:
            raise ImportError("no import spec could be built for " + str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as error:
        raise RuntimeError(
            "cannot import the sibling AF3 implementation from %s (required functions: %s): %s"
            % (path, ", ".join(REUSED), error)) from error
    missing = [name for name in REUSED if not hasattr(module, name)]
    if missing:
        raise RuntimeError("%s does not expose %s; the AF3 construction is not reused"
                           % (path, ", ".join(missing)))
    _SIBLING = module
    return module


# --------------------------------------------------------------------------------------
# the item-id space
# --------------------------------------------------------------------------------------

def embedding_file() -> Path:
    """The embedding table the trainer loads, resolved from the sibling's repo root."""
    root = sibling().repository_root()
    return root.joinpath(*EMBEDDING_RELATIVE)


def embedding_rows() -> int:
    """How many item ids the embedding table has (the size of the full table)."""
    path = embedding_file()
    if not path.is_file():
        raise RuntimeError("the embedding table that fixes the item-id space is missing: "
                           + str(path))
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise RuntimeError("the embedding table is not a matrix: %s shape=%s"
                           % (path, array.shape))
    return int(array.shape[0])


# --------------------------------------------------------------------------------------
# ranks
# --------------------------------------------------------------------------------------

def average_rank(values) -> np.ndarray:
    """Ascending ranks with ties averaged (never broken by item id)."""
    from scipy.stats import rankdata

    array = np.asarray(values, dtype=np.float64)
    return rankdata(array, method="average").astype(np.float64)


def rank_of_value(values, value: float) -> float:
    """The average rank `value` itself would receive in `values`."""
    array = np.asarray(values, dtype=np.float64)
    less = float((array < value).sum())
    equal = float((array == value).sum())
    return less + (equal + 1.0) / 2.0


# --------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------

def build_table(frame: pd.DataFrame | None = None, *, num_items: int | None = None) -> dict:
    """Compute `kappa_bar_i`, the full-table rank and `u_i` once, over the train split.

    Nothing here trains and nothing here is differentiable: every value is numpy
    `float64`. `is_missing` marks the items the AF3 graph never touches (degree 0,
    which includes the ids absent from every sequence); their potential is the graph
    mean and their rank is therefore `rank_of_graph_mean`, i.e. `u_miss`.
    """
    af3 = sibling()
    if frame is None:
        frame = pd.read_parquet(af3.train_parquet())
    if num_items is None:
        num_items = embedding_rows()
    num_items = int(num_items)
    if num_items <= 0:
        raise RuntimeError("the item-id space is empty; cannot rank a structural potential")

    items = af3.sequences(frame)
    rows, cols = af3.edge_arrays(items)
    matrix = af3.adjacency(rows, cols, num_items)
    curvature = af3.edge_curvatures(matrix, rows, cols)
    potential = np.asarray(af3.structural_potentials(matrix, rows, cols, curvature),
                           dtype=np.float64)
    if potential.shape != (num_items,):
        raise RuntimeError("structural_potentials returned shape %s for %d items"
                           % (potential.shape, num_items))
    if not np.isfinite(potential).all():
        raise RuntimeError("the AF3 structural potential contains non-finite values")

    degrees = np.asarray(matrix.sum(1)).ravel() if num_items else np.zeros(0)
    is_missing = degrees <= 0
    if is_missing.any():
        # Fail loud, one line per item: a silently defaulted share would be invisible.
        for item_id in np.flatnonzero(is_missing):
            print("[af3_item_potential] item_id=%d has no AF3 edge (degree 0); "
                  "u_i is the rank position of the graph-mean potential"
                  % int(item_id), flush=True)

    rank = average_rank(potential)
    u = (rank - 0.5) / float(num_items)

    has_edges = ~is_missing
    graph_mean = float(np.mean(potential[has_edges])) if has_edges.any() else 0.0
    rank_of_graph_mean = rank_of_value(potential, graph_mean)

    item_ids = np.arange(num_items, dtype=np.int64)
    table = {
        "item_ids": item_ids,
        "potential": potential,
        "u": u,
        "rank": rank,
        "is_missing": is_missing,
        "rank_of_graph_mean": float(rank_of_graph_mean),
    }
    _validate(table)
    # The missing rows are exactly the graph-mean rows, so their u must be u_miss.
    if is_missing.any():
        u_miss = (rank_of_graph_mean - 0.5) / float(num_items)
        drift = float(np.max(np.abs(u[is_missing] - u_miss)))
        if drift > 1e-9:
            raise RuntimeError("degree-0 items do not sit at the graph-mean rank "
                               "(max drift %.3e)" % drift)
    return table


def _validate(table: dict) -> None:
    """Lengths, ranges and finiteness; anything off raises instead of writing a file."""
    arrays = {}
    for name in FIELDS:
        if name not in table:
            raise RuntimeError("the AF3 table is missing the field %r" % (name,))
        arrays[name] = np.asarray(table[name])
    item_ids = arrays["item_ids"]
    if item_ids.ndim != 1 or not np.issubdtype(item_ids.dtype, np.integer):
        raise RuntimeError("item_ids must be a 1-D integer array")
    if len(np.unique(item_ids)) != len(item_ids):
        raise RuntimeError("item_ids contains duplicates")
    reference = int(item_ids.shape[0])
    # `rank_of_graph_mean` is one scalar rank, not a per-item field: it must not be
    # forced to length N by the vector fields' consistency check.
    graph_mean_rank = arrays["rank_of_graph_mean"]
    if graph_mean_rank.ndim != 0:
        raise RuntimeError("rank_of_graph_mean must be a scalar; got shape %r"
                           % (graph_mean_rank.shape,))
    sizes = {name: arrays[name].shape[0] if arrays[name].ndim else 1
             for name in FIELDS if name != "rank_of_graph_mean"}
    wrong = {name: size for name, size in sizes.items() if size != reference}
    if wrong:
        raise RuntimeError("the AF3 table fields disagree in length: %s (reference %d)"
                           % (wrong, reference))
    u = arrays["u"]
    if not np.isfinite(u).all():
        raise RuntimeError("u contains non-finite values")
    if bool((u <= 0.0).any()) or bool((u >= 1.0).any()):
        raise RuntimeError("u must sit strictly inside (0, 1); got [%r, %r]"
                           % (float(u.min()), float(u.max())))
    rank = arrays["rank"]
    if not np.isfinite(rank).all():
        raise RuntimeError("rank contains non-finite values")
    if float(rank.min()) < 1.0 or float(rank.max()) > float(reference):
        raise RuntimeError("rank must lie in [1, N]; got [%r, %r]"
                           % (float(rank.min()), float(rank.max())))
    if not np.isfinite(arrays["potential"]).all():
        raise RuntimeError("potential contains non-finite values")
    if arrays["is_missing"].dtype != np.bool_:
        raise RuntimeError("is_missing must be boolean")
    graph_mean_rank = float(graph_mean_rank)
    if not np.isfinite(graph_mean_rank):
        raise RuntimeError("rank_of_graph_mean is not finite")
    if graph_mean_rank < 1.0 or graph_mean_rank > float(reference):
        raise RuntimeError("rank_of_graph_mean must lie in [1, N]; got %r"
                           % (graph_mean_rank,))


# --------------------------------------------------------------------------------------
# the artefact
# --------------------------------------------------------------------------------------

def output_path(root: Path | None = None) -> Path:
    """`<candidate tree>/eval/af3_item_potential.npz`."""
    base = Path(__file__).resolve().parent if root is None else Path(root)
    return base / OUTPUT_DIRECTORY / OUTPUT_NAME


def digest(payload: dict) -> str:
    """`sha256` over the field names, dtypes, shapes and bytes of the artefact."""
    hasher = hashlib.sha256()
    for name in FIELDS:
        array = np.ascontiguousarray(np.asarray(payload[name]))
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def save(table: dict, path: Path | None = None) -> Path:
    """Validate, then write the single artefact with its digest inside."""
    _validate(table)
    payload = {name: np.asarray(table[name]) for name in FIELDS}
    fingerprint = digest(payload)
    target = output_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, digest=np.asarray(fingerprint), **payload)
    return target


def load(path: Path | None = None) -> dict:
    """Read the artefact back and refuse it if the digest does not match."""
    target = output_path() if path is None else Path(path)
    if not target.is_file():
        raise RuntimeError("the frozen AF3 potential table is missing: " + str(target))
    with np.load(target) as data:
        stored = str(np.asarray(data["digest"]))
        payload = {name: np.asarray(data[name]) for name in FIELDS}
    recomputed = digest(payload)
    if stored != recomputed:
        raise RuntimeError("the AF3 potential table at %s is corrupt: digest %s != %s"
                           % (target, stored, recomputed))
    _validate(payload)
    payload["digest"] = recomputed
    return payload


def build_and_write(frame: pd.DataFrame | None = None, *, num_items: int | None = None,
                    path: Path | None = None) -> Path:
    """`build_table` -> `save`; the one call the trainer's preparation step makes."""
    table = build_table(frame, num_items=num_items)
    return save(table, path)


def main() -> None:
    target = build_and_write()
    table = load(target)
    print("[af3_item_potential] wrote %s digest=%s items=%d missing=%d u=[%.6f, %.6f] "
          "u_miss=%.6f"
          % (target, table["digest"], len(table["item_ids"]), int(table["is_missing"].sum()),
             float(table["u"].min()), float(table["u"].max()),
             (float(table["rank_of_graph_mean"]) - 0.5) / float(len(table["item_ids"]))),
          flush=True)


if __name__ == "__main__":
    main()
