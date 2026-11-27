#!/usr/bin/env python3
"""Issue #274, step 3: build the frozen positive/negative pair index from the full triplet table.

The mechanism (issue section 1) selects, per centre A and per comparable centre-edge direction
pattern:
  * the positive outer pair (B, C) from the **closed triangle** triplet with the **highest** raw
    graph sectional curvature;
  * the negative outer pair (D, E) from the **open wedge** triplet with the **lowest** raw graph
    sectional curvature;
with the curvature value entering only this offline selection.

Frozen rules implemented here (see CONTRACT.md):
  * comparability group = (centre, the unordered multiset of the two centre-edge directions);
  * closed = ``bc_dist_sq == 1``, open = ``bc_dist_sq == 4``;
  * minimum valid support = at least 2 closed and at least 2 open candidates in the group;
  * an exact tie at the extremum excludes the group (no arbitrary tie-break);
  * invalid rows (``valid == False``) are excluded;
  * the hold-out is a deterministic centre split, disjoint from the training centres.

Outputs (all under results/research/issue274/):
  pairs_index.npz        the selected pairs and their provenance columns
  pairs_manifest.json    the counts, the exclusions, the hashes, the coverage
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"
HOLDOUT_MODULUS = 5
HOLDOUT_RESIDUE = 0
MIN_CLOSED = 2
MIN_OPEN = 2

# the direction codes: 0 none, 1 out (centre -> neighbour), 2 in (neighbour -> centre), 3 both
DIR_NONE, DIR_OUT, DIR_IN, DIR_BOTH = 0, 1, 2, 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_flag_matrix() -> np.ndarray:
    """flag[u, v]: 1 if u -> v, 2 if v -> u, 3 if both (the visible training graph)."""
    forward = np.load(REPO / "assets/train/visible_forward.npz")
    reverse = np.load(REPO / "assets/train/visible_reverse.npz")
    nodes = len(forward["indptr"]) - 1
    flags = np.zeros((nodes, nodes), dtype=np.uint8)
    for graph, bit in ((forward, DIR_OUT), (reverse, DIR_IN)):
        indices = graph["indices"]
        indptr = graph["indptr"]
        rows = np.repeat(np.arange(nodes), np.diff(indptr))
        flags[rows, indices] |= bit
    return flags


def main() -> None:
    global MIN_CLOSED, MIN_OPEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-closed", type=int, default=MIN_CLOSED)
    parser.add_argument("--min-open", type=int, default=MIN_OPEN)
    parser.add_argument("--out", default=str(OUT / "pairs_index.npz"))
    args = parser.parse_args()
    MIN_CLOSED, MIN_OPEN = args.min_closed, args.min_open
    OUT.mkdir(parents=True, exist_ok=True)
    flags = build_flag_matrix()
    index = np.load(REPO / "assets/index.npz")
    shard_names = [str(s) for s in index["shard_names"]]

    columns = {name: [] for name in
               ("centre", "pattern", "pos_b", "pos_c", "neg_d", "neg_e",
                "pos_curvature", "neg_curvature", "pos_class", "neg_class",
                "pos_dir_b", "pos_dir_c", "neg_dir_b", "neg_dir_c")}
    counts = {
        "rows_total": 0, "rows_valid": 0, "rows_invalid": 0,
        "groups_total": 0, "groups_below_support": 0, "groups_tied": 0,
        "groups_closed_missing": 0, "groups_open_missing": 0, "groups_selected": 0,
        "groups_selected_train": 0, "groups_selected_holdout": 0,
    }

    for shard_index, shard_name in enumerate(shard_names):
        path = REPO / "assets/shards" / f"shard_{shard_index:03d}.npz"
        z = np.load(path)
        centre = z["centre"]
        b = z["b"]
        c = z["c"]
        dist = z["bc_dist_sq"]
        curvature = z["curvature"]
        valid = z["valid"]
        klass = z["structure_class"]
        counts["rows_total"] += int(centre.size)
        counts["rows_valid"] += int(valid.sum())
        counts["rows_invalid"] += int((~valid).sum())
        keep = valid & ((dist == 1) | (dist == 4))
        centre, b, c = centre[keep], b[keep], c[keep]
        dist, curvature, klass = dist[keep], curvature[keep], klass[keep]
        if centre.size == 0:
            continue

        dir_b = flags[centre, b]
        dir_c = flags[centre, c]
        # the comparability pattern: the unordered multiset of the two centre-edge directions
        lo = np.minimum(dir_b, dir_c)
        hi = np.maximum(dir_b, dir_c)
        pattern = lo.astype(np.int32) * 4 + hi.astype(np.int32)
        group_key = centre.astype(np.int64) * 64 + pattern.astype(np.int64)

        closed = dist == 1
        # order: group, then closed flag, then the extremum direction (max curvature for the
        # closed rows, min curvature for the open rows)
        order = np.lexsort((np.where(closed, curvature, -curvature), closed, group_key))
        gkey = group_key[order]
        cflag = closed[order]
        curv = curvature[order]
        bb, cc, kk, db, dc = b[order], c[order], klass[order], dir_b[order], dir_c[order]

        # the group boundaries
        starts = np.flatnonzero(np.r_[True, gkey[1:] != gkey[:-1]])
        ends = np.r_[starts[1:], gkey.size]
        for start, end in zip(starts, ends):
            counts["groups_total"] += 1
            key = gkey[start]
            group_closed = cflag[start:end]
            group_curv = curv[start:end]
            n_closed = int(group_closed.sum())
            n_open = int((~group_closed).sum())
            if n_closed == 0:
                counts["groups_closed_missing"] += 1
                continue
            if n_open == 0:
                counts["groups_open_missing"] += 1
                continue
            if n_closed < MIN_CLOSED or n_open < MIN_OPEN:
                counts["groups_below_support"] += 1
                continue
            # the closed extremum: the first closed row of the group (sorted by -curvature)
            first_closed = start + int(np.argmax(group_closed))
            # the open extremum: the first open row of the group (sorted by +curvature)
            open_positions = np.flatnonzero(~group_closed)
            first_open = start + int(open_positions[0])
            pos_tie = int((group_curv[group_closed] == curv[first_closed]).sum()) > 1
            neg_tie = int((group_curv[~group_closed] == curv[first_open]).sum()) > 1
            if pos_tie or neg_tie:
                counts["groups_tied"] += 1
                continue
            centre_value = int(key // 64)
            counts["groups_selected"] += 1
            if centre_value % HOLDOUT_MODULUS == HOLDOUT_RESIDUE:
                counts["groups_selected_holdout"] += 1
            else:
                counts["groups_selected_train"] += 1
            columns["centre"].append(centre_value)
            columns["pattern"].append(int(key % 64))
            columns["pos_b"].append(int(bb[first_closed]))
            columns["pos_c"].append(int(cc[first_closed]))
            columns["neg_d"].append(int(bb[first_open]))
            columns["neg_e"].append(int(cc[first_open]))
            columns["pos_curvature"].append(float(curv[first_closed]))
            columns["neg_curvature"].append(float(curv[first_open]))
            columns["pos_class"].append(int(kk[first_closed]))
            columns["neg_class"].append(int(kk[first_open]))
            columns["pos_dir_b"].append(int(db[first_closed]))
            columns["pos_dir_c"].append(int(dc[first_closed]))
            columns["neg_dir_b"].append(int(db[first_open]))
            columns["neg_dir_c"].append(int(dc[first_open]))

    arrays = {name: np.asarray(values, dtype=(np.int64 if name in
                                              ("centre", "pattern", "pos_b", "pos_c", "neg_d", "neg_e",
                                               "pos_class", "neg_class", "pos_dir_b", "pos_dir_c",
                                               "neg_dir_b", "neg_dir_c") else np.float64))
              for name, values in columns.items()}
    centre = arrays["centre"]
    arrays["is_holdout"] = (centre % HOLDOUT_MODULUS == HOLDOUT_RESIDUE)
    index_path = Path(args.out).resolve()
    np.savez_compressed(index_path, **arrays)

    manifest = {
        "schema": "genrec.issue274.pairs_manifest/1",
        "frozen_rules": {
            "comparability_group": "the centre plus the unordered multiset of the two centre-edge directions",
            "closed": "bc_dist_sq == 1", "open": "bc_dist_sq == 4",
            "minimum_support": {"closed": MIN_CLOSED, "open": MIN_OPEN},
            "tie_rule": "an exact tie at either extremum excludes the group",
            "invalid_rule": "rows with valid == False are excluded",
            "positive": "the closed-triangle triplet with the highest raw graph curvature",
            "negative": "the open-wedge triplet with the lowest raw graph curvature",
            "holdout": f"centre % {HOLDOUT_MODULUS} == {HOLDOUT_RESIDUE}",
            "curvature_role": "offline selection only; it never enters the loss, the margin or a weight",
        },
        "counts": counts,
        "rows": {name: int(arrays[name].size) for name in arrays},
        "centres_with_pairs": int(np.unique(centre).size),
        "holdout_groups": int(arrays["is_holdout"].sum()),
        "train_groups": int((~arrays["is_holdout"]).sum()),
        "index_path": str(index_path.relative_to(REPO)),
        "index_sha256": sha256(index_path),
        "shards": [str(p.relative_to(REPO)) for p in sorted((REPO / "assets/shards").glob("shard_*.npz"))],
    }
    (index_path.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "centres_with_pairs", "train_groups",
                                               "holdout_groups", "index_sha256")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
