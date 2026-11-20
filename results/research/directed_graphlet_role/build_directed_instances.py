"""Issue #221 phase 2: directed labels on the audited #217 instance sets.

The undirected reference pool (#217) sampled strict induced node sets from the
same TRAIN transitions; the directed graph built here is the symmetric closure of
exactly those transitions, so every one of those node sets is still an induced
subgraph with the same skeleton.  Re-labelling the SAME sets under the directed
adjacency gives

  * a directed subtype per instance (lexicographically minimal directed pattern),
  * a directed orbit per node (automorphism classes of that pattern),

and keeps #217's stable split, so the undirected and directed arms are compared on
identical instances rather than on a fresh sample.  Sets whose directed skeleton
would fall outside the five families are recorded and dropped (none are expected).

Subtypes with fewer than MIN_SUPPORT instances are merged into `other_<size>` and
counted in coverage.json instead of being silently excluded after seeing results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directed_graphlets import OUT, TYPES, classify, load_adjacency  # noqa: E402

SOURCE = HERE.parent / "graphlet_roles" / "instances.npz"
MIN_SUPPORT = 30


def main() -> None:
    adjacency = load_adjacency()
    source = np.load(SOURCE, allow_pickle=False)
    payload: dict[str, np.ndarray] = {}
    subtype_index: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict] = {}
    for family in TYPES:
        nodes = source[f"{family}_nodes"].astype(np.int64)
        split = source[f"{family}_split"].astype(np.int8)
        keys, orbits = [], []
        for instance in nodes:
            rows = adjacency[instance][:, instance]
            matrix = np.asarray(rows.todense(), dtype=bool)
            result = classify(matrix)
            if result is None or result[0] != family:
                keys.append(None)
                orbits.append(tuple([0] * len(instance)))
                continue
            _skeleton, key, orbit_ids = result
            keys.append(key)
            orbits.append(orbit_ids)
        kept_keys = sorted({key for key in keys if key is not None})
        support = {key: sum(1 for value in keys if value == key) for key in kept_keys}
        # Number only the subtypes that survive min-support: a gap in the
        # enumeration let a kept subtype share the `other` id (P4 did: subtype 11
        # and other=11), which trained the type head on a false class.
        kept = [key for key in kept_keys if support[key] >= MIN_SUPPORT]
        indexed = {key: position for position, key in enumerate(kept)}
        other = len(kept)
        subtype_index[family] = {**indexed, "__other__": other}
        subtype_array = np.array([indexed.get(key, other) for key in keys], dtype=np.int64)
        roles_array = np.array(orbits, dtype=np.int64)
        dropped = np.array([key is None for key in keys], dtype=bool)
        payload[f"{family}_nodes"] = nodes
        payload[f"{family}_roles"] = roles_array
        payload[f"{family}_subtype"] = subtype_array
        payload[f"{family}_split"] = split
        coverage[family] = {
            "instances": int(len(nodes)),
            "skeleton_mismatch": int(dropped.sum()),
            "subtypes_seen": len(kept_keys),
            "subtypes_kept": len(indexed),
            "other_id": other,
            "other_instances": int((subtype_array == other).sum()),
            "train": int((split == 0).sum()),
            "heldout": int((split == 1).sum()),
            "min_support": MIN_SUPPORT,
            "support": support,
            "source": str(SOURCE.relative_to(HERE.parents[2])),
        }
        print(f"{family}: {len(nodes)} instances, {len(kept_keys)} subtypes "
              f"({len(indexed)} kept), mismatch {int(dropped.sum())}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "instances.npz", **payload)
    (OUT / "subtypes.json").write_text(json.dumps(subtype_index, indent=1))
    (OUT / "coverage.json").write_text(json.dumps({
        "min_support": MIN_SUPPORT,
        "source_pool": str(SOURCE.relative_to(HERE.parents[2])),
        "graph": "results/research/behavior_graph_directed",
        "families": coverage,
    }, indent=1))
    print("pool written:", OUT / "instances.npz")


if __name__ == "__main__":
    main()
