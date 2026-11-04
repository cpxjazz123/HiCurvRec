"""How far one exported SID partition is from another, without looking at recall.

Two SIDs can be compared as *partitions of the item set* at every prefix level.
The numbers below say whether a run changed the identifier at all, which is a
different question from whether the change was good:

* ``raw_identity`` share of items whose raw prefix tokens are unchanged;
* ``identity``   the same on the relabelled partition (invariant to a renaming);
* ``ari``        adjusted Rand index (chance-corrected pair agreement);
* ``nmi``        normalised mutual information;
* ``vi``         variation of information in nats (a metric; 0 = same partition);
* ``bucket_delta`` how many buckets each side occupies.

Every level is reported separately because a coarse level can be stable while the
fine levels move (or the reverse).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEVELS = (1, 2, 3)


def load_labels(sid_path: Path, level: int) -> tuple[np.ndarray, np.ndarray, list]:
    """The item ids, the integer prefix label and the raw prefix of each item."""
    table = json.loads(Path(sid_path).read_text())
    items = np.array(sorted(int(key) for key in table), dtype=np.int64)
    prefixes = [tuple(table[str(int(item))][:level]) for item in items]
    mapping: dict[tuple, int] = {}
    labels = np.empty(items.size, dtype=np.int64)
    for position, prefix in enumerate(prefixes):
        labels[position] = mapping.setdefault(prefix, len(mapping))
    return items, labels, prefixes


def _entropy(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    probabilities = counts[counts > 0] / labels.size
    return float(-(probabilities * np.log(probabilities)).sum())


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    joint, _, _ = np.histogram2d(a, b, bins=(a.max() + 1, b.max() + 1))
    joint = joint / joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log(joint / (pa * pb))
    return float(np.nansum(term))


def compare(a_labels: np.ndarray, b_labels: np.ndarray,
            a_raw: list | None = None, b_raw: list | None = None) -> dict:
    """The partition distance between two labelings of the same items."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    identity = float(np.mean(a_labels == b_labels))
    ha, hb = _entropy(a_labels), _entropy(b_labels)
    mi = _mutual_information(a_labels, b_labels)
    raw_identity = (float(np.mean([x == y for x, y in zip(a_raw, b_raw)]))
                    if a_raw is not None and b_raw is not None else None)
    return {
        "raw_identity": raw_identity,
        "identity": identity,
        "ari": float(adjusted_rand_score(a_labels, b_labels)),
        "nmi": float(normalized_mutual_info_score(a_labels, b_labels)),
        "vi": float(ha + hb - 2.0 * mi),
        "buckets_a": int(a_labels.max() + 1),
        "buckets_b": int(b_labels.max() + 1),
    }


def sid_distance(reference: Path, candidate: Path, levels=LEVELS) -> dict:
    """Per-level partition distance between two SID exports."""
    report = {}
    for level in levels:
        ref_items, ref_labels, ref_raw = load_labels(reference, level)
        cand_items, cand_labels, cand_raw = load_labels(candidate, level)
        if not np.array_equal(ref_items, cand_items):
            raise ValueError(f"the two SIDs cover different item sets at level {level}")
        report[f"L{level}"] = compare(ref_labels, cand_labels, ref_raw, cand_raw)
    return report


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: sid_distance.py <reference item_sids.json> <candidate item_sids.json>",
              file=sys.stderr)
        return 2
    report = sid_distance(Path(argv[0]), Path(argv[1]))
    for level, values in report.items():
        print(level, " ".join(
            f"{key}={value:.4f}" if isinstance(value, float) and value is not None
            else f"{key}={value}" for key, value in values.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
