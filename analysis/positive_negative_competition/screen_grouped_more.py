"""Measure the newly requested arms with the unchanged grouped screen metrics."""
from __future__ import annotations

import json

import screen_grouped as S

import numpy as np


ARM_LABELS = ("cap_d128", "cap_l1c128", "cap_l1c512")
CODEBOOK_SIZES = {
    "cap_d128": (256, 256, 256),
    "cap_l1c128": (128, 256, 256),
    "cap_l1c512": (512, 256, 256),
}
CODEBOOK_DIMS = {"cap_d128": 128, "cap_l1c128": 32, "cap_l1c512": 32}
_ACTIVE_ARM = None
_ORIGINAL_CODE_METRICS = S.G.code_metrics
_ORIGINAL_PERSIST = S._persist_results


def _codebook_size(label):
    return CODEBOOK_SIZES[label][0]


def _capacity_metadata(label, codes):
    return {"arm": label, "codebook_size_per_level": list(CODEBOOK_SIZES[label]),
            "codebook_dimension": CODEBOOK_DIMS[label], "items": int(len(codes)), "levels": 3}


def _code_metrics(codes, item_ids, codebook_size):
    sizes = CODEBOOK_SIZES[_ACTIVE_ARM]
    output = {}
    for level, size in enumerate(sizes):
        metrics = _ORIGINAL_CODE_METRICS(codes[:, level:level + 1], item_ids, size)
        output[f"L{level + 1}"] = metrics[f"L{level + 1}"]
    sid = codes[np.asarray(item_ids, dtype="int64")]
    distinct = int(len(np.unique(sid, axis=0)))
    output["full_sid"] = {"distinct_triples": distinct, "items": int(len(item_ids)),
                          "collision_rate": float(1.0 - distinct / len(item_ids))}
    return output


def _persist_merged(results):
    path = S.CAPACITY / "capacity_grouped_metrics.json"
    existing = json.loads(path.read_text()) if path.is_file() else {"protocol": {}, "arms": []}
    updated = {arm["arm"]: arm for arm in results}
    combined = [arm for arm in existing.get("arms", []) if arm["arm"] not in updated]
    combined.extend(results)
    _ORIGINAL_PERSIST(combined)


def run():
    S._codebook_size = _codebook_size
    S._capacity_metadata = _capacity_metadata
    S.G.code_metrics = _code_metrics
    S._persist_results = _persist_merged
    try:
        for label in ARM_LABELS:
            global _ACTIVE_ARM
            _ACTIVE_ARM = label
            S.ARM_LABELS = (label,)
            S.run()
    finally:
        S.G.code_metrics = _ORIGINAL_CODE_METRICS
        S._persist_results = _ORIGINAL_PERSIST


if __name__ == "__main__":
    run()
