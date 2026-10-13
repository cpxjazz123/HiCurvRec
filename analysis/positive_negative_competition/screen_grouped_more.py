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
    return CODEBOOK_SIZES[_ACTIVE_ARM][0]


def _capacity_metadata(label, codes):
    return {"arm": label, "codebook_size_per_level": list(CODEBOOK_SIZES[label]),
            "codebook_dimension": CODEBOOK_DIMS[label], "items": int(len(codes)), "levels": 3}


def _code_metrics(codes, item_ids, codebook_size):
    sizes = CODEBOOK_SIZES[_ACTIVE_ARM]
    output = {}
    for level, size in enumerate(sizes):
        metrics = _ORIGINAL_CODE_METRICS(codes[:, level:level + 1], item_ids, size)
        output[f"L{level + 1}"] = metrics["L1"]
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
    _add_stratified_density_and_prefix()


def _add_stratified_density_and_prefix():

    pairs = S.S._load_pairs()
    frame = S.pd.read_parquet(S.VALID)
    successors = S.S._successors(frame, pairs)
    embeddings = np.load(S.EMBEDDINGS, mmap_mode="r")
    pop_by_item = S._popularity(frame, len(embeddings))
    popularity_deciles = S.G._rank_bins(pop_by_item, 10)
    anchors, positives, negatives = S.C._indices(pairs)
    successor_counts = np.asarray(
        [len(successors.get(int(anchor), ())) for anchor in anchors], dtype=np.int64)
    result_path = S.CAPACITY / "capacity_grouped_metrics.json"
    payload = json.loads(result_path.read_text())
    from eval_behavior_metrics import metrics_for

    for arm in payload["arms"]:
        label = arm["arm"]
        if label not in ARM_LABELS:
            continue
        residual_path = S._stage_and_extract(label, S.ARMS_ROOT / label)
        _, spaces, codes = S.S._load_levels(label, residual_path)
        for level, vectors in spaces.items():
            hard_relative = S.G._hardness_distance(vectors, pairs)
            actual_strata = S.G.make_strata(
                pop_by_item[anchors], successor_counts, hard_relative,
                popularity_deciles=popularity_deciles[anchors])
            level_result = arm["levels"][level]
            for axis, groups in actual_strata.items():
                for name, mask in groups.items():
                    row = level_result["strata"][axis][name]
                    if row.get("metrics") == "undefined":
                        continue
                    row["local_density"] = S.G.local_density(vectors, anchors[mask], S.K_VALUES)
                    sub = {"anchors": anchors[mask], "positives": positives[mask, 0],
                           "negatives": negatives[mask]}
                    prefix = metrics_for(codes, sub)
                    row["prefix_sharing_and_er"] = {
                        key: prefix[key] for key in
                        ("L1", "L1_L2", "L1_L2_L3", "L2_given_L1", "L3_given_L2")}
    _ORIGINAL_PERSIST(payload["arms"])


if __name__ == "__main__":
    run()
