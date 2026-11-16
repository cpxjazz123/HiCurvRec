"""Enrich context events with the five relation-level feature groups.

All values come from the TRAIN behaviour graph; TEST events only read the already-built edge table.
The enriched frame is the input to `incremental.py`.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/context_structure_validation/enrich.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OCC = HERE / "occurrences.parquet"
EDGES = REPO / "results" / "research" / "graphlet_relation_sid" / "edge_features.parquet"
OUT = HERE / "occurrences_enriched.parquet"
ITEMS = REPO / "results" / "stage0_build_parquet" / "items.parquet"
EMB = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
META = HERE / "enrich_meta.json"

COLUMNS = [
    "left", "right", "degree_left", "degree_right", "popularity_left", "popularity_right",
    "role_left", "role_right", "core_left", "core_right", "cluster_left", "cluster_right",
    "triangles", "is_bridge", "same_community", "trans_ab", "trans_ba", "p_trans_ab", "p_trans_ba",
]


STOPWORDS = set("""
a an the and or for with without of to in on at by from as is are was were be been being this that
these those it its his her their our your my you we they he she them him us i not no nor but if then
than so such into over under between within per each other more most less least very can could may
might will would should do does did done has have had having use used using new set pack piece pieces
""".split())


def tokenize(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(title).lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def main() -> int:
    occurrences = pd.read_parquet(OCC)
    items = pd.read_parquet(ITEMS, columns=["item_id", "title"]).sort_values("item_id").reset_index(drop=True)
    titles = items["title"].tolist()
    token_sets = [tokenize(title) for title in titles]
    embeddings = np.load(EMB).astype(np.float64)
    embeddings = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    edges = pd.read_parquet(EDGES, columns=COLUMNS)
    af3_all = 4.0 - edges["degree_left"] - edges["degree_right"] + 3.0 * edges["triangles"]
    q1, q3 = (float(value) for value in np.quantile(af3_all, [0.25, 0.75]))
    index = {
        (int(left), int(right)): row
        for left, right, row in zip(edges["left"], edges["right"], edges.to_dict("records"))
    }

    rows = []
    for row in occurrences.itertuples(index=False):
        record = row._asdict()
        predecessor = int(row.predecessor)
        target = int(row.target)
        first_tokens, second_tokens = token_sets[predecessor], token_sets[target]
        union = len(first_tokens | second_tokens)
        record["sem_cos"] = float(embeddings[predecessor] @ embeddings[target])
        record["title_jaccard"] = float(len(first_tokens & second_tokens) / union) if union else 0.0
        key = (min(predecessor, target), max(predecessor, target))
        edge = index.get(key)
        if edge is None:
            record.update({
                "degree_pred": np.nan, "degree_target": np.nan,
                "pop_pred": np.nan, "pop_target": np.nan,
                "role_pred": np.nan, "role_target": np.nan,
                "core_pred": np.nan, "core_target": np.nan,
                "cluster_pred": np.nan, "cluster_target": np.nan,
                "triangles": np.nan, "is_bridge": np.nan, "same_community": np.nan,
                "af3": np.nan, "af3_high": np.nan, "af3_low": np.nan,
                "trans_f": np.nan, "trans_b": np.nan, "p_f": np.nan, "p_b": np.nan,
                "p_asym": np.nan, "direction_balance": np.nan,
                "hub_is_pred": np.nan, "degree_diff": np.nan, "core_diff": np.nan,
            })
            rows.append(record)
            continue
        forward = int(row.predecessor) == int(edge["left"])
        degree_pred = int(edge["degree_left"] if forward else edge["degree_right"])
        degree_target = int(edge["degree_right"] if forward else edge["degree_left"])
        pop_pred = int(edge["popularity_left"] if forward else edge["popularity_right"])
        pop_target = int(edge["popularity_right"] if forward else edge["popularity_left"])
        role_pred = int(edge["role_left"] if forward else edge["role_right"])
        role_target = int(edge["role_right"] if forward else edge["role_left"])
        core_pred = int(edge["core_left"] if forward else edge["core_right"])
        core_target = int(edge["core_right"] if forward else edge["core_left"])
        cluster_pred = int(edge["cluster_left"] if forward else edge["cluster_right"])
        cluster_target = int(edge["cluster_right"] if forward else edge["cluster_left"])
        trans_f = int(edge["trans_ab"] if forward else edge["trans_ba"])
        trans_b = int(edge["trans_ba"] if forward else edge["trans_ab"])
        p_f = float(edge["p_trans_ab"] if forward else edge["p_trans_ba"])
        p_b = float(edge["p_trans_ba"] if forward else edge["p_trans_ab"])
        af3 = 4.0 - int(edge["degree_left"]) - int(edge["degree_right"]) + 3.0 * int(edge["triangles"])
        record.update({
            "degree_pred": degree_pred, "degree_target": degree_target,
            "pop_pred": pop_pred, "pop_target": pop_target,
            "role_pred": role_pred, "role_target": role_target,
            "core_pred": core_pred, "core_target": core_target,
            "cluster_pred": cluster_pred, "cluster_target": cluster_target,
            "triangles": int(edge["triangles"]), "is_bridge": int(edge["is_bridge"]),
            "same_community": int(edge["same_community"]),
            "af3": af3, "af3_high": int(af3 >= q3), "af3_low": int(af3 <= q1),
            "trans_f": trans_f, "trans_b": trans_b, "p_f": p_f, "p_b": p_b,
            "p_asym": abs(p_f - p_b) / (p_f + p_b + 1e-12),
            "direction_balance": (trans_f - trans_b) / (trans_f + trans_b + 1.0),
            "hub_is_pred": int(degree_pred > degree_target),
            "degree_diff": float(np.log1p(degree_pred) - np.log1p(degree_target)),
            "core_diff": float(np.log1p(core_pred) - np.log1p(core_target)),
        })
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT, index=False)
    meta = {
        "occurrences": str(OCC.relative_to(REPO)),
        "edges": str(EDGES.relative_to(REPO)),
        "af3_rule": "4 - degree_left - degree_right + 3 * triangles (TRAIN graph)",
        "af3_q1": q1,
        "af3_q3": q3,
        "rows": int(len(frame)),
        "seen_edges": int(frame["seen_edge"].sum()),
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
