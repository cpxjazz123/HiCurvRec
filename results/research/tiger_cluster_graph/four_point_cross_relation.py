"""Targeted four-point (Buneman) search for crossing semantic relations.

We look for four items A,B,C,D such that:
  * A,B share a fine category path;
  * C,D share a different fine category path and coarse branch;
  * A-C and B-D are close in the text embedding;
  * A-D and B-C are far.
The four-point sums S1=dAB+dCD, S2=dAC+dBD, S3=dAD+dBC should have
equal largest two for a tree metric.  A large residual here is a concrete
tree-metric violation with a category/scene interpretation to inspect.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/four_point_cross_relation.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
LABELS = REPO / "results" / "research" / "semantic_tree_hypothesis" / "semantic_tree_labels.csv"
OUT_DIR = REPO / "results" / "research" / "four_point_cross_relation"
OUT_JSON = OUT_DIR / "four_point_cross_relation.json"
OUT_MD = OUT_DIR / "four_point_cross_relation.md"
OUT_CSV = OUT_DIR / "four_point_cross_relation_candidates.csv"
GROUP_SAMPLE = 18
GROUP_PAIRS = 250
TOP_CROSS = 35
EXTRA_PAIRS = (
    ("Solid Body", "Electric Guitar Strings"),
    ("Condenser Microphones > Multipurpose", "Microphone Cables"),
    ("Ukuleles", "Ukulele Accessories"),
    ("Bass Drums", "Bass Drumheads"),
    ("Solid Body", "Combo Amps"),
    ("Electric Guitar Effects", "Instrument Cables"),
)
EXPLICIT_PAIRS = (
    ("Solid Body", "Electric Guitar Strings", "electric guitars + electric guitar strings (complementary)"),
    ("Condenser Microphones > Multipurpose", "Microphone Cables", "condenser microphones + microphone cables (complementary)"),
    ("Ukuleles", "Ukulele Accessories", "ukuleles + ukulele cases (complementary)"),
)
SEED = 20261123


def load() -> tuple[pd.DataFrame, np.ndarray, dict, dict]:
    labels = pd.read_csv(LABELS).drop_duplicates("item_id").reset_index(drop=True)
    embedding = np.load(EMBEDDING).astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    paths = {int(row.item_id): str(row.path) for row in labels.itertuples(index=False)}
    coarse = {int(row.item_id): str(row.coarse) for row in labels.itertuples(index=False)}
    return labels, embedding, paths, coarse


def groups(labels: pd.DataFrame) -> dict[str, list[int]]:
    output = defaultdict(list)
    for index, row in enumerate(labels.itertuples(index=False)):
        output[str(row.path)].append(index)
    return {path: members for path, members in output.items() if len(members) >= 4}


def group_pair_candidates(labels: pd.DataFrame, embedding: np.ndarray,
                          paths: dict, coarse: dict, group_map: dict) -> list[tuple[str, str, float]]:
    names = list(group_map)
    centroids = {}
    for name in names:
        member_ids = [int(labels["item_id"].iloc[index]) for index in group_map[name]]
        centroids[name] = embedding[member_ids].mean(axis=0)
    matrix = np.stack([centroids[name] for name in names])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarity = matrix @ matrix.T
    candidates = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            first_name, second_name = names[first], names[second]
            first_coarse = coarse[int(labels["item_id"].iloc[group_map[first_name][0]])]
            second_coarse = coarse[int(labels["item_id"].iloc[group_map[second_name][0]])]
            if first_coarse == second_coarse:
                continue
            candidates.append((first_name, second_name, float(similarity[first, second])))
    candidates.sort(key=lambda value: -value[2])
    selected = candidates[:GROUP_PAIRS]
    names_set = set(names)
    for first_keyword, second_keyword in EXTRA_PAIRS:
        first_names = [name for name in names_set if first_keyword in name]
        second_names = [name for name in names_set if second_keyword in name]
        for first_name in first_names:
            for second_name in second_names:
                if first_name == second_name:
                    continue
                first_coarse = coarse[int(labels["item_id"].iloc[group_map[first_name][0]])]
                second_coarse = coarse[int(labels["item_id"].iloc[group_map[second_name][0]])]
                if first_coarse == second_coarse:
                    continue
                if (first_name, second_name) not in {(a, b) for a, b, _ in selected}:
                    selected.append((first_name, second_name, -1.0))
    return selected


def quadruple_candidates(first: str, second: str, group_map: dict,
                         labels: pd.DataFrame, embedding: np.ndarray,
                         rng: np.random.Generator) -> list[dict]:
    first_indices = group_map[first]
    second_indices = group_map[second]
    if len(first_indices) > GROUP_SAMPLE:
        first_indices = list(rng.choice(first_indices, GROUP_SAMPLE, replace=False))
    if len(second_indices) > GROUP_SAMPLE:
        second_indices = list(rng.choice(second_indices, GROUP_SAMPLE, replace=False))
    if len(first_indices) < 3 or len(second_indices) < 3:
        return []
    ids = np.asarray([
        int(labels["item_id"].iloc[index]) for index in (*first_indices, *second_indices)
    ], dtype=np.int64)
    vectors = embedding[ids]
    distance = np.sqrt(np.maximum(2.0 - 2.0 * (vectors @ vectors.T), 0.0))
    n1 = len(first_indices)
    cross = []
    for first_local in range(n1):
        for second_local in range(n1, len(ids)):
            cross.append((float(distance[first_local, second_local]), first_local, second_local))
    cross.sort(key=lambda value: value[0])
    cross = cross[:TOP_CROSS]
    output = []
    for first_distance, a, c in cross:
        for second_distance, b, d in cross:
            if a == b or c == d or a == d or b == c:
                continue
            s1 = float(distance[a, b] + distance[c, d])
            s2 = float(distance[a, c] + distance[b, d])
            s3 = float(distance[a, d] + distance[b, c])
            values = sorted([s1, s2, s3], reverse=True)
            residual = (values[0] - values[1]) / max(values[0], 1e-12)
            # The intended crossing shape: the two cross pairs A-C and B-D
            # are the close pairs; A-D and B-C must be clearly farther.
            if distance[a, d] <= 1.35 * distance[a, c] or distance[b, c] <= 1.35 * distance[b, d]:
                continue
            output.append({
                "path_a": first, "path_b": second,
                "a_id": int(ids[a]), "b_id": int(ids[b]),
                "c_id": int(ids[c]), "d_id": int(ids[d]),
                "s1_ab_cd": s1, "s2_ac_bd": s2, "s3_ad_bc": s3,
                "four_point_residual": float(residual),
                "d_ab": float(distance[a, b]), "d_cd": float(distance[c, d]),
                "d_ac": float(distance[a, c]), "d_bd": float(distance[b, d]),
                "d_ad": float(distance[a, d]), "d_bc": float(distance[b, c]),
            })
    return output


def explicit_best(first_keyword: str, second_keyword: str, relation_note: str,
                  group_map: dict, labels: pd.DataFrame, embedding: np.ndarray,
                  coarse: dict) -> dict | None:
    first_names = [name for name in group_map if name.endswith(first_keyword)]
    second_names = [name for name in group_map if name.endswith(second_keyword)]
    best = None
    for first in first_names:
        for second in second_names:
            if first == second:
                continue
            first_index = group_map[first]
            second_index = group_map[second]
            first_coarse = coarse[int(labels.item_id.iloc[first_index[0]])]
            second_coarse = coarse[int(labels.item_id.iloc[second_index[0]])]
            if first_coarse == second_coarse:
                continue
            first_indices = first_index[:12]
            second_indices = second_index[:12]
            ids = np.asarray([int(labels.item_id.iloc[i]) for i in (*first_indices, *second_indices)])
            vectors = embedding[ids]
            distance = np.sqrt(np.maximum(2.0 - 2.0 * (vectors @ vectors.T), 0.0))
            n1 = len(first_indices)
            for a in range(n1):
                for b in range(n1):
                    if a == b:
                        continue
                    for c in range(n1, len(ids)):
                        for d in range(n1, len(ids)):
                            if c == d:
                                continue
                            s1 = float(distance[a, b] + distance[c, d])
                            s2 = float(distance[a, c] + distance[b, d])
                            s3 = float(distance[a, d] + distance[b, c])
                            # The intended crossing: A-C and B-D are the close
                            # pairs, A-D and B-C are clearly farther.
                            if not (distance[a, c] < distance[a, d]
                                    and distance[a, c] < distance[b, c]
                                    and distance[b, d] < distance[a, d]
                                    and distance[b, d] < distance[b, c]):
                                continue
                            values = sorted([s1, s2, s3], reverse=True)
                            residual = (values[0] - values[1]) / max(values[0], 1e-12)
                            if best is None or residual > best["four_point_residual"]:
                                best = {
                                    "first_path": first, "second_path": second,
                                    "semantic_relation": relation_note,
                                    "a_id": int(ids[a]), "b_id": int(ids[b]),
                                    "c_id": int(ids[c]), "d_id": int(ids[d]),
                                    "a_title": str(labels["title"].iloc[first_indices[a]]),
                                    "b_title": str(labels["title"].iloc[first_indices[b]]),
                                    "c_title": str(labels["title"].iloc[second_indices[c - n1]]),
                                    "d_title": str(labels["title"].iloc[second_indices[d - n1]]),
                                    "s1_ab_cd": s1, "s2_ac_bd": s2, "s3_ad_bc": s3,
                                    "four_point_residual": float(residual),
                                    "d_ab": float(distance[a, b]), "d_cd": float(distance[c, d]),
                                    "d_ac": float(distance[a, c]), "d_bd": float(distance[b, d]),
                                    "d_ad": float(distance[a, d]), "d_bc": float(distance[b, c]),
                                }
    return best


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, embedding, paths, coarse = load()
    group_map = groups(labels)
    rng = np.random.default_rng(SEED)
    candidates = group_pair_candidates(labels, embedding, paths, coarse, group_map)
    print("group pairs", len(candidates), flush=True)

    rows = []
    for first, second, centroid_similarity in candidates:
        for value in quadruple_candidates(first, second, group_map, labels, embedding, rng):
            value["group_centroid_cosine"] = centroid_similarity
            rows.append(value)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no crossing quadruples found")
    frame = frame.sort_values("four_point_residual", ascending=False).reset_index(drop=True)
    frame.to_csv(OUT_CSV, index=False)
    explicit = []
    for first_keyword, second_keyword, note in EXPLICIT_PAIRS:
        value = explicit_best(first_keyword, second_keyword, note, group_map, labels, embedding, coarse)
        if value is not None:
            explicit.append(value)

    # Attach titles for manual audit.
    title = {int(row.item_id): str(row.title) for row in labels.itertuples(index=False)}
    top = frame.head(40).copy()
    for column in ("a_id", "b_id", "c_id", "d_id"):
        top[column.replace("_id", "_title")] = top[column].map(title)

    result = {
        "labelled_items": int(len(labels)),
        "fine_groups": int(len(group_map)),
        "group_pairs_examined": int(len(candidates)),
        "candidate_quadruples": int(len(frame)),
        "residual_definition": "(largest of S1,S2,S3 minus second largest) / largest",
        "top_candidates": top.to_dict(orient="records"),
        "top_group_pairs": [
            {"first": first, "second": second, "centroid_cosine": score}
            for first, second, score in candidates[:20]
        ],
        "explicit_complementary_quadruples": explicit,
        "file": str(OUT_CSV.relative_to(REPO)),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Targeted four-point crossing relations",
        "",
        "A,B share a fine category; C,D share a different fine category and coarse branch.",
        "Candidates require A-C and B-D close, A-D and B-C clearly farther, and a large",
        "four-point residual (tree requires the largest two of S1/S2/S3 to be equal).",
        "",
        f"Fine groups: **{len(group_map)}**; group pairs examined: **{len(candidates)}**; "
        f"candidate quadruples: **{len(frame)}**.",
        "",
        "| residual | A | B | C | D |",
        "|---|---|---|---|---|",
    ]
    for row in top.head(20).itertuples(index=False):
        lines.append(f"| {row.four_point_residual:.4f} | {row.a_title} | {row.b_title} | "
                     f"{row.c_title} | {row.d_title} |")
    lines += [
        "",
        "## Explicit complementary pairs",
        "",
        "| relation | A | B | C | D | S1 | S2 | S3 | residual |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for value in explicit:
        lines.append(f"| {value['semantic_relation']} | {value['a_title']} | {value['b_title']} | "
                     f"{value['c_title']} | {value['d_title']} | {value['s1_ab_cd']:.3f} | "
                     f"{value['s2_ac_bd']:.3f} | {value['s3_ad_bc']:.3f} | "
                     f"{value['four_point_residual']:.4f} |")
    lines += [
        "",
        "The table is candidate generation only. The semantic relation of each example",
        "must still be checked in the titles/descriptions; a high residual alone is not",
        "a proof of a meaningful cross-category relation.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
