"""Build the occurrence table for the context/structure validation.

Each row is one observed purchase of a selected target item X at position i in a user session:

    predecessor   the item at i-1   (the *only* context input; never i+1)
    context       global text cluster of the predecessor
    next          the item at i+1   (the outcome; never used to define context)
    outcome       global text cluster of the next item
    structure     the TRAIN edge (predecessor, X): graphlet pattern, triangles, roles

TRAIN sessions discover the context->outcome and context->structure associations; TEST sessions are
the independent future data.  The text clusters are unsupervised item-text clusters, fitted without
any behaviour label, so test outcomes cannot leak into the context definition.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/context_structure_validation/prepare.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ITEMS = REPO / "results" / "stage0_build_parquet" / "items.parquet"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
TEST = REPO / "results" / "stage0_build_parquet" / "test.parquet"
EMB = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
EDGES = REPO / "results" / "research" / "graphlet_relation_sid" / "edge_features.parquet"
GRAPHLETS = REPO / "results" / "research" / "graphlet_relation_sid" / "graphlet_names.json"

OUT_OCC = HERE / "occurrences.parquet"
OUT_TARGETS = HERE / "targets.json"
OUT_CLUSTERS = HERE / "text_clusters.json"
OUT_MODEL = HERE / "text_clusters.npz"

K_TEXT = 20
MIN_TARGET_EVENTS = 700
MIN_CONTEXT_EVENTS = 40
MIN_OUTCOME_EVENTS = 20
MAX_TARGETS = 6
SEED = 42


def load_sessions(path: Path) -> dict[int, list[int]]:
    frame = pd.read_parquet(path, columns=["user", "seen_history", "target"])
    best: dict[int, tuple[int, list[int]]] = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)
    ):
        history = [] if history is None else [int(x) for x in history]
        user = int(user)
        if user not in best or len(history) > best[user][0]:
            best[user] = (len(history), [*history, int(target)])
    return {user: session for user, (_, session) in best.items()}


def main() -> int:
    items = pd.read_parquet(ITEMS).sort_values("item_id").reset_index(drop=True)
    titles = items["title"].tolist()
    embeddings = normalize(np.load(EMB).astype(np.float64))
    model = MiniBatchKMeans(n_clusters=K_TEXT, random_state=SEED, n_init=10, batch_size=4096)
    labels = model.fit_predict(embeddings)
    np.savez(OUT_MODEL, labels=labels, centers=model.cluster_centers_)

    cluster_titles = {}
    for cluster in range(K_TEXT):
        members = np.flatnonzero(labels == cluster)
        similarity = embeddings[members] @ model.cluster_centers_[cluster]
        top = members[np.argsort(-similarity)[:8]]
        cluster_titles[str(cluster)] = {
            "size": int(len(members)),
            "top_titles": [titles[int(item)] for item in top],
        }
    OUT_CLUSTERS.write_text(json.dumps(cluster_titles, indent=2) + "\n")

    train = load_sessions(TRAIN)
    test = load_sessions(TEST)

    # candidate targets: selected on TRAIN only; one pass over TRAIN collects contexts/outcomes
    train_events = Counter()
    item_contexts: dict[int, Counter] = defaultdict(Counter)
    item_outcomes: dict[int, Counter] = defaultdict(Counter)
    for session in train.values():
        for position, item in enumerate(session):
            if 0 < position < len(session) - 1:
                train_events[item] += 1
                item_contexts[item][int(labels[session[position - 1]])] += 1
                item_outcomes[item][int(labels[session[position + 1]])] += 1
    candidates = []
    for item, count in train_events.items():
        if count < MIN_TARGET_EVENTS:
            continue
        context_counts = item_contexts[item]
        outcome_counts = item_outcomes[item]
        contexts = sum(1 for value in context_counts.values() if value >= MIN_CONTEXT_EVENTS)
        outcomes = sum(1 for value in outcome_counts.values() if value >= MIN_OUTCOME_EVENTS)
        if contexts >= 4 and outcomes >= 5:
            candidates.append((item, count, contexts, outcomes))
    candidates.sort(key=lambda row: row[1], reverse=True)
    selected = [row[0] for row in candidates[:MAX_TARGETS]]
    print("selected targets:", [(item, titles[item][:70]) for item in selected], flush=True)

    edges = pd.read_parquet(
        EDGES,
        columns=[
            "left", "right", "graphlet_mask", "triangles", "role_left", "role_right",
            "cooc_users", "p_trans_max", "same_community", "graphlet_count",
        ],
    )
    edge_index = {
        (int(left), int(right)): row
        for left, right, row in zip(edges["left"], edges["right"], edges.to_dict("records"))
    }
    names = json.loads(GRAPHLETS.read_text())["order"]

    rows = []
    for split, sessions in (("train", train), ("test", test)):
        for user, session in sessions.items():
            for position, item in enumerate(session):
                if item not in selected or position == 0 or position == len(session) - 1:
                    continue
                predecessor = int(session[position - 1])
                following = int(session[position + 1])
                context = int(labels[predecessor])
                outcome = int(labels[following])
                key = (min(predecessor, item), max(predecessor, item))
                edge = edge_index.get(key)
                if edge is None:
                    pattern = "unseen"
                    seen = 0
                    triangles = np.nan
                    tri_bin = "unseen"
                    role_pair = "unseen"
                    cooc = np.nan
                    p_trans = np.nan
                    same_community = np.nan
                    graphlet_count = np.nan
                    dense_label = "unseen"
                else:
                    seen = 1
                    mask = int(edge["graphlet_mask"])
                    patterns = [name for bit, name in enumerate(names) if (mask >> bit) & 1]
                    if not patterns:
                        pattern = "none"
                    elif len(patterns) == 1:
                        pattern = patterns[0]
                    else:
                        pattern = "multi"
                    dense = any(name in patterns for name in ("K5", "K4_tail", "K2_3"))
                    sparse = any(name in patterns for name in ("double_triangle", "P5_centre", "C5", "P5_hops", "K1_4", "T5"))
                    dense_label = "dense" if dense and not sparse else "sparse" if sparse and not dense else "mixed" if dense and sparse else "none"
                    triangles = int(edge["triangles"])
                    tri_bin = "0" if triangles == 0 else "1" if triangles == 1 else "2+"
                    low, high = sorted((int(edge["role_left"]), int(edge["role_right"])))
                    role_pair = f"{low}-{high}"
                    cooc = int(edge["cooc_users"])
                    p_trans = float(edge["p_trans_max"])
                    same_community = int(edge["same_community"])
                    graphlet_count = int(edge["graphlet_count"])
                rows.append(
                    {
                        "split": split,
                        "user": int(user),
                        "target": int(item),
                        "position": int(position),
                        "history_length": int(position),
                        "predecessor": predecessor,
                        "context": context,
                        "next": following,
                        "outcome": outcome,
                        "seen_edge": seen,
                        "pattern": pattern,
                        "dense_label": dense_label,
                        "triangles": triangles,
                        "tri_bin": tri_bin,
                        "role_pair": role_pair,
                        "cooc_users": cooc,
                        "p_trans_max": p_trans,
                        "same_community": same_community,
                        "graphlet_count": graphlet_count,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_OCC, index=False)

    target_meta = {}
    for item in selected:
        train_rows = frame[(frame["target"] == item) & (frame["split"] == "train")]
        test_rows = frame[(frame["target"] == item) & (frame["split"] == "test")]
        target_meta[str(item)] = {
            "title": titles[item],
            "train_events": int(len(train_rows)),
            "test_events": int(len(test_rows)),
            "train_contexts": int(train_rows["context"].nunique()),
            "train_outcomes": int(train_rows["outcome"].nunique()),
            "seen_edge_share_train": float(train_rows["seen_edge"].mean()),
            "seen_edge_share_test": float(test_rows["seen_edge"].mean()),
        }
    OUT_TARGETS.write_text(json.dumps(target_meta, indent=2) + "\n")
    print(json.dumps(target_meta, indent=2), flush=True)
    print(f"wrote {OUT_OCC} {frame.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
