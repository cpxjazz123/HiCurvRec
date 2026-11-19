"""Build the behaviour graph from adjacent transitions in the training sequences.

    G = (V, E),  E = { {i_t, i_{t+1}} | (i_1..i_T) in S_train }

A user's sequence is the longest `history` of that user's rows plus its `target`
(the rows of a user are growing prefixes of the same sequence).  Only
**consecutive** items are joined: A -> B -> C -> D gives A-B, B-C, C-D and never
A-C, A-D or B-D.  Edges are made undirected and deduplicated; the directed
transition counts are kept alongside, in their own file.

Outputs (results/research/behavior_graph_adjacent/):
    edges_undirected.csv   u,v                     one row per edge, u < v
    transitions.csv        src,dst,count           directed, deduplicated
    graph.npz              edges (E,2), weights (E,), num_items
    build_summary.json     counts, degree stats, provenance

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        analysis_sid_recall/build_behavior_graph.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = REPO / "results" / "research" / "behavior_graph_adjacent"


def sequences() -> list:
    """One ordered item sequence per user: the longest history plus its target."""
    frame = pd.read_parquet(TRAIN)
    longest = {}
    for user, history, target in zip(frame["user"].to_numpy(), frame["history"], frame["target"]):
        key = int(user)
        if key not in longest or len(history) > len(longest[key]):
            longest[key] = list(history)
    by_user = {}
    for user, history, target in zip(frame["user"].to_numpy(), frame["history"], frame["target"]):
        key = int(user)
        if len(history) == len(longest[key]):
            by_user[key] = [int(item) for item in history] + [int(target)]
    return [by_user[user] for user in sorted(by_user)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sequences()
    directed = Counter()
    items = set()
    skipped_short = 0
    for sequence in seqs:
        if len(sequence) < 2:
            skipped_short += 1
            continue
        items.update(sequence)
        for left, right in zip(sequence, sequence[1:]):
            if left == right:
                continue
            directed[(left, right)] += 1
    undirected = defaultdict(int)
    for (left, right), count in directed.items():
        key = (left, right) if left < right else (right, left)
        undirected[key] += count
    edges = sorted(undirected)
    weights = np.array([undirected[edge] for edge in edges], dtype=np.int64)
    edge_array = np.array(edges, dtype=np.int64)

    with (OUT / "edges_undirected.csv").open("w") as sink:
        sink.write("u,v\n")
        for left, right in edges:
            sink.write(f"{left},{right}\n")
    with (OUT / "transitions.csv").open("w") as sink:
        sink.write("src,dst,count\n")
        for (left, right), count in sorted(directed.items()):
            sink.write(f"{left},{right},{count}\n")
    np.savez_compressed(OUT / "graph.npz", edges=edge_array, weights=weights,
                        num_items=len(items))

    degrees = Counter()
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    values = np.array(sorted(degrees.values()))
    summary = {
        "definition": "adjacent transitions of the training sequences, undirected, deduplicated",
        "source": str(TRAIN.relative_to(REPO)),
        "sequences": len(seqs),
        "sequences_shorter_than_2": skipped_short,
        "items": len(items),
        "directed_transitions_distinct": len(directed),
        "directed_transitions_total": int(sum(directed.values())),
        "edges": len(edges),
        "edges_with_both_directions": int(sum(
            1 for left, right in edges if (left, right) in directed and (right, left) in directed)),
        "degree": {"min": int(values.min()), "median": float(np.median(values)),
                   "p90": float(np.percentile(values, 90)),
                   "p99": float(np.percentile(values, 99)), "max": int(values.max())},
        "nodes_without_edge": len(seqs) and 24587 - len(degrees),
        "files": {"edges_undirected.csv": len(edges), "transitions.csv": len(directed),
                  "graph.npz": [len(edges), 3]},
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
