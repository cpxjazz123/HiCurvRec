"""Train-only transition analysis for Stage0 prefix-expanded RecBole Parquet.

The train split repeats the same user history at multiple prefix lengths.
For each user, only the longest train prefix plus its target represents the
complete *available TRAIN* sequence; this prevents duplicate event counting.
No valid/test parquet or full raw interaction JSON is opened.
"""
from collections import Counter, defaultdict
from math import log2
from pathlib import Path

from ..core.evidence import atomic_json, digest_file


def recover_train_sequences(rows):
    latest = {}
    for row in rows:
        if isinstance(row, dict):
            user, history, target = int(row["user"]), row["history"], int(row["target"])
        else:
            user, history, target = int(row.user), row.history, int(row.target)
        seq = tuple(int(x) for x in history) + (target,)
        if user not in latest or len(seq) > len(latest[user]):
            latest[user] = seq
        elif len(seq) == len(latest[user]) and seq != latest[user]:
            raise ValueError("conflicting maximum training prefix for user")
    return latest


def analyze_sequences(sequences):
    edges, successors, items = Counter(), defaultdict(Counter), Counter()
    n_events = 0
    for uid, seq in sorted(sequences.items()):
        del uid
        n_events += len(seq)
        items.update(seq)
        for src, dst in zip(seq, seq[1:]):
            edges[(src, dst)] += 1
            successors[src][dst] += 1
    n_edges = sum(edges.values())
    asymmetric = sum(n for (a, b), n in edges.items() if a != b and edges[(b, a)] == 0)
    weighted_entropy = 0.0
    for group in successors.values():
        total = sum(group.values())
        weighted_entropy += total * (-sum((n / total) * log2(n / total) for n in group.values()))
    return {
        "schema": 1, "population": "train_only_longest_prefix_per_user",
        "users": len(sequences), "events": n_events, "transitions": n_edges,
        "distinct_items": len(items), "directed_pairs": len(edges),
        "no_observed_reverse_events": asymmetric,
        "no_reverse_fraction": (asymmetric / n_edges if n_edges else None),
        "mean_next_item_entropy_bits": (weighted_entropy / n_edges if n_edges else None),
        "top_edges": [{"from": a, "to": b, "count": n}
                      for (a, b), n in sorted(edges.items(), key=lambda x: (-x[1], x[0]))[:20]],
        "interpretation_limit": "Observed direction and entropy do NOT establish negative curvature or a causal next-item improvement.",
    }


def run(train_path: Path, output: Path) -> Path:
    # pandas is needed only for this analysis, not by the stdlib agent/ledger/MCP.
    import pandas as pd
    train_path = Path(train_path)
    frame = pd.read_parquet(train_path, columns=["user", "history", "target"])
    report = analyze_sequences(recover_train_sequences(frame.itertuples(index=False)))
    report["source_file"] = str(train_path)
    report["source_sha256"] = digest_file(train_path)
    atomic_json(Path(output), report)
    return Path(output)
