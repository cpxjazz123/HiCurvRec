"""What the exported identifiers do to the behaviour neighbourhood.

The arm's own training record says whether the new term moved the encoder, the
codebook and the exported identifiers. This script asks the other half of the
question, on the **exported** identifiers only: do two items that co-occur in
the training sequences end up sharing a prefix, and did the hierarchy survive?

Everything here is read from disk: the identifiers of an arm and the training
split's behaviour graph. The graph is the same convention the arm was
supervised with (undirected co-occurrence on consecutive items of a user's
training sequence), so the measurement is of the thing the arm was asked to
preserve.

The comparison against the frozen baseline is the point: a share that does not
beat the baseline's is not a gain, and a hierarchy that loses prefixes or
collides more is a loss even when the share rises.

Run:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/neighbour_preservation.py \
        --arm TIGER_community_graph_recon
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
OUT = REPO / "results" / "research" / "issue121" / "community_curvature"
SEED = 42


def behaviour_edges(frame, n_items: int) -> np.ndarray:
    """The training split's undirected co-occurrence graph, as pairs."""
    pairs: set[tuple[int, int]] = set()
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second or not (0 <= first < n_items and 0 <= second < n_items):
                continue
            pairs.add((first, second) if first < second else (second, first))
    return np.asarray(sorted(pairs), dtype=np.int64)


def load_identifiers(path: Path) -> np.ndarray:
    """The identifier of every item, indexed by item id."""
    table = json.loads(path.read_text())
    width = len(next(iter(table.values())))
    out = np.full((max(int(key) for key in table) + 1, width), -1, dtype=np.int64)
    for key, row in table.items():
        out[int(key)] = [int(value) for value in row]
    return out


def share(identifiers: np.ndarray, left: np.ndarray, right: np.ndarray, depth: int) -> float:
    """The fraction of pairs that share their first ``depth`` tokens."""
    same = np.ones(left.shape[0], dtype=bool)
    for level in range(depth):
        same &= identifiers[left, level] == identifiers[right, level]
    return float(same.mean())


def hierarchy(identifiers: np.ndarray) -> dict:
    """The identifiers' own structure: usage, prefixes, collisions."""
    total = identifiers.shape[0]
    level_one = len(np.unique(identifiers[:, 0]))
    level_two = len(np.unique(identifiers[:, :2], axis=0))
    level_three = len(np.unique(identifiers[:, :3], axis=0))
    return {
        "items": int(total),
        "level1_codes_used": int(level_one),
        "level2_prefixes": int(level_two),
        "level3_combinations": int(level_three),
        "level3_collision": float(1.0 - level_three / total),
        "level1_usage_max_share": float(
            np.bincount(identifiers[:, 0]).max() / total),
    }


def analyse(path: Path, edges: np.ndarray, n_items: int, rng) -> dict:
    identifiers = load_identifiers(path)
    rows = hierarchy(identifiers)
    valid = ((edges[:, 0] < identifiers.shape[0])
             & (edges[:, 1] < identifiers.shape[0]))
    pairs = edges[valid]
    left, right = pairs[:, 0], pairs[:, 1]
    rows["edges"] = int(pairs.shape[0])
    for depth, name in ((1, "level1"), (2, "level2"), (3, "level3")):
        observed = share(identifiers, left, right, depth)
        # the same number of random pairs is the null the share has to beat
        random_left = rng.integers(0, n_items, size=left.shape[0])
        random_right = rng.integers(0, n_items, size=left.shape[0])
        null = share(identifiers, random_left, random_right, depth)
        rows["neighbour_share_" + name] = observed
        rows["random_share_" + name] = null
        rows["enrichment_" + name] = observed / null if null > 0 else None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=[],
                        help="a results/stage2_RQ-VAE/<arm>/item_sids.json to analyse")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    frame = pd.read_parquet(TRAIN, columns=["history", "target"])
    n_items = 24587
    edges = behaviour_edges(frame, n_items)
    print("behaviour edges:", edges.shape[0])

    results = {}
    rng = np.random.default_rng(SEED)
    results["frozen baseline"] = analyse(BASELINE, edges, n_items, rng)
    for arm in arguments.arm:
        path = REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"
        if not path.is_file():
            print("missing:", path)
            continue
        results[arm] = analyse(path, edges, n_items, rng)

    header = ("%-30s %8s %8s %8s %8s %8s %8s %8s" %
              ("arm", "L1 neb", "L2 neb", "L3 neb", "L2 pref", "L3 comb",
               "L3 coll", "L1 max"))
    print(header)
    for name, row in results.items():
        print("%-30s %8.4f %8.4f %8.4f %8d %8d %8.4f %8.4f" % (
            name, row["neighbour_share_level1"], row["neighbour_share_level2"],
            row["neighbour_share_level3"], row["level2_prefixes"],
            row["level3_combinations"], row["level3_collision"],
            row["level1_usage_max_share"]))
    print()
    print("%-30s %8s %8s %8s" % ("arm", "L1 enr", "L2 enr", "L3 enr"))
    for name, row in results.items():
        print("%-30s %8.3f %8.3f %8.3f" % (
            name, row["enrichment_level1"], row["enrichment_level2"],
            row["enrichment_level3"]))

    out = Path(arguments.out) if arguments.out else (OUT / "neighbour_preservation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nwritten:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
