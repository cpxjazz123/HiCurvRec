"""Build seed-42 item co-purchase graph from train.parquet only."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "results/stage0_build_parquet/train.parquet"
OUTPUT = ROOT / "results/crowding/graph_s42.npz"
SEED = 42
K_VALUES = (5, 20, 50)


def main() -> None:
    digest = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    frame = pd.read_parquet(TRAIN)
    if "user" not in frame.columns or "target" not in frame.columns:
        raise ValueError("train.parquet must contain user and target columns")
    user_items = frame.groupby("user", sort=False)["target"].apply(lambda x: sorted(set(map(int, x))))
    frequency: Counter[int] = Counter()
    co_count: Counter[tuple[int, int]] = Counter()
    for items in user_items:
        frequency.update(items)
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                co_count[(a, b)] += 1
                co_count[(b, a)] += 1
    n_items = int(max(frequency, default=-1) + 1)
    neighbors = np.full((n_items, max(K_VALUES)), -1, dtype=np.int64)
    weights = np.zeros((n_items, max(K_VALUES)), dtype=np.float32)
    degrees = np.zeros(n_items, dtype=np.int64)
    freqs = np.zeros(n_items, dtype=np.int64)
    for item, value in frequency.items():
        freqs[item] = value
    by_item: dict[int, list[tuple[int, int]]] = {}
    for (a, b), count in co_count.items():
        by_item.setdefault(a, []).append((b, count))
    rng = np.random.default_rng(SEED)
    for item, values in by_item.items():
        # Stable deterministic tie-breaking; seed is recorded for reproducibility.
        rng.random(len(values))
        values.sort(key=lambda pair: (-pair[1], pair[0]))
        values = values[:max(K_VALUES)]
        degrees[item] = len(by_item[item])
        for j, (neighbor, count) in enumerate(values):
            neighbors[item, j] = neighbor
            weights[item, j] = count / max(freqs[item], 1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, neighbors=neighbors, weights=weights, degree=degrees,
                        frequency=freqs, k_values=np.asarray(K_VALUES), seed=np.asarray(SEED))
    OUTPUT.with_suffix(".json").write_text(json.dumps({"seed": SEED, "source": str(TRAIN),
        "source_sha256": digest, "item_count": n_items, "k_values": K_VALUES,
        "mean_degree": float(degrees.mean())}, indent=2) + "\n")
    print(f"output={OUTPUT}")
    print(f"items={n_items} k_values={K_VALUES} mean_degree={degrees.mean():.6f}")
    print(f"source_sha256={digest}")


if __name__ == "__main__":
    main()
