"""Issue #220 evaluation: SID structure retention per Graphlet plus the head's F1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "template"))

import trq  # noqa: E402
from graphlet_roles import TYPES, load_instances  # noqa: E402

REPO = HERE.parents[2]
ARMS = ("A", "B", "C", "D", "E")


def load_sids(arm: str) -> np.ndarray:
    path = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_trq_{arm}" / "item_sids.json"
    payload = json.loads(path.read_text())
    return np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)


def null_share(codes: np.ndarray, nodes: np.ndarray, depth: int) -> float:
    """Analytic pair-share expectation from the code frequencies (Issue #218)."""
    counts = np.bincount(codes[:, 0], minlength=int(codes[:, 0].max()) + 1).astype(np.float64)
    share = counts / counts.sum()
    expected = float((share ** 2).sum())
    return expected


def instance_pairs(codes: np.ndarray, nodes: np.ndarray, depth: int) -> np.ndarray:
    sliced = codes[nodes, :depth]
    equal = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    pairs = np.triu_indices(nodes.shape[1], 1)
    return equal[:, pairs[0], pairs[1]].mean(1)


def bootstrap(values: np.ndarray, rounds: int = 2000, seed: int = 220) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(rounds)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def global_fidelity(codes: np.ndarray) -> dict:
    payload = np.load(REPO / "results" / "research" / "behavior_graph" / "graph.npz",
                      allow_pickle=False)
    rows, cols = payload["rows"].astype(np.int64), payload["cols"].astype(np.int64)
    keep = rows < cols
    sources, targets = rows[keep], cols[keep]
    out = {}
    for depth, label in ((1, "L1"), (2, "L1L2")):
        out[f"edge_{label}"] = float((codes[sources, :depth] == codes[targets, :depth]).all(-1).mean())
    rng = np.random.default_rng(7)
    a = rng.integers(0, len(codes), 200000)
    b = rng.integers(0, len(codes), 200000)
    same = a == b
    a, b = a[~same], b[~same]
    keys = np.sort(sources * len(codes) + targets)
    test = a * len(codes) + b
    position = np.clip(np.searchsorted(keys, test), 0, len(keys) - 1)
    is_edge = keys[position] == test
    a, b = a[~is_edge], b[~is_edge]
    for depth, label in ((1, "L1"), (2, "L1L2")):
        out[f"nonedge_{label}"] = float((codes[a, :depth] == codes[b, :depth]).all(-1).mean())
    return out


def describe(arm: str, instances: dict) -> dict:
    codes = load_sids(arm)
    record = {"arm": arm, "items": int(len(codes))}
    per_type = {}
    for name, (nodes, _orbits) in instances.items():
        row = {}
        for depth, label in ((1, "L1"), (2, "L1L2"), (3, "full")):
            values = instance_pairs(codes, nodes, depth)
            low, high = bootstrap(values)
            row[label] = {"pair_share": float(values.mean()),
                          "null": null_share(codes, nodes, depth),
                          "gain": float(values.mean() - null_share(codes, nodes, depth)),
                          "ci95": [low, high]}
        per_type[name] = row
    record["per_type"] = per_type
    record["global"] = global_fidelity(codes)
    unique = len(np.unique(codes[:, :3], axis=0))
    record["collision"] = float(1.0 - unique / len(codes))
    buckets = np.bincount(codes[:, 0], minlength=256) / len(codes)
    record["max_bucket_L1"] = float(buckets.max())
    record["used_L1"] = int((buckets > 0).sum())
    metrics = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_trq_{arm}" / "trq_metrics.json"
    record["head"] = json.loads(metrics.read_text())["heldout_scores"] if metrics.is_file() else None
    stage3 = REPO / "results" / "stage3_T5Train" / f"TIGER_trq_{arm}" / "test_final.json"
    record["recall"] = float(json.loads(stage3.read_text())["test_recall@10"]) if stage3.is_file() else None
    return record


def main() -> None:
    instances = load_instances("heldout")
    panel = {}
    for arm in ARMS:
        path = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_trq_{arm}" / "item_sids.json"
        if path.is_file():
            panel[arm] = describe(arm, instances)
    (HERE / "evaluation.json").write_text(json.dumps(panel, indent=1))
    for arm, row in panel.items():
        head = row.get("head") or {}
        type_f1 = head.get("type", {}).get("macro_f1")
        role_f1 = head.get("role", {}).get("macro_over_types")
        gains = " ".join(f"{name}:{row['per_type'][name]['L1']['gain']:+.4f}" for name in TYPES)
        print(f"arm {arm}: typeF1={type_f1} roleF1={role_f1} collision={row['collision']:.3f} "
              f"edgeL1={row['global']['edge_L1']:.4f} gains {gains}")


if __name__ == "__main__":
    main()
