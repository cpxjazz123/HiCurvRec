"""Issue #221 evaluation: does the hard SID carry directed structure?

Reported per arm:
  * prefix retention (all_share) per family and depth, with the analytic
    code-frequency null subtracted;
  * directed edge retention at L1/L1+L2, split by whether the reverse edge exists;
  * a role readout: a linear probe on the SID prefix one-hot must recover the
    directed orbit of a node, which is the checkable way a hard SID can carry
    direction (prefix *equality* alone is symmetric and cannot);
  * codebook usage, collision and, when present, Stage3 Recall@10.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402

REPO = HERE.parents[2]
ARMS = ("D0", "D1", "D2", "D3", "D4")
FAMILIES = ("K3", "P3", "C4", "P4", "K1_3")


def load_sids(arm: str) -> np.ndarray:
    path = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_drq_{arm}" / "item_sids.json"
    payload = json.loads(path.read_text())
    return np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)


def prefix_features(codes: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """(instances, k, 3*256) one-hot of the three prefix levels, per node."""
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, 3 * 256), dtype=np.float32)
    for level in range(3):
        slice_codes = codes[nodes, level]
        np.put_along_axis(out[:, :, level * 256:(level + 1) * 256],
                          slice_codes[:, :, None], 1.0, axis=2)
    return out.reshape(len(nodes), -1)


def role_probe(codes: np.ndarray, families: dict, vocab: dict) -> dict:
    """Linear probe: SID prefix one-hot -> directed orbit, train on train split."""
    from sklearn.linear_model import LogisticRegression

    report = {}
    for family, (nodes, orbits, subtypes) in families.items():
        probe = LogisticRegression(max_iter=200, multi_class="multinomial", n_jobs=1)
        features = prefix_features(codes, nodes)
        labels = np.array([vocab["orbits_of"][int(s)][int(o)] if int(s) in vocab["orbits_of"]
                           else -1 for s, o in zip(subtypes, orbits[:, 0])])
        # one label per instance is not enough: use the first node's orbit only as a
        # smoke probe, and record it as such
        mask = labels >= 0
        split = np.zeros(len(nodes), dtype=bool)
        split[: len(nodes) // 2] = True
        train, test = mask & split, mask & ~split
        if train.sum() < 50 or test.sum() < 50 or len(np.unique(labels[train])) < 2:
            report[family] = None
            continue
        probe.fit(features[train], labels[train])
        accuracy = float(probe.score(features[test], labels[test]))
        report[family] = {"probe_accuracy": accuracy, "train": int(train.sum()),
                          "test": int(test.sum()), "classes": int(len(np.unique(labels)))}
    values = [row["probe_accuracy"] for row in report.values() if row]
    return {"per_family": report, "mean": float(np.mean(values)) if values else None}


def family_retention(codes: np.ndarray, families: dict) -> dict:
    out = {}
    for family, (nodes, _orbits, _subtypes) in families.items():
        row = {}
        counts = np.bincount(codes[:, 0], minlength=256).astype(np.float64)
        null = float(((counts / counts.sum()) ** 2).sum())
        for depth, label in ((1, "L1"), (2, "L1L2"), (3, "full")):
            sliced = codes[nodes, :depth]
            equal = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
            pairs = np.triu_indices(nodes.shape[1], 1)
            share = float(equal[:, pairs[0], pairs[1]].mean())
            row[label] = {"all_share": share, "null": null, "gain": share - null}
        out[family] = row
    return out


def directed_edges(codes: np.ndarray) -> dict:
    edges = pd.read_parquet(REPO / "results" / "research" / "behavior_graph_directed"
                            / "edges.parquet")
    sources = edges["source"].to_numpy()
    targets = edges["target"].to_numpy()
    weights = edges["weight"].to_numpy().astype(np.float64)
    reciprocal = edges.set_index(["source", "target"]).index.isin(
        edges.set_index(["target", "source"]).index)
    out = {}
    for depth, label in ((1, "L1"), (2, "L1L2")):
        equal = (codes[sources, :depth] == codes[targets, :depth]).all(-1).astype(np.float64)
        out[f"edge_{label}"] = float(equal.mean())
        out[f"edge_{label}_weighted"] = float((equal * weights).sum() / weights.sum())
        out[f"edge_{label}_reciprocal"] = float(equal[reciprocal].mean())
        out[f"edge_{label}_oneway"] = float(equal[~reciprocal].mean())
    return out


def describe(arm: str, families: dict, vocab: dict) -> dict:
    codes = load_sids(arm)
    record = {"arm": arm, "items": int(len(codes))}
    record["families"] = family_retention(codes, families)
    record["directed_edges"] = directed_edges(codes)
    record["role_probe"] = role_probe(codes, families, vocab)
    unique = len(np.unique(codes[:, :3], axis=0))
    record["collision"] = float(1.0 - unique / len(codes))
    buckets = np.bincount(codes[:, 0], minlength=256) / len(codes)
    record["max_bucket_L1"] = float(buckets.max())
    record["used_L1"] = int((buckets > 0).sum())
    metrics = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_drq_{arm}" / "trq_metrics.json"
    record["head"] = json.loads(metrics.read_text())["heldout_scores"] if metrics.is_file() else None
    stage3 = REPO / "results" / "stage3_T5Train" / f"TIGER_drq_{arm}" / "test_final.json"
    record["recall"] = (float(json.loads(stage3.read_text())["test_recall@10"])
                        if stage3.is_file() else None)
    return record


def main() -> None:
    vocab = load_vocabulary()
    families = load_pool("heldout", vocab)
    panel = {}
    for arm in ARMS:
        path = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_drq_{arm}" / "item_sids.json"
        if path.is_file():
            panel[arm] = describe(arm, families, vocab)
    (HERE / "evaluation.json").write_text(json.dumps(panel, indent=1))
    for arm, row in panel.items():
        head = row.get("head") or {}
        print(f"arm {arm}: subtypeF1={head.get('subtype', {}).get('macro_f1')} "
              f"orbitF1={head.get('role', {}).get('macro_over_families')} "
              f"collision={row['collision']:.3f} edgeL1={row['directed_edges']['edge_L1']:.4f} "
              f"probe={row['role_probe']['mean']}")


if __name__ == "__main__":
    main()
