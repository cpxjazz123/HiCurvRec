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


def prefix_features(codes: np.ndarray, nodes: np.ndarray, depth: int = 1) -> np.ndarray:
    """(instances*k, depth*256) one-hot of the prefix levels, per node."""
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, depth * 256), dtype=np.float32)
    for level in range(depth):
        np.put_along_axis(out[:, :, level * 256:(level + 1) * 256],
                          codes[nodes, level][:, :, None], 1.0, axis=2)
    return out.reshape(len(nodes), -1)


def role_probe(codes: np.ndarray, families: dict, vocab: dict) -> dict:
    """Linear probe: SID prefix one-hot -> directed orbit of the first node.

    A hard SID can only carry direction if some checkable readout recovers it;
    prefix equality is symmetric, so this probe is the readout.  It is a
    measurement, not a gate: a capped sample and a linear model keep it cheap
    and identical across arms.
    """
    from sklearn.linear_model import SGDClassifier

    report = {}
    for family, (nodes, orbits, subtypes) in families.items():
        labels = np.array([vocab["orbits_of"].get(int(s), [-1])[int(o)]
                           if int(s) in vocab["orbits_of"] else -1
                           for s, o in zip(subtypes, orbits[:, 0])])
        valid = labels >= 0
        if valid.sum() < 200:
            report[family] = None
            continue
        features = prefix_features(codes, nodes)[valid]
        target = labels[valid]
        rng = np.random.default_rng(221)
        order = rng.permutation(len(target))
        cut = int(0.7 * len(order))
        train, test = order[:cut], order[cut:]
        if len(np.unique(target[train])) < 2:
            report[family] = None
            continue
        model = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=10, random_state=221)
        model.fit(features[train], target[train])
        report[family] = {"probe_accuracy": float(model.score(features[test], target[test])),
                          "train": int(len(train)), "test": int(len(test)),
                          "chance": float(1.0 / len(np.unique(target)))}
    values = [row["probe_accuracy"] for row in report.values() if row]
    return {"per_family": report, "mean": float(np.mean(values)) if values else None}


def family_retention(codes: np.ndarray, families: dict) -> dict:
    out = {}
    for family, (nodes, _orbits, _subtypes) in families.items():
        row = {}
        for depth, label in ((1, "L1"), (2, "L1L2"), (3, "full")):
            # The null is the chance that two random items share the PREFIX of
            # this depth, from that depth's own prefix counts -- using the L1
            # frequency for every depth (the earlier version) made the deep
            # gains meaningless.
            _, prefix_counts = np.unique(codes[:, :depth], axis=0, return_counts=True)
            frequencies = prefix_counts.astype(np.float64) / prefix_counts.sum()
            null = float((frequencies ** 2).sum())
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
