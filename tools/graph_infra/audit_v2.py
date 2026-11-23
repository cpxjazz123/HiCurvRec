"""Isolation audit of the frozen v2 banks (issue #246 C6, C7, C9).

Answers three different questions that must not be mixed:

* **user-disjoint** — the evaluation user sets never enter the visible graph;
* **pair-disjoint (strict)** — the undirected keys of the strict evaluation banks appear in no
  training bank, in no probe and in no other evaluation split;
* **user-isolated but edge-seen (relaxed)** — reported separately, never merged into strict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets, config, graph_build, isolation, splits  # noqa: E402

ROOT = config.ASSET_ROOT
OUT = REPO / "results" / "research" / "infra246" / "isolation_audit.json"


def bank_keys(pattern: str) -> dict:
    result = {}
    for path in sorted((ROOT / "pairs").glob(pattern)):
        result[path.stem] = isolation.pair_keys(np.load(path)["pairs"])
    return result


def main() -> int:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    held = assets.load_sparse(ROOT / "graph" / "held_out_support.npz")
    train = bank_keys("train_*.npz")
    strict = bank_keys("strict_*.npz")
    relaxed = bank_keys("relaxed_*.npz")
    probe = isolation.pair_keys(np.load(ROOT / "pairs" / "probe.npz")["pairs"])

    train_union = set()
    for keys in train.values():
        train_union |= keys
    strict_union = set()
    for keys in strict.values():
        strict_union |= keys
    relaxed_union = set()
    for keys in relaxed.values():
        relaxed_union |= keys

    strict_names = sorted(strict)
    cross = {}
    for index, left in enumerate(strict_names):
        for right in strict_names[index + 1:]:
            cross[f"{left}&{right}"] = len(strict[left] & strict[right])
    train_cross = {}
    for left in sorted(train):
        for right in sorted(train):
            if left < right:
                train_cross[f"{left}&{right}"] = len(train[left] & train[right])

    sequences = graph_build.sequences_from_parquet(config.TRAIN)
    users = np.array(sorted(sequences), dtype=np.int64)
    blob = np.load(ROOT / "graph" / "split_users.npz")
    split = {name: blob[name] for name in splits.NAMES}
    audit = splits.audit(split, users)

    # per-bank strata: centre degree and the number of typed instances owned by the centre
    counts = np.load(ROOT / "motifs" / "motif_counts.npz")
    instances_per_centre = counts["instances"]
    degree = np.asarray(visible.getnnz(axis=1)).ravel()
    strata = {}
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        pairs = np.load(path)["pairs"]
        centres = np.unique(pairs[:, 0])
        strata[path.stem] = {
            "pairs": int(len(pairs)),
            "centres": int(len(centres)),
            "centre_degree_median": float(np.median(degree[centres])) if len(centres) else None,
            "centre_degree_p10": float(np.percentile(degree[centres], 10)) if len(centres) else None,
            "centre_degree_p90": float(np.percentile(degree[centres], 90)) if len(centres) else None,
            "centre_instances_median": float(np.median(instances_per_centre[centres]))
            if len(centres) else None,
            "group": str(np.load(path)["group"]),
        }

    payload = {
        "user_isolation": {"audit": audit,
                           "statement": "struct_val / struct_test users never enter the visible graph"},
        "pair_isolation_strict": {
            "strict_vs_train_overlap": len(strict_union & train_union),
            "strict_vs_probe_overlap": len(strict_union & probe),
            "strict_cross_split_overlap": cross,
            "statement": "an undirected key of a strict evaluation bank appears in no training bank, "
                         "no probe and no other evaluation split",
        },
        "train_cross_group_overlap": train_cross,
        "relaxed_reported_separately": {
            "relaxed_vs_train_overlap": len(relaxed_union & train_union),
            "relaxed_vs_strict_overlap": len(relaxed_union & strict_union),
            "statement": "user-isolated but edge-seen pairs are listed separately and are not "
                         "treated as unseen",
        },
        "held_out_support": {"entries": int(held.nnz // 2), "symmetric": (held != held.T).nnz == 0},
        "bank_strata": strata,
        "cross_group_multi_label": {
            "policy": "a pair may appear in several relation groups; the groups are relation "
                      "types, not disjoint labels, and no pair is weighted twice inside one group",
            "train_pairs_in_more_than_one_group": sum(
                count for name, count in train_cross.items() if count),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "strict_vs_train": payload["pair_isolation_strict"]["strict_vs_train_overlap"],
        "strict_vs_probe": payload["pair_isolation_strict"]["strict_vs_probe_overlap"],
        "strict_cross": payload["pair_isolation_strict"]["strict_cross_split_overlap"],
        "user_disjoint": audit["disjoint"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
