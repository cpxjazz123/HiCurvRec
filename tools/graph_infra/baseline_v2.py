"""Recompute the TIGER baseline on the v2 banks, under the unified caliber (issue #246 §7).

The confirmed table is the one Stage3 actually read; the alternatives are reported separately
and never mixed.  Every number is recomputed from the frozen assets and the SID tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import config, metrics  # noqa: E402

ROOT = config.ASSET_ROOT
OUT = REPO / "results" / "research" / "infra246" / "tiger_baseline_v2.json"


def load_sid(path: Path) -> np.ndarray:
    table = json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(int(key) for key in table)
    return np.array([list(table[str(key)]) for key in keys], dtype=np.int64)


def prefix_scores(codes_left, codes_right, depth: int = 4) -> np.ndarray:
    """Number of leading positions that match, per pair."""
    left = np.asarray(codes_left, dtype=np.int64)
    right = np.asarray(codes_right, dtype=np.int64)
    depth = min(depth, left.shape[1], right.shape[1])
    matches = np.zeros(left.shape[0], dtype=np.int64)
    for position in range(depth):
        same = (left[:, position] == right[:, position]) & (matches == position)
        matches += same.astype(np.int64)
    return matches


def spearman_on_finite(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    return metrics.spearman(x[mask], y[mask])


def evaluate(sid: np.ndarray, name: str, banks: list) -> dict:
    report = {"sid": name, "groups": {}}
    for label, path in banks:
        blob = np.load(path)
        pairs = blob["pairs"]
        left = sid[pairs[:, 0]]
        right = sid[pairs[:, 1]]
        matches = prefix_scores(left, right)
        group = str(blob["group"])
        entry = {
            "pairs": int(len(pairs)),
            "prefix@1": float((matches >= 1).mean()),
            "prefix@2": float((matches >= 2).mean()),
            "prefix@3": float((matches >= 3).mean()),
            "mean_prefix_length": float(matches.mean()),
        }
        for score_name in ("two_hop_count", "ppr", "jaccard_out", "bhattacharyya_out",
                           "hop_full_visible"):
            if score_name not in blob.files:
                continue
            score = np.asarray(blob[score_name], dtype=np.float64)
            entry[f"spearman_{score_name}_vs_prefix"] = spearman_on_finite(score, matches)
        # multi-subgraph view: one centre (the pair source) may own several subgraphs
        centres, inverse = np.unique(pairs[:, 0], return_inverse=True)
        per_centre = [matches[inverse == index] for index in range(len(centres))]
        similarity = metrics.subgraph_summary(
            [values.astype(np.float64) / 4.0 for values in per_centre], threshold=0.5,
            direction="maximize")
        distortion = metrics.subgraph_summary(
            [1.0 - values.astype(np.float64) / 4.0 for values in per_centre], threshold=0.5,
            direction="minimize")
        entry["multi_subgraph"] = {
            "similarity": {key: similarity[key] for key in
                           ("centers", "subgraphs", "subgraph_pass_fraction",
                            "center_all_pass_fraction", "center_any_pass_fraction", "worst", "p90",
                            "centers_without_valid_subgraph")},
            "distortion": {key: distortion[key] for key in
                           ("centers", "subgraphs", "subgraph_pass_fraction",
                            "center_all_pass_fraction", "center_any_pass_fraction", "worst", "p90",
                            "centers_without_valid_subgraph")},
        }
        report["groups"][f"{label}/{group}"] = entry
    return report


def main() -> int:
    banks = []
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        banks.append((path.stem, path))
    for path in sorted((ROOT / "pairs").glob("strict_*.npz")):
        banks.append((path.stem, path))
    if not banks:
        raise SystemExit("no v2 banks found; run build_assets_v2.py first")
    confirmed = load_sid(REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json")
    alternatives = {
        "checkpoint_reexport": np.load(REPO / "results" / "research" / "infra246" / "tiger_reexport"
                                        / "sids_for_hgrec.npy"),
        "legacy_npy": np.load(REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "sids_for_hgrec.npy"),
    }
    payload = {
        "confirmed_sid": "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json (the file Stage3 read)",
        "confirmed": evaluate(confirmed, "item_sids_json", banks),
    }
    for name, matrix in alternatives.items():
        payload[name] = evaluate(matrix, name, banks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {key: {group: {"prefix@1": entry["prefix@1"], "prefix@2": entry["prefix@2"],
                             "prefix@3": entry["prefix@3"]}
                     for group, entry in payload["confirmed"]["groups"].items()}
               for key in ("confirmed",)}
    print(json.dumps(summary, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
