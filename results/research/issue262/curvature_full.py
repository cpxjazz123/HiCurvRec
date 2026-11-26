"""Issue #262: full predicted-vs-target curvature table and stratified diagnostic.

Uses the actual exported hard SID + its checkpoints' codebooks and the frozen
equal-weight three-expert metric. No model forward, no training, no optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
RAW = OUT / "_raw"
sys.path.insert(0, str(ROOT))

from tools.issue251.full_evaluation import _moments  # noqa: E402
from tools.issue253.evaluation_reference import (  # noqa: E402
    fixed_three_paired_numpy, fixed_three_squared_numpy)

ASSETS = ROOT / "assets"
ARMS = {
    100: "TIGER_gnn_attention_sectional_rq",
    101: "TIGER_three_expert_attention_rq",
    102: "TIGER_module_budget_prefix_rq",
    103: "TIGER_periodic_budget_prefix_rq",
    104: "TIGER_role_conditioned_l1_rq",
    105: "TIGER_role_conditioned_l3_rq",
    106: "TIGER_role_conditioned_l2_rq",
    107: "TIGER_topology_residual_l2_rq",
}
ROLE_FILE = ROOT / "stage2_RQ-VAE/TIGER_topology_residual_l2_rq/role_table.json"
DIRECTED = ROOT / "results/research/behavior_graph_directed"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_final(arm: str) -> Path:
    paths = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{arm}: expected one test_final")
    return paths[0]


def q_path(index: int) -> Path:
    return RAW / "q" / f"{index}.npz"


def load_q(index: int) -> dict:
    path = q_path(index)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def roles() -> np.ndarray:
    payload = json.loads(ROLE_FILE.read_text())
    values = np.asarray(payload["table"], dtype=np.int64)
    if values.shape != (24587,):
        raise ValueError("role table must cover the catalogue")
    return values


def degrees() -> np.ndarray:
    edges = pd.read_parquet(DIRECTED / "edges.parquet", columns=["source", "target"])
    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    in_degree = np.bincount(target, minlength=24587)
    out_degree = np.bincount(source, minlength=24587)
    return in_degree + out_degree


def sign_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    truth = np.where(target < 0, -1, np.where(target > 0, 1, 0))
    pred = np.where(predicted < 0, -1, np.where(predicted > 0, 1, 0))
    out = {"accuracy": float((truth == pred).mean())}
    for name, value in (("negative", -1), ("zero", 0), ("positive", 1)):
        actual = truth == value
        called = pred == value
        tp = int((actual & called).sum())
        fp = int((~actual & called).sum())
        fn = int((actual & ~called).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        out[name] = {"support": int(actual.sum()), "predicted": int(called.sum()),
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1}
    recalls = [out[n]["recall"] for n in ("negative", "zero", "positive")
               if out[n]["recall"] is not None]
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else None
    return out


def stratified(target: np.ndarray, predicted: np.ndarray, keys: np.ndarray,
               name: str) -> dict:
    result = {}
    for key in sorted(set(int(x) for x in keys.tolist())):
        mask = keys == key
        if not mask.any():
            continue
        error = np.abs(predicted[mask] - target[mask])
        result[str(key)] = {
            "count": int(mask.sum()),
            "mae": float(error.mean()),
            "rmse": float(np.sqrt(np.mean((predicted[mask] - target[mask]) ** 2))),
            "target_mean": float(target[mask].mean()),
            "prediction_mean": float(predicted[mask].mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    index = args.index
    if index not in ARMS:
        raise SystemExit("unknown arm")
    arm = ARMS[index]
    final = find_final(arm)
    fd = json.loads(final.read_text())
    q_blob = load_q(index)
    q = q_blob["q"].astype(np.float64)
    scale = float(q_blob["scale"])
    sid = q_blob["sid3"].astype(np.int64)
    labels = np.load(ASSETS / "node_moments.npz", allow_pickle=False)["component"]
    names = [str(x) for x in np.load(ASSETS / "index.npz", allow_pickle=False)["shard_names"]]
    if labels.shape != (len(q),) or q.shape != (24587, 32):
        raise ValueError("coordinate/catalogue identity mismatch")
    with threadpool_limits(limits=1):
        moments = _moments(q, labels, scale, fixed_three_squared_numpy, 512)
    predicted_parts = []
    target_parts = []
    centre_parts = []
    excluded = 0
    for name in names:
        with np.load(ASSETS / "shards" / name, allow_pickle=False) as blob:
            valid = blob["valid"]
            excluded += int((~valid).sum())
            centre = blob["centre"][valid].astype(np.int32)
            b = blob["b"][valid].astype(np.int32)
            c = blob["c"][valid].astype(np.int32)
            target = blob["curvature"][valid].astype(np.float64)
        if not len(centre):
            continue
        pair = fixed_three_paired_numpy(q[b], q[c], scale=scale)
        predicted = (moments[centre] - 0.5 * (moments[b] + moments[c]) + 0.25 * pair)
        predicted_parts.append(predicted)
        target_parts.append(target)
        centre_parts.append(centre)
        print("[curvature] %s rows=%d cumulative=%d" % (
            name, len(centre), sum(len(x) for x in predicted_parts)), flush=True)
    predicted = np.concatenate(predicted_parts)
    target = np.concatenate(target_parts)
    centre = np.concatenate(centre_parts)
    if len(predicted) != 28391948 or excluded != 0:
        raise ValueError("full coverage mismatch: %d valid, %d excluded" % (len(predicted), excluded))
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite full curve")
    pearson = float(pearsonr(target, predicted).statistic)
    spearman = float(spearmanr(target, predicted).statistic)
    rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
    mae = float(np.mean(np.abs(predicted - target)))
    role_table = roles()
    degree_table = degrees()
    degree_bins = np.quantile(degree_table, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    degree_bins = np.unique(degree_bins)
    degree_key = np.clip(np.digitize(degree_table, degree_bins[1:-1], right=True),
                         0, len(degree_bins) - 2)
    result = {
        "schema": "genrec.issue262.curvature_full/1",
        "arm": arm,
        "index": index,
        "identity": {
            "final": str(final),
            "final_sha256": sha(final),
            "sid_sha256": sha(Path(fd["sid_path"])),
            "stage2_checkpoint": str(fd["stage2_checkpoint"]),
            "stage2_checkpoint_sha256": sha(Path(fd["stage2_checkpoint"])),
            "q_file": str(q_path(index)),
            "q_sha256": sha(q_path(index)),
            "scale": scale,
            "scale_fitted_on_evaluation": False,
        },
        "coverage": {"count": int(len(predicted)), "excluded": int(excluded),
                      "expected": 28391948, "shards": len(names)},
        "metrics": {
            "pearson": pearson,
            "spearman": spearman,
            "rmse": rmse,
            "mae": mae,
            "sign": sign_metrics(target, predicted),
            "target_quantiles": {str(q): float(x) for q, x in
                                 zip((0, .01, .05, .25, .5, .75, .95, .99, 1),
                                     np.quantile(target, [0, .01, .05, .25, .5, .75, .95, .99, 1]))},
            "prediction_quantiles": {str(q): float(x) for q, x in
                                     zip((0, .01, .05, .25, .5, .75, .95, .99, 1),
                                         np.quantile(predicted, [0, .01, .05, .25, .5, .75, .95, .99, 1]))},
        },
        "role_strata": stratified(target, predicted, role_table[centre], "role"),
        "degree_strata": stratified(target, predicted, degree_key[centre], "degree"),
        "role_table": str(ROLE_FILE),
        "degree_source": str(DIRECTED / "edges.parquet"),
        "scope": "hard exported SID; equal-weight three-expert prediction; structural curvature proxy",
    }
    (OUT / f"CURVATURE_FULL_{index}.json").write_text(json.dumps(result, indent=2) + "\n")
    raw = RAW / "curvature"
    raw.mkdir(parents=True, exist_ok=True)
    raw_path = raw / f"{index}.npz"
    np.savez_compressed(raw_path, predicted=predicted.astype(np.float32),
                        target=target.astype(np.float32), centre=centre.astype(np.int32))
    result["raw"] = {"path": str(raw_path), "sha256": sha(raw_path)}
    (OUT / f"CURVATURE_FULL_{index}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"index": index, "pearson": pearson, "spearman": spearman,
                      "rmse": rmse, "mae": mae, "sign_accuracy": result["metrics"]["sign"]["accuracy"],
                      "raw": str(raw_path)}, indent=2))


if __name__ == "__main__":
    main()
