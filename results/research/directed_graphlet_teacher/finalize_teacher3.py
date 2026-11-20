"""Finish the #228 report from the saved checkpoints.

The long run was killed inside the orbit profile because a numpy array was used
as a membership set (O(N) per test).  Every grid metric was written to
`teacher/opt/graph_teacher3_report.json` before that point; this script
recomputes the missing tail from the checkpoints, which is also a check on the
recorded numbers: a mismatch is an error, not a new result.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher3 as gt3  # noqa: E402


def checkpoint_path(name: str, task: str, tag: str, seed: int) -> Path:
    return gt3.OUT / f"{name}_{task}_{tag}_seed{seed}.pth"


def load_variant(name: str, path: Path, device) -> gt3.TeacherVariant:
    variant = gt3.build_variant(name)
    gt3.load_checkpoint(path, variant)
    return variant.to(device).eval()


def main() -> None:
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    report_path = gt3.OUT / "graph_teacher3_report.json"
    report = json.loads(report_path.read_text())
    n = data["n_items"]

    stored = np.load(gt3.OUT / "shuffled_edges.npz")
    shuffled_index = stored["index"]
    shuffled_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(shuffled_index.T.astype(np.int64)),
        torch.ones(shuffled_index.shape[0], dtype=torch.float32),
        size=(n, n)).coalesce().to(device)
    adjacency = {"A_embedding": None, "B_graph": data["adjacency"],
                 "C_residual": data["adjacency"], "D_shuffled": shuffled_tensor,
                 "E_nograph": None}

    grid = {}
    for task in ("edge", "edge+direction"):
        grid[task] = {}
        for name in ("A_embedding", "B_graph", "C_residual", "D_shuffled", "E_nograph"):
            grid[task][name] = {}
            for seed in gt3.SEEDS:
                path = checkpoint_path(name, task, "main", seed)
                if not path.is_file():
                    continue
                variant = load_variant(name, path, device)
                metrics = gt3.evaluate_splits(variant, data["features"], adjacency[name],
                                              data, device)
                recorded = report["stages"][task][name][f"seed{seed}"]["splits"]["test"]["auc"]
                assert abs(metrics["test"]["auc"] - recorded) < 1e-6, \
                    (task, name, seed, metrics["test"]["auc"], recorded)
                grid[task][name][f"seed{seed}"] = {
                    "scores_test": metrics["test"]["scores"],
                    "auc": metrics["test"]["auc"],
                    "checkpoint": path.name,
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                print(f"rechecked {task} {name} seed{seed} test={metrics['test']['auc']:.4f}",
                      flush=True)

    control_path = checkpoint_path("B_graph", "edge", "control40", gt3.SEED)
    control_variant = load_variant("B_graph", control_path, device)
    control_metrics = gt3.evaluate_splits(control_variant, data["features"],
                                          data["adjacency"], data, device)
    report["control_40_step"] = {
        "steps": 40, "accum": gt3.ACCUM, "micro": gt3.MICRO,
        "splits": {split: {k: v for k, v in control_metrics[split].items() if k != "scores"}
                   for split in ("train", "valid", "test")},
        "checkpoint": control_path.name,
        "checkpoint_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        "source": "checkpoint re-evaluation; the original process was killed before "
                  "writing this entry"}

    report["baselines"] = {"raw_semantic": gt3.raw_semantic_baseline(data),
                           "high_order": gt3.high_order_gate(data, shuffled_index),
                           "degree_bands": gt3.degree_bands(data)}
    report["recognition"] = gt3.recognition_audit()
    checkpoints = {name: checkpoint_path(name, "edge", "main", gt3.SEEDS[0])
                   for name in ("C_residual", "D_shuffled")
                   if checkpoint_path(name, "edge", "main", gt3.SEEDS[0]).is_file()}
    report["orbit_probe"] = gt3.orbit_probe(data, checkpoints, device, adjacency)
    report["comparisons"] = gt3.collect_comparisons(grid)
    report["verdict"] = gt3._verdict(report)
    report["seconds"] = time.time() - started
    report["finalised_by"] = "finalize_teacher3.py"
    report_path.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["verdict"], indent=1), flush=True)
    print("written", report_path, flush=True)


if __name__ == "__main__":
    main()
