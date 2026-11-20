"""Issue #229 §2.2: where the teacher helps and where it does not.

The issue asks for the structural environments in which the Teacher has evidence
and those in which it does not.  Using the existing checkpoints (no retraining),
this stratifies the held-out edge task by

  * closure: common neighbours in the observed graph (0 vs >= 1),
  * degree: quartiles of max(degree) of the pair,
  * community: same vs different connected component of the observed graph,
  * direction: reciprocal observed edge vs one-way,

and reports C (real topology), A (semantics) and D (degree-preserving rewire) AUC
in each slice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher3 as gt3  # noqa: E402


def scores_for(variant, features, adjacency, positive, negative, device):
    variant.to(device).eval()
    with torch.no_grad():
        state = variant.encode(features, adjacency)
        left = torch.from_numpy(positive[:, 0]).to(device)
        right = torch.from_numpy(positive[:, 1]).to(device)
        target = torch.from_numpy(negative).to(device)
        pos = variant.edge_head(variant.pair(state, left, right)).cpu().numpy()
        neg = variant.edge_head(variant.pair(state, left, target)).cpu().numpy()
    return np.stack([pos, neg], axis=1)


def auc_of(scores):
    labels = np.concatenate([np.ones(scores.shape[0]), np.zeros(scores.shape[0])])
    return float(gt3.auc(np.concatenate([scores[:, 0], scores[:, 1]]), labels))


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    positive = data["test_positive"]
    negative = data["test_negative"]
    edges = data["message_edges"]
    n = data["n_items"]

    symmetric = sp.csr_matrix(
        (np.ones(2 * len(edges)),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n))
    symmetric.data[:] = 1.0
    symmetric.sum_duplicates()
    common = np.asarray(symmetric[positive[:, 0]].multiply(
        symmetric[positive[:, 1]]).sum(1)).ravel()
    degree = np.asarray(symmetric.sum(1)).ravel()
    strength = np.maximum(degree[positive[:, 0]], degree[positive[:, 1]])
    quantiles = np.quantile(strength, [0.25, 0.5, 0.75])
    _, component = sp.csgraph.connected_components(symmetric, directed=False)
    same_component = component[positive[:, 0]] == component[positive[:, 1]]
    observed = set(map(tuple, edges.tolist()))
    reciprocal = np.array([(int(b), int(a)) in observed for a, b in positive.tolist()])

    masks = {
        "closure:cn0": common == 0,
        "closure:cn>=1": common >= 1,
        "degree:q1": strength <= quantiles[0],
        "degree:q2": (strength > quantiles[0]) & (strength <= quantiles[1]),
        "degree:q3": (strength > quantiles[1]) & (strength <= quantiles[2]),
        "degree:q4": strength > quantiles[2],
        "community:same": same_component,
        "community:different": ~same_component,
        "direction:one_way": ~reciprocal,
        "direction:reciprocal": reciprocal,
    }

    torch.manual_seed(226)
    c = gt3.build_variant("C_residual")
    gt3.load_checkpoint(gt3.OUT / "C_residual_edge_main_seed226.pth", c)
    a = gt3.build_variant("A_embedding")
    gt3.load_checkpoint(gt3.OUT / "A_embedding_edge_main_seed226.pth", a)
    d = gt3.build_variant("D_shuffled")
    gt3.load_checkpoint(gt3.OUT / "D_shuffled_edge_main_seed226.pth", d)
    shuffled = torch.sparse_coo_tensor(
        torch.from_numpy(np.load(gt3.OUT / "shuffled_edges.npz")["index"].T.astype(np.int64)),
        torch.ones(len(np.load(gt3.OUT / "shuffled_edges.npz")["index"]),
                   dtype=torch.float32),
        size=(n, n)).coalesce().to(device)

    scores = {
        "C_residual": scores_for(c, data["features"], data["adjacency"], positive,
                                 negative, device),
        "A_embedding": scores_for(a, data["features"], None, positive, negative, device),
        "D_shuffled": scores_for(d, data["features"], shuffled, positive, negative, device),
    }
    report = {"test_pairs": int(len(positive)), "quantiles": [float(q) for q in quantiles],
              "slices": {}, "overall": {}}
    for name, scores_ in scores.items():
        report["overall"][name] = {"auc": auc_of(scores_)}
    for label, mask in masks.items():
        if mask.sum() < 50:
            report["slices"][label] = {"pairs": int(mask.sum()), "skipped": True}
            continue
        entry = {"pairs": int(mask.sum())}
        for name, scores_ in scores.items():
            entry[name] = auc_of(scores_[mask])
        entry["C_minus_A"] = entry["C_residual"] - entry["A_embedding"]
        entry["C_minus_D"] = entry["C_residual"] - entry["D_shuffled"]
        report["slices"][label] = entry
    out = HERE / "teacher" / "opt" / "structure_environment.json"
    out.write_text(json.dumps(report, indent=1))
    lines = ["# Teacher 在哪些结构环境有用(#229 §2.2)", "",
             f"held-out 对: {report['test_pairs']};度数分位: "
             + ", ".join(f"{q:.0f}" for q in report["quantiles"]), "",
             "| 环境 | 对数 | C 真图 | A 语义 | D 重接线 | C−A | C−D |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in report["slices"].items():
        if entry.get("skipped"):
            lines.append(f"| {label} | {entry['pairs']} | — | — | — | — | — |")
            continue
        lines.append(f"| {label} | {entry['pairs']} | {entry['C_residual']:.4f} | "
                     f"{entry['A_embedding']:.4f} | {entry['D_shuffled']:.4f} | "
                     f"{entry['C_minus_A']:+.4f} | {entry['C_minus_D']:+.4f} |")
    lines += ["", "整体:" + ", ".join(f"{name} {entry['auc']:.4f}"
                                     for name, entry in report["overall"].items())]
    (HERE / "REPORT_STRUCTURE_ENVIRONMENT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
