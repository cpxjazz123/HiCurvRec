"""Issue #229 section 4: the non-learned structural-heuristic control.

The contract requires that any claimed structural value be compared against
"common non-learned structural statistics / heuristics" on the *same* target
set, with the same evaluation protocol.  The high-order gate already used such
features on the graphlet pool; the edge-existence claim never did.

This script scores the held-out edge split (the same positives/negatives used by
`structure_environment.py` and the C/A/D comparison) with:

  * common neighbours (CN),
  * Adamic-Adar (AA),
  * resource allocation (RA),
  * Jaccard,
  * log degree product,
  * directed length-2 walks,

all computed from the TRAIN-observable message graph only, plus a logistic
stack fitted on the TRAIN pairs (no test information).  The teacher's C_residual
checkpoint is scored on the identical pairs and the difference is bootstrapped.

Nothing is trained on the test split; no heuristic sees a held-out edge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher3 as gt3  # noqa: E402

N_BOOTSTRAP = 300


def symmetric_csr(edges: np.ndarray, n: int) -> sp.csr_matrix:
    matrix = sp.csr_matrix(
        (np.ones(2 * len(edges)),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n))
    matrix.data[:] = 1.0
    matrix.sum_duplicates()
    return matrix


def heuristic_features(symmetric: sp.csr_matrix, directed: sp.csr_matrix,
                       rows: np.ndarray, cols: np.ndarray,
                       degree: np.ndarray) -> dict[str, np.ndarray]:
    left = symmetric[rows]
    right = symmetric[cols]
    common = np.asarray(left.multiply(right).sum(1)).ravel()
    inverse_log = 1.0 / np.log(np.clip(degree, 3.0, None))
    adamic = np.asarray(left.multiply(right) @ inverse_log).ravel()
    inverse = 1.0 / np.clip(degree, 1.0, None)
    resource = np.asarray(left.multiply(right) @ inverse).ravel()
    union = degree[rows] + degree[cols] - common
    jaccard = np.where(union > 0, common / np.maximum(union, 1.0), 0.0)
    two_step = np.asarray((directed @ directed)[rows, cols]).ravel()
    return {"cn": common, "aa": adamic, "ra": resource, "jaccard": jaccard,
            "degree_product": degree[rows] * degree[cols],
            "two_step": two_step}


def auc_of_pair_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(gt3.auc(np.concatenate([pos, neg]), labels))


def teacher_scores(device, features, adjacency, positive, negative) -> np.ndarray:
    torch.manual_seed(226)
    variant = gt3.build_variant("C_residual")
    gt3.load_checkpoint(gt3.OUT / "C_residual_edge_main_seed226.pth", variant)
    variant.to(device).eval()
    with torch.no_grad():
        state = variant.encode(features, adjacency)
        left = torch.from_numpy(positive[:, 0]).to(device)
        right = torch.from_numpy(positive[:, 1]).to(device)
        target = torch.from_numpy(negative).to(device)
        pos = variant.edge_head(variant.pair(state, left, right)).cpu().numpy()
        neg = variant.edge_head(variant.pair(state, left, target)).cpu().numpy()
    return np.concatenate([pos, neg])


def logit_stack(train_x, train_y, test_x, epochs: int = 400, lr: float = 0.1):
    x = torch.from_numpy(np.asarray(train_x, dtype=np.float32))
    y = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    mean, std = x.mean(0), x.std(0).clamp(min=1e-6)
    x = (x - mean) / std
    test = (torch.from_numpy(np.asarray(test_x, dtype=np.float32)) - mean) / std
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x @ weight + bias, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return (test @ weight + bias).numpy()


def bootstrap_delta(teacher: np.ndarray, other: np.ndarray,
                    n_pairs: int, seed: int = 226) -> dict:
    rng = np.random.default_rng(seed)
    deltas = np.empty(N_BOOTSTRAP)
    for step in range(N_BOOTSTRAP):
        index = rng.integers(0, n_pairs, n_pairs)
        positive = np.concatenate([index, index + n_pairs])
        labels = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
        deltas[step] = (gt3.auc(teacher[positive], labels)
                        - gt3.auc(other[positive], labels))
    return {"mean": float(deltas.mean()), "low": float(np.quantile(deltas, 0.025)),
            "high": float(np.quantile(deltas, 0.975)), "n": N_BOOTSTRAP}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    positive = data["test_positive"]
    negative = data["test_negative"]
    edges = data["message_edges"]
    n = data["n_items"]

    symmetric = symmetric_csr(edges, n)
    directed = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                             shape=(n, n))
    directed.sum_duplicates()
    degree = np.asarray(symmetric.sum(1)).ravel()

    test_rows = np.concatenate([positive[:, 0], positive[:, 0]])
    test_cols = np.concatenate([positive[:, 1], negative])
    test_labels = np.concatenate([np.ones(len(positive)), np.zeros(len(positive))])
    train_rows = np.concatenate([data["train_positive"][:, 0],
                                 data["train_positive"][:, 0]])
    train_cols = np.concatenate([data["train_positive"][:, 1],
                                 data["train_negative"]])

    test_features = heuristic_features(symmetric, directed, test_rows, test_cols,
                                       degree)
    train_features = heuristic_features(symmetric, directed, train_rows, train_cols,
                                        degree)

    teacher = teacher_scores(device, data["features"], data["adjacency"],
                             positive, negative)

    report = {"test_pairs": int(len(positive)), "n_items": int(n),
              "heuristics": {}, "teacher": {"auc": auc_of_pair_scores(
                  teacher[:len(positive)], teacher[len(positive):])},
              "delta_vs_teacher": {}}
    for name, values in test_features.items():
        pos, neg = values[:len(positive)], values[len(positive):]
        report["heuristics"][name] = {"auc": auc_of_pair_scores(pos, neg)}

    stack_train = np.stack([np.log1p(np.maximum(train_features[k], 0.0))
                            for k in test_features], axis=1)
    stack_test = np.stack([np.log1p(np.maximum(test_features[k], 0.0))
                           for k in test_features], axis=1)
    stack_scores = logit_stack(stack_train, np.concatenate(
        [np.ones(len(train_rows) // 2), np.zeros(len(train_rows) // 2)]), stack_test)
    report["heuristic_stack"] = {"auc": auc_of_pair_scores(
        stack_scores[:len(positive)], stack_scores[len(positive):]),
        "features": list(test_features), "fit_on": "train pairs only",
        "pairs": int(len(train_rows))}

    for name, values in test_features.items():
        report["delta_vs_teacher"][name] = bootstrap_delta(teacher, values,
                                                           len(positive))
    report["delta_vs_teacher"]["heuristic_stack"] = bootstrap_delta(
        teacher, stack_scores, len(positive))

    best_name = max(report["heuristics"], key=lambda k: report["heuristics"][k]["auc"])
    report["best_heuristic"] = best_name
    report["teacher_beats_best_heuristic"] = bool(
        report["delta_vs_teacher"][best_name]["low"] > 0.0)
    out = HERE / "teacher" / "opt" / "structure_heuristics.json"
    out.write_text(json.dumps(report, indent=1))

    lines = ["# Teacher vs 非学习结构启发式(#229 §4)", "",
             f"held-out 对: {report['test_pairs']};同一正负集合;启发式只用 TRAIN 可见图", "",
             "| 打分器 | AUC | C − 该打分器 (95% CI) |",
             "|---|---:|---|"]
    lines.append(f"| C_residual(Teacher) | {report['teacher']['auc']:.4f} | — |")
    for name, entry in list(report["heuristics"].items()) + [
            ("heuristic_stack", report["heuristic_stack"])]:
        delta = report["delta_vs_teacher"][name]
        lines.append(f"| {name} | {entry['auc']:.4f} | "
                     f"{delta['mean']:+.4f} [{delta['low']:+.4f}, {delta['high']:+.4f}] |")
    lines += ["", f"最强启发式: {best_name}; "
              + ("Teacher 的增量 CI 为正" if report["teacher_beats_best_heuristic"]
                 else "Teacher 未能超过该启发式(CI 含 0 或为负)")]
    (HERE / "REPORT_STRUCTURE_HEURISTICS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
