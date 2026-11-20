"""Re-evaluate the saved graph Teacher with the direction metric fixed.

The first pass asked whether a held-out pair's reverse edge was in the *observed*
graph; held-out pairs were removed in both directions, so no pair qualified and the
accuracy came out NaN with n=0.  The direction label has to come from the original
graph: among held-out pairs that are one-way in the original graph, does the head
pick the direction that exists?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import load_adjacency  # noqa: E402
from graph_teacher import GraphTeacher, build_splits, degree_matched_negative  # noqa: E402
from relation_audit import auc  # noqa: E402

SEED = 226


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(HERE / "graph_teacher_v1.pth", map_location="cpu", weights_only=False)
    model = GraphTeacher(**{"in_dim": 768, "hidden": checkpoint["config"]["hidden"],
                            "out_dim": checkpoint["config"]["output"]}).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    original = set((int(a), int(b)) for a, b in zip(rows.tolist(), cols.tolist()))
    observed_directed, held_pairs, valid_pairs = build_splits(rows, cols, n_items)
    index = np.array(observed_directed, dtype=np.int64)
    adjacency = torch.sparse_coo_tensor(
        torch.from_numpy(index.T).to(device),
        torch.ones(len(index), dtype=torch.float32, device=device),
        size=(n_items, n_items)).coalesce()
    features = torch.from_numpy(
        np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(np.float32)
    ).to(device)
    observed_set = set((int(a), int(b)) for a, b in index.tolist())
    degree = np.bincount(index[:, 0], minlength=n_items) + np.bincount(index[:, 1], minlength=n_items)
    rng = np.random.default_rng(SEED)

    with torch.no_grad():
        state = model.encode(features, adjacency)
        held = torch.from_numpy(held_pairs).to(device)
        forward_labels = np.array([
            (int(a), int(b)) in original and (int(b), int(a)) not in original
            for a, b in held_pairs.tolist()])
        reverse_labels = np.array([
            (int(b), int(a)) in original and (int(a), int(b)) not in original
            for a, b in held_pairs.tolist()])
        one_way = forward_labels | reverse_labels
        logits = model.direction_head(model.pair_features(state, held[:, 0], held[:, 1]))
        predictions = logits.argmax(-1).cpu().numpy()
        direction_accuracy = float(np.mean(predictions[one_way]
                                           == reverse_labels[one_way].astype(np.int64)))

        positives = model.edge_head(model.pair_features(state, held[:, 0], held[:, 1])).squeeze(-1)
        negatives = np.array([degree_matched_negative(degree, t, rng, observed_set, n_items)
                              for t in held_pairs[:, 0]])
        negative_scores = model.edge_head(model.pair_features(
            state, held[:, 0], torch.from_numpy(negatives).to(device))).squeeze(-1)
        labels = np.concatenate([np.ones(len(held_pairs)), np.zeros(len(held_pairs))])
        teacher_auc = auc(np.concatenate([positives.cpu().numpy(), negative_scores.cpu().numpy()]),
                          labels)
        normalised = F.normalize(features, dim=-1)
        semantic = np.concatenate([(normalised[held[:, 0]] * normalised[held[:, 1]]).sum(-1).cpu().numpy(),
                                   (normalised[held[:, 0]] * normalised[
                                       torch.from_numpy(negatives).to(device)]).sum(-1).cpu().numpy()])
        semantic_auc = auc(semantic, labels)
        degree_scores = np.concatenate([degree[held_pairs[:, 0]], degree[negatives]])
        degree_auc = auc(degree_scores, labels)

    report = json.loads((HERE / "graph_teacher_report.json").read_text())
    report["direction"] = {"accuracy_on_one_way_held_pairs": direction_accuracy,
                           "n": int(one_way.sum()), "forward": int(forward_labels.sum()),
                           "reverse": int(reverse_labels.sum())}
    report["masked_edges"] = {"teacher_auc": float(teacher_auc), "semantic_auc": float(semantic_auc),
                              "degree_auc": float(degree_auc),
                              "positive_rate": float(labels.mean()),
                              "held_pairs": int(len(held_pairs))}
    report["evaluation_fix"] = "direction label from the original graph, not the observed graph"
    (HERE / "graph_teacher_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({key: report[key] for key in ("masked_edges", "direction", "orbit_profile")},
                     indent=1))


if __name__ == "__main__":
    main()
