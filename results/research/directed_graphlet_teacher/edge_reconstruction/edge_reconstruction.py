#!/usr/bin/env python3
"""Predict the directed edges of a candidate subgraph, then classify with the repo's own rule.

Instead of asking for one of 376 orbits, the model predicts each ordered pair of members:

    p(A->B), p(B->A), p(B->C), p(C->B), p(A->C), p(C->A)

The predicted matrix is thresholded and handed to `directed_graphlets.classify`, the same
deterministic classifier that produced the labels, so the graphlet family and every
member's orbit are derived from the reconstruction and cannot disagree with it.

Labels come from the real training graph; the model never sees the true adjacency.
Three inputs share one architecture, split and budget: 768-d semantics, `z` and `q`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import classify, load_adjacency  # noqa: E402

CONTRACT = TEACHER / "teacher" / "opt" / "data_contract.npz"
INSTANCES = REPO / "results" / "research" / "directed_graphlet_role" / "instances.npz"
STATES = TEACHER / "fidelity" / "fidelity_states.npz"
STATES_META = TEACHER / "fidelity" / "fidelity_states.json"

FAMILIES = ("P3", "K3", "C4", "P4", "K1_3")
WIDTHS = {"P3": 3, "K3": 3, "C4": 4, "P4": 4, "K1_3": 4}
SECTION = ("P3", "K3")                 # width 3, both skeletons of that width
SEED = 0
EPOCHS = 60
HIDDEN = 128
HEADS = 4
LAYERS = 2
TRAIN_ROWS = 60000
HELD_ROWS = 6000
THRESHOLDS = (0.3, 0.5, 0.7)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows() -> dict:
    """Width-3 instances of the training graph, with every ordered pair's truth."""
    payload = np.load(INSTANCES)
    contract = np.load(CONTRACT)
    observed = set(map(tuple, contract["message_edges"].tolist()))
    nodes, family, part, adjacency = [], [], [], []
    for name in SECTION:
        block = payload[f"{name}_nodes"]
        split = payload[f"{name}_split"]
        rows = np.concatenate([np.flatnonzero(split == 0), np.flatnonzero(split == 1)])
        pieces = np.concatenate([np.zeros(int((split == 0).sum()), dtype=np.int8),
                                 np.ones(int((split == 1).sum()), dtype=np.int8)])
        nodes.append(block[rows])
        family.append(np.full(len(rows), FAMILIES.index(name), dtype=np.int64))
        part.append(pieces)
        edges = np.zeros((len(rows), 3, 3), dtype=bool)
        for index, members in enumerate(block[rows].tolist()):
            for i in range(3):
                for j in range(3):
                    if i != j and (members[i], members[j]) in observed:
                        edges[index, i, j] = True
        adjacency.append(edges)
    nodes = np.concatenate(nodes)
    family = np.concatenate(family)
    part = np.concatenate(part)
    adjacency = np.concatenate(adjacency)
    # held-out rows must not share a member with any training row
    train_items = set(nodes[part == 0].ravel().tolist())
    pure = np.array([not (set(row.tolist()) & train_items) for row in nodes])
    train_rows = np.flatnonzero(part == 0)
    held_rows = np.flatnonzero((part == 1) & pure)
    isolation = "item"
    if len(held_rows) < 500:
        held_rows = np.flatnonzero(part == 1)
        isolation = "instance_only"
    rng = np.random.default_rng(SEED)
    if len(train_rows) > TRAIN_ROWS:
        train_rows = np.sort(rng.choice(train_rows, TRAIN_ROWS, replace=False))
    if len(held_rows) > HELD_ROWS:
        held_rows = np.sort(rng.choice(held_rows, HELD_ROWS, replace=False))
    return {"nodes": nodes, "family": family, "adjacency": adjacency,
            "train_rows": train_rows, "held_rows": held_rows,
            "isolation": isolation,
            "dropped_for_item_overlap": int(((part == 1) & ~pure).sum()),
            "train_items": len(train_items)}


def build_model(in_dim: int):
    import torch
    import torch.nn as nn

    class EdgeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(in_dim, HIDDEN)
            self.layers = nn.ModuleList(
                nn.ModuleDict({
                    "query": nn.Linear(HIDDEN, HIDDEN),
                    "key": nn.Linear(HIDDEN, HIDDEN),
                    "value": nn.Linear(HIDDEN, HIDDEN),
                    "out": nn.Linear(HIDDEN, HIDDEN),
                    "ff": nn.Sequential(nn.Linear(HIDDEN, 2 * HIDDEN), nn.GELU(),
                                        nn.Linear(2 * HIDDEN, HIDDEN))}) for _ in range(LAYERS))
            # every ordered pair is scored from its two endpoints: (u, v, u*v)
            self.pair = nn.Sequential(nn.Linear(3 * HIDDEN, HIDDEN), nn.GELU(),
                                      nn.Linear(HIDDEN, 1))

        def forward(self, tokens):
            h = self.input(tokens)
            for layer in self.layers:
                q = layer["query"](h).view(len(h), -1, HEADS, HIDDEN // HEADS)
                k = layer["key"](h).view(len(h), -1, HEADS, HIDDEN // HEADS)
                v = layer["value"](h).view(len(h), -1, HEADS, HIDDEN // HEADS)
                scores = torch.einsum("nqhd,nkhd->nhqk", q, k) / (HIDDEN // HEADS) ** 0.5
                weights = torch.softmax(scores, dim=-1)
                attended = torch.einsum("nhqk,nkhd->nqhd", weights, v).reshape(
                    len(h), -1, HIDDEN)
                h = h + layer["out"](attended)
                h = h + layer["ff"](h)
            left = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            right = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            logits = self.pair(torch.cat([left, right, left * right], dim=-1)).squeeze(-1)
            return logits

    return EdgeNet()


def macro_f1(truth, guess, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        tp = int(((truth == label) & (guess == label)).sum())
        fp = int(((truth != label) & (guess == label)).sum())
        fn = int(((truth == label) & (guess != label)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall
                      / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(probabilities: np.ndarray, adjacency: np.ndarray, family: np.ndarray,
             threshold: float, family_bands: dict) -> dict:
    """Threshold the predicted edges and let the deterministic classifier decide."""
    predicted = probabilities > threshold
    # the candidate matrix is what a degree-only control gives as well: any pair of busy
    # items is predicted connected.  Diagonal entries are never counted as edges.
    predicted = np.broadcast_to(predicted[:, :, None], adjacency.shape).copy() \
        if predicted.ndim == 2 else predicted
    off_diagonal = ~np.eye(adjacency.shape[1], dtype=bool)[None]
    truth_edges = adjacency & off_diagonal
    guess_edges = predicted & off_diagonal
    tp = int((truth_edges & guess_edges).sum())
    fp = int((~truth_edges & guess_edges).sum())
    fn = int((truth_edges & ~guess_edges).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    edge_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = float((guess_edges == truth_edges).all(axis=(1, 2)).mean())
    filled = float(guess_edges.sum() / truth_edges.sum()) if truth_edges.sum() else float("nan")

    predicted_family, predicted_orbit, truth_orbit, truth_family = [], [], [], []
    for row in range(len(predicted)):
        result = classify(guess_edges[row].copy())
        predicted_family.append(FAMILIES.index(result[0]) if result else -1)
        predicted_orbit.append(result[2] if result else (None,) * adjacency.shape[1])
        true_result = classify(truth_edges[row].copy())
        truth_family.append(FAMILIES.index(true_result[0]) if true_result else -1)
        truth_orbit.append(true_result[2] if true_result else (None,) * adjacency.shape[1])
    predicted_family = np.array(predicted_family)
    truth_family = np.array(truth_family)
    # orbit ids are local per subtype; compare only where both sides produced one
    orbit_hits, orbit_total = 0, 0
    for guess_row, truth_row in zip(predicted_orbit, truth_orbit):
        if guess_row[0] is None or truth_row[0] is None:
            continue
        for guess, truth in zip(guess_row, truth_row):
            orbit_hits += int(guess == truth)
            orbit_total += 1
    return {"threshold": threshold,
            "edge_precision": precision, "edge_recall": recall, "edge_f1": edge_f1,
            "adjacency_exact": exact, "predicted_edge_share": filled,
            "classifiable_share": float((predicted_family >= 0).mean()),
            "family_accuracy": float((predicted_family == truth_family).mean()),
            "family_macro_f1": macro_f1(truth_family, predicted_family, len(FAMILIES)),
            "orbit_accuracy": orbit_hits / max(orbit_total, 1),
            "orbit_positions_compared": orbit_total}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "edge_reconstruction.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = build_rows()
    payload = np.load(STATES)
    states = {key: payload[key] for key in ("semantics", "z", "q")}
    report = {"inputs": {"instances": sha256(INSTANCES), "states": sha256(STATES),
                         "contract": sha256(CONTRACT),
                         "classifier": "directed_graphlets.classify on the predicted matrix"},
              "labels": {"section": list(SECTION), "isolation": rows["isolation"],
                         "dropped_for_item_overlap": rows["dropped_for_item_overlap"],
                         "train_rows": int(len(rows["train_rows"])),
                         "held_rows": int(len(rows["held_rows"])),
                         "train_items": rows["train_items"]},
              "protocol": {"seed": SEED, "epochs": EPOCHS, "heads": HEADS, "layers": LAYERS,
                           "hidden": HIDDEN, "thresholds": list(THRESHOLDS),
                           "prediction": "one logit per ordered pair of members",
                           "supervision": "real training-graph edges, never given as input",
                           "threshold_rule": "chosen on the training split by edge F1; the "
                                             "held-out rows are only reported",
                           "baseline": "degree-only: predict a connection whenever both "
                                       "members are above the same degree threshold"},
              "runs": {},
              "controls": {}}

    # behaviour degree: how many directed message edges touch an item.  A subgraph of
    # busy items looks "well connected" whatever its shape, so the score is normalised
    # by this quantity and a degree-only baseline is reported next to every run.
    degree = np.bincount(np.load(CONTRACT)["message_edges"].ravel(),
                         minlength=len(states["z"])).astype(np.float64)
    degree = degree / max(degree.max(), 1.0)
    member_degree = degree[rows["nodes"]]
    x_train_idx = rows["train_rows"]
    x_held_idx = rows["held_rows"]
    held_degree = member_degree[x_held_idx]
    degree_baseline = {
        f"threshold_{threshold}": evaluate(
            held_degree.astype(np.float64), rows["adjacency"][x_held_idx],
            rows["family"][x_held_idx], threshold, {})
        for threshold in THRESHOLDS}
    train_degree = member_degree[x_train_idx].astype(np.float64)
    train_scores = {threshold: evaluate(train_degree, rows["adjacency"][x_train_idx],
                                        rows["family"][x_train_idx], threshold, {})
                    for threshold in THRESHOLDS}
    best_degree = max(train_scores, key=lambda value: train_scores[value]["edge_f1"])
    degree_baseline["threshold_chosen_on_train"] = {
        "threshold": best_degree, "train_edge_f1": train_scores[best_degree]["edge_f1"],
        "heldout": degree_baseline[f"threshold_{best_degree}"]}
    report["controls"]["degree_only"] = degree_baseline
    for name, value in degree_baseline.items():
        if "edge_f1" not in value:
            continue
        print(f"  degree-only {name}: edgeF1 {value['edge_f1']:.3f} "
              f"exact {value['adjacency_exact']:.3f}", flush=True)
    print(f"  degree-only chosen threshold {best_degree}: held-out edgeF1 "
          f"{degree_baseline[f'threshold_{best_degree}']['edge_f1']:.3f}", flush=True)
    y_train = torch.from_numpy(rows["adjacency"][x_train_idx]).float().to(device)
    degree_scaled = float(np.linalg.norm(states["z"], axis=1).mean())
    for name in ("semantics", "z", "q"):
        torch.manual_seed(SEED)
        model = build_model(states[name].shape[1] + 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        def features(index: np.ndarray) -> np.ndarray:
            base = states[name][rows["nodes"][index]].astype(np.float32)
            extra = member_degree[index].astype(np.float32)[..., None]
            return np.concatenate([base, degree_scaled * extra], axis=-1)

        tokens = torch.from_numpy(features(x_train_idx)).to(device)
        weight = torch.tensor(1.0).to(device)
        for _ in range(EPOCHS):
            model.train()
            logits = model(tokens)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_train, pos_weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            held_tokens = torch.from_numpy(features(x_held_idx)).to(device)
            probabilities = torch.sigmoid(model(held_tokens)).cpu().numpy()
        with torch.no_grad():
            train_probabilities = torch.sigmoid(model(tokens)).cpu().numpy()
        # the threshold is picked on the training split; the held-out rows only report
        scores = {threshold: evaluate(train_probabilities, rows["adjacency"][x_train_idx],
                                      rows["family"][x_train_idx], threshold, {})
                  for threshold in THRESHOLDS}
        best = max(scores, key=lambda value: scores[value]["edge_f1"])
        report["runs"][name] = {
            f"threshold_{threshold}": evaluate(
                probabilities, rows["adjacency"][x_held_idx], rows["family"][x_held_idx],
                threshold, {})
            for threshold in THRESHOLDS}
        report["runs"][name]["threshold_chosen_on_train"] = {
            "threshold": best, "train_edge_f1": scores[best]["edge_f1"],
            "heldout": report["runs"][name][f"threshold_{best}"]}
        chosen = report["runs"][name]["threshold_chosen_on_train"]
        print(f"  {name}: threshold {chosen['threshold']} chosen on train "
              f"(edgeF1 {chosen['train_edge_f1']:.3f}) -> held-out edgeF1 "
              f"{chosen['heldout']['edge_f1']:.3f}, exact "
              f"{chosen['heldout']['adjacency_exact']:.3f}, family "
              f"{chosen['heldout']['family_accuracy']:.3f}, orbit "
              f"{chosen['heldout']['orbit_accuracy']:.3f}", flush=True)
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps(report, indent=1) + "\n")
        tmp.replace(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
