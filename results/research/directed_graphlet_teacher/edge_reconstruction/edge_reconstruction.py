#!/usr/bin/env python3
"""Edge reconstruction against the **complete** behaviour graph, with the classifier in the loop.

The previous version trained against `message_edges`, which is the behaviour graph with
the held-out pairs removed: an instance's label could itself be part of the test split, and
a P3 whose middle edge is held out is an empty graph.  That makes the training target
inconsistent with the question being asked.

Here the target is the complete directed graph
(`behavior_graph_directed/graph.npz`, 323633 edges), so every instance's truth classifies
back to the family the instance pool named it by -- the script asserts this before
training and stops otherwise.

Also fixed here: the diagonal is excluded from the loss, the degree control now uses both
endpoints, every reported number is computed over **all** held-out instances (an
unclassifiable prediction counts as wrong, never as excluded), orbit ids are the global
ones from `directed_role.instance_targets`, and both widths are covered.
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
ROLE_REPO = REPO / "results" / "research" / "directed_graphlet_role"
sys.path.insert(0, str(ROLE_REPO))
sys.path.insert(0, str(HERE.parent / "full_attention"))

from directed_graphlets import classify  # noqa: E402
from directed_role import FAMILIES, instance_targets, load_vocabulary  # noqa: E402

VOCABULARY = load_vocabulary("directed")


def global_orbit(local_ids, family: str, key: str):
    """`classify` returns per-subtype local orbit ids; compare only global ones.

    The label side went through `instance_targets`, which maps (family, subtype key,
    local orbit) into the single global vocabulary.  Reading `classify`'s local ids
    against those global ids compares two different numberings and every hit is chance.
    """
    subtype_id = VOCABULARY["subtype_of"].get(f"{family}:{key}")
    if subtype_id is None:
        return None
    mapping = VOCABULARY["orbits_of"][subtype_id]
    return [int(mapping[int(value)]) for value in local_ids]
from full_attention import build_labels  # noqa: E402

FULL_GRAPH = REPO / "results" / "research" / "behavior_graph_directed" / "graph.npz"
INSTANCES = ROLE_REPO / "instances.npz"
STATES = TEACHER / "fidelity" / "fidelity_states.npz"
STATES_META = TEACHER / "fidelity" / "fidelity_states.json"
WIDTHS = {"P3": 3, "K3": 3, "C4": 4, "P4": 4, "K1_3": 4}
N_FAMILIES = len(FAMILIES)
SEED = 0
EPOCHS = 40      # legacy constant, unused: training is counted in steps now
HIDDEN = 128
HEADS = 4
LAYERS = 2
TRAIN_ROWS = 200000
BATCH = 512
STEPS = 6000
HELD_ROWS = 6000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_adjacency(nodes: np.ndarray, edges: set) -> np.ndarray:
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, width), dtype=bool)
    for index, members in enumerate(nodes.tolist()):
        for i in range(width):
            for j in range(width):
                if i != j and (members[i], members[j]) in edges:
                    out[index, i, j] = True
    return out


def build_rows(width: int) -> dict:
    """Instances with the complete-graph truth, the classifier's own labels, and a split."""
    payload = np.load(INSTANCES)
    graph = np.load(FULL_GRAPH)
    edges = set(zip(graph["rows"].tolist(), graph["cols"].tolist()))
    vocabulary = load_vocabulary("directed")
    nodes, family_local, family_global, orbit_global, part = [], [], [], [], []
    members = [name for name in FAMILIES if WIDTHS[name] == width]
    for name in members:
        block = payload[f"{name}_nodes"]
        split = payload[f"{name}_split"]
        targets = instance_targets(name, payload[f"{name}_subtype"],
                                   payload[f"{name}_roles"], vocabulary)
        keep = targets["subtype"] >= 0
        rows = np.concatenate([np.flatnonzero((split == 0) & keep),
                               np.flatnonzero((split == 1) & keep)])
        mask = np.concatenate([np.zeros(int(((split == 0) & keep).sum()), dtype=np.int8),
                               np.ones(int(((split == 1) & keep).sum()), dtype=np.int8)])
        nodes.append(block[rows])
        family_local.append(np.full(len(rows), FAMILIES.index(name), dtype=np.int64))
        orbit_global.append(targets["orbit"][rows])
        part.append(mask)
    nodes = np.concatenate(nodes)
    part = np.concatenate(part)
    adjacency = full_adjacency(nodes, edges)

    # the classifier must reproduce each instance's own family before anything is trained
    mismatched = 0
    for row in range(len(nodes)):
        result = classify(adjacency[row].copy())
        if result is None or result[0] != FAMILIES[int(np.concatenate(family_local)[row])]:
            mismatched += 1
    if mismatched:
        raise SystemExit(f"{mismatched} instances do not classify back to their family; "
                         "the label source is still inconsistent with the question")

    train_rows = np.flatnonzero(part == 0)
    held_rows = np.flatnonzero(part == 1)
    train_items = set(nodes[train_rows].ravel().tolist())
    pure = np.array([not (set(row.tolist()) & train_items) for row in nodes])
    dropped = int(((part == 1) & ~pure).sum())
    isolation = "item"
    if ((part == 1) & pure).sum() < 500:
        isolation = "instance_only"
        pure = np.ones(len(nodes), dtype=bool)
    held_rows = np.flatnonzero((part == 1) & pure)
    rng = np.random.default_rng(SEED)
    if len(train_rows) > TRAIN_ROWS:
        train_rows = np.sort(rng.choice(train_rows, TRAIN_ROWS, replace=False))
    if len(held_rows) > HELD_ROWS:
        held_rows = np.sort(rng.choice(held_rows, HELD_ROWS, replace=False))
    return {"nodes": nodes, "part": part, "adjacency": adjacency,
            "family": np.concatenate(family_local), "orbit": np.concatenate(orbit_global),
            "width": width,
            "train_rows": train_rows, "held_rows": held_rows, "isolation": isolation,
            "dropped_for_item_overlap": dropped, "train_items": len(train_items),
            "label_header_mismatch": mismatched}


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
            return self.pair(torch.cat([left, right, left * right], dim=-1)).squeeze(-1)

    return EdgeNet()


def orbit_pairs(width: int):
    return [(i, j) for i in range(width) for j in range(width) if i != j]


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
             orbit_global: np.ndarray, threshold: float) -> dict:
    """Every held-out instance is scored; an unclassifiable prediction counts as wrong."""
    predicted = probabilities > threshold
    width = adjacency.shape[1]
    off_diagonal = ~np.eye(width, dtype=bool)[None]
    truth_edges = adjacency & off_diagonal
    guess_edges = predicted & off_diagonal
    tp = int((truth_edges & guess_edges).sum())
    fp = int((~truth_edges & guess_edges).sum())
    fn = int((truth_edges & ~guess_edges).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    edge_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = float((guess_edges == truth_edges).all(axis=(1, 2)).mean())

    predicted_family, predicted_orbit = [], []
    for row in range(len(predicted)):
        result = classify(guess_edges[row].copy())
        predicted_family.append(FAMILIES.index(result[0]) if result else -1)
        predicted_orbit.append(global_orbit(result[2], result[0], result[1])
                               if result else None)
    predicted_family = np.array(predicted_family)
    family_correct = predicted_family == family
    # family score over the full instance, unclassifiable = wrong
    family_overall = float(family_correct.mean())
    # orbit score over the full instance: pair the predicted matrix's own orbit ids with
    # the global orbit ids of the same subtype; unmatched/None counts as wrong
    orbit_hits, orbit_total = 0, 0
    for row, ids in enumerate(predicted_orbit):
        orbit_total += width
        if ids is None:
            continue
        truth_ids = orbit_global[row]
        for position in range(width):
            orbit_hits += int(int(ids[position]) == int(truth_ids[position]))
    return {"threshold": threshold,
            "edge_precision": precision, "edge_recall": recall, "edge_f1": edge_f1,
            "adjacency_exact": exact,
            "classifiable_share": float((predicted_family >= 0).mean()),
            "family_accuracy_overall": family_overall,
            "family_macro_f1": macro_f1(family, predicted_family, N_FAMILIES),
            "orbit_accuracy_overall": orbit_hits / max(orbit_total, 1),
            "instances": int(len(predicted))}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "edge_reconstruction.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {key: np.load(STATES)[key] for key in ("semantics", "z", "q")}
    report = {"inputs": {"instances": sha256(INSTANCES), "states": sha256(STATES),
                         "full_graph": sha256(FULL_GRAPH),
                         "checkpoint": json.loads(STATES_META.read_text())},
              "labels": {"source": "behavior_graph_directed/graph.npz (complete graph)",
                         "widths": {}},
              "protocol": {"seed": SEED, "epochs": EPOCHS, "heads": HEADS, "layers": LAYERS,
                           "hidden": HIDDEN, "widths": sorted(set(WIDTHS.values())),
                           "loss": "BCE over ordered pairs, diagonal excluded",
                           "inputs": "item representations only, no degree feature",
                           "optimisation": f"{STEPS} shuffled mini-batches of {BATCH} "
                                           f"(legacy EPOCHS={EPOCHS} is unused)",
                           "threshold": "chosen on the training split by edge F1",
                           "scoring": "every held-out instance is scored; an unclassifiable "
                                      "prediction counts as wrong",
                           "orbit_ids": "global, via directed_role.instance_targets"},
              "runs": {}, "controls": {}}

    graph = np.load(FULL_GRAPH)
    degree_out = np.bincount(graph["rows"], minlength=len(states["z"])).astype(np.float64)
    degree_in = np.bincount(graph["cols"], minlength=len(states["z"])).astype(np.float64)
    # min(out, in) degree already separates hubs from cold items; no max-normalisation,
    # because a score compressed into a tiny range makes any fixed threshold empty
    both = np.minimum(degree_out, degree_in)

    for width in (3, 4):
        rows = build_rows(width)
        report["labels"]["widths"][str(width)] = {
            "classifier_reproduces_family": rows["label_header_mismatch"] == 0,
            "isolation": rows["isolation"],
            "dropped_for_item_overlap": rows["dropped_for_item_overlap"],
            "train_rows": int(len(rows["train_rows"])),
            "held_rows": int(len(rows["held_rows"]))}
        train_rows, held_rows = rows["train_rows"], rows["held_rows"]
        y_truth = rows["adjacency"][train_rows]
        keep_mask = np.broadcast_to(~np.eye(width, dtype=bool)[None],
                                    (len(train_rows), width, width)).copy()
        report["runs"][str(width)] = {}
        for name in ("semantics", "z", "q"):
            torch.manual_seed(SEED)
            model = build_model(states[name].shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            tokens = torch.from_numpy(
                states[name][rows["nodes"][train_rows]].astype(np.float32)).to(device)
            target = torch.from_numpy(y_truth.astype(np.float32)).to(device)
            keep = torch.from_numpy(keep_mask).to(device)
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            # one full-batch update per "epoch" is 60 updates in total; a shuffled
            # mini-batch loop with a fixed step budget is what the number of epochs stood
            # for.  The step count is reported so the optimisation is comparable.
            for _ in range(STEPS):
                model.train()
                pick = torch.randint(0, len(tokens), (BATCH,), generator=generator)
                logits = model(tokens[pick])
                off_diagonal = torch.ones((BATCH, width, width), dtype=torch.bool,
                                          device=device)
                for position in range(width):
                    off_diagonal[:, position, position] = False
                # the diagonal is never a prediction target
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[off_diagonal], target[pick][off_diagonal])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                held_tokens = torch.from_numpy(
                    states[name][rows["nodes"][held_rows]].astype(np.float32)).to(device)
                probabilities = torch.sigmoid(model(held_tokens)).cpu().numpy()
                train_probabilities = torch.sigmoid(model(tokens)).cpu().numpy()
            train_scores = {t: evaluate(train_probabilities, y_truth,
                                        rows["family"][train_rows],
                                        rows["orbit"][train_rows], t)
                            for t in (0.3, 0.5, 0.7)}
            best = max(train_scores, key=lambda value: train_scores[value]["edge_f1"])
            report["runs"][str(width)][name] = {
                "threshold_chosen_on_train": best,
                "train_edge_f1": train_scores[best]["edge_f1"],
                "heldout": evaluate(probabilities, rows["adjacency"][held_rows],
                                    rows["family"][held_rows], rows["orbit"][held_rows], best)}
            row = report["runs"][str(width)][name]["heldout"]
            print(f"  w{width} {name} thr {best}: edgeF1 {row['edge_f1']:.3f} "
                  f"exact {row['adjacency_exact']:.3f} "
                  f"classifiable {row['classifiable_share']:.3f} "
                  f"family {row['family_accuracy_overall']:.3f} "
                  f"orbit {row['orbit_accuracy_overall']:.3f}", flush=True)

        train_members = both[rows["nodes"][train_rows]]
        held_members = both[rows["nodes"][held_rows]]
        pair_train = train_members[:, :, None] * train_members[:, None, :]
        pair_held = held_members[:, :, None] * held_members[:, None, :]
        # The degree score is an unbounded product, so a fixed threshold is meaningless
        # and the two degenerate ends collapse it: "nothing connected" scores 0 on every
        # candidate and "everything connected" scores the share of true edges.  Both are
        # reported next to the quantiles.
        quantiles = [float(np.quantile(pair_train, q)) for q in (0.5, 0.75, 0.9)]
        candidates = quantiles + [float("inf")]
        train_scores = {t: evaluate(pair_train, y_truth, rows["family"][train_rows],
                                    rows["orbit"][train_rows], t) for t in candidates}
        train_scores["none_connected"] = evaluate(
            np.zeros_like(pair_train), y_truth, rows["family"][train_rows],
            rows["orbit"][train_rows], 0.5)
        best = max(quantiles, key=lambda value: train_scores[value]["edge_f1"])
        train_edge_share = float(y_truth.sum() / (len(y_truth) * width * (width - 1)))
        report["controls"].setdefault("both_endpoint_degree", {})[str(width)] = {
            "threshold_scale": "min(out_degree, in_degree) of the two endpoints, product",
            "train_quantiles_tried": quantiles,
            "train_f1_by_threshold": {
                ("inf" if t == float("inf") else round(t, 3)): round(v["edge_f1"], 4)
                for t, v in train_scores.items() if not isinstance(t, str)},
            "threshold_chosen_on_train": best,
            "heldout": evaluate(pair_held, rows["adjacency"][held_rows],
                                rows["family"][held_rows], rows["orbit"][held_rows], best),
            "everything_connected_baseline": evaluate(
                np.ones_like(pair_held), rows["adjacency"][held_rows],
                rows["family"][held_rows], rows["orbit"][held_rows], -1.0),
            "nothing_connected_baseline": evaluate(
                np.zeros_like(pair_held), rows["adjacency"][held_rows],
                rows["family"][held_rows], rows["orbit"][held_rows], 0.5),
            "true_edge_share": train_edge_share}
        control = report["controls"]["both_endpoint_degree"][str(width)]
        print(f"  w{width} degree: quantile thr {best} -> edgeF1 "
              f"{control['heldout']['edge_f1']:.3f}; everything-connected "
              f"{control['everything_connected_baseline']['edge_f1']:.3f}; "
              f"nothing-connected {control['nothing_connected_baseline']['edge_f1']:.3f}; "
              f"true edge share {train_edge_share:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
