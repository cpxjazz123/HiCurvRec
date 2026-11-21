#!/usr/bin/env python3
"""Structure recognition with full attention: how much structure can be read, and from what.

One architecture, five inputs, identical split and budget:

* **A** semantics (768-d)      -- what the RQ-VAE encoder reads;
* **B** z (32-d)               -- before quantisation;
* **C** q (32-d)               -- after quantisation, rebuilt from the stored SID;
* **D** shuffled labels        -- same pipeline, labels permuted across instances;
* **E** real adjacency + B     -- the capability ceiling: the model is given the actual
                                  directed edges, so a failure here is the model's, not
                                  the representation's.

Widths are trained and evaluated separately, so node count alone never identifies a
family.  Labels are global: `directed_role.instance_targets` maps every family's local
numbering into one subtype vocabulary (47) and one orbit vocabulary (376), and a
held-out row is additionally required not to reuse a member of any training instance.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROLE_REPO))

INSTANCES = ROLE_REPO / "instances.npz"
CONTRACT = TEACHER / "teacher" / "opt" / "data_contract.npz"
STATES = TEACHER / "fidelity" / "fidelity_states.npz"
STATES_META = TEACHER / "fidelity" / "fidelity_states.json"

FAMILIES = ("P3", "K3", "C4", "P4", "K1_3")
WIDTHS = {"P3": 3, "K3": 3, "C4": 4, "P4": 4, "K1_3": 4}
N_FAMILIES = len(FAMILIES)
N_SUBTYPES = 47
N_ORBITS = 376
SEED = 0
EPOCHS = 150
HIDDEN = 128
HEADS = 4
LAYERS = 2
TRAIN_PER_FAMILY = 8000
HELDOUT_PER_FAMILY = 3000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ label building
def real_adjacency(nodes: np.ndarray, observed: set) -> np.ndarray:
    """(n, width, width) directed adjacency of each instance, from the real graph."""
    width = nodes.shape[1]
    out = np.zeros((len(nodes), width, width), dtype=bool)
    for row, members in enumerate(nodes.tolist()):
        for i in range(width):
            for j in range(width):
                if i != j and (members[i], members[j]) in observed:
                    out[row, i, j] = True
    return out


def build_labels() -> dict:
    """Global subtype/orbit ids, member-disjoint split, adjacency for the ceiling run."""
    from directed_role import instance_targets, load_vocabulary

    vocabulary = load_vocabulary("directed")
    payload = np.load(INSTANCES)
    contract = np.load(CONTRACT)
    # `message_edges` are the training edges; the pool's split 0 rows come from them
    observed = set(map(tuple, contract["message_edges"].tolist()))
    blocks = {}
    for family in FAMILIES:
        width = WIDTHS[family]
        split = payload[f"{family}_split"]
        targets = instance_targets(family, payload[f"{family}_subtype"],
                                   payload[f"{family}_roles"], vocabulary)
        keep = targets["subtype"] >= 0
        nodes = payload[f"{family}_nodes"][keep]
        roles = targets["orbit"][keep]
        subtypes = targets["subtype"][keep]
        part = np.where(split[keep] == 0, 0, 1).astype(np.int8)
        rng = np.random.default_rng(SEED)
        train_rows = np.flatnonzero(part == 0)
        held_rows = np.flatnonzero(part == 1)
        if len(train_rows) > TRAIN_PER_FAMILY:
            train_rows = np.sort(rng.choice(train_rows, TRAIN_PER_FAMILY, replace=False))
        blocks[family] = {"nodes": nodes, "roles": roles, "subtype": subtypes,
                          "family": np.full(len(nodes), FAMILIES.index(family), dtype=np.int64),
                          "train_rows": train_rows, "held_rows": held_rows, "width": width}
    train_items = set()
    for family in FAMILIES:
        block = blocks[family]
        train_items |= set(block["nodes"][block["train_rows"]].ravel().tolist())
    dropped = 0
    isolation = "item"
    for family in FAMILIES:
        block = blocks[family]
        held = block["held_rows"]
        pure = np.array([not (set(row.tolist()) & train_items)
                         for row in block["nodes"][held]], dtype=bool)
        dropped += int((~pure).sum())
        if pure.sum() < 500:
            isolation = "instance_only"
            pure = np.ones(len(held), dtype=bool)
        held = held[pure]
        rng = np.random.default_rng(SEED + 1)
        if len(held) > HELDOUT_PER_FAMILY:
            held = np.sort(rng.choice(held, HELDOUT_PER_FAMILY, replace=False))
        block["held_rows"] = held
    return {"blocks": blocks, "isolation": isolation,
            "dropped_for_item_overlap": int(dropped),
            "train_items": int(len(train_items)),
            "vocabulary": {"subtypes": vocabulary["n_subtypes"],
                           "orbits": vocabulary["n_orbits"]},
            "observed_edges": len(observed),
            "adjacency": {family: real_adjacency(
                blocks[family]["nodes"][np.concatenate([blocks[family]["train_rows"],
                                                        blocks[family]["held_rows"]])],
                observed) for family in FAMILIES}}


# ----------------------------------------------------------------------- model bits
def build_model(in_dim: int, extra_tokens: int):
    import torch
    import torch.nn as nn

    class FullAttention(nn.Module):
        """Every member attends to every member; no adjacency, no KNN mask."""

        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(in_dim, HIDDEN)
            self.edge = nn.Linear(1, HIDDEN) if extra_tokens else None
            self.layers = nn.ModuleList(
                nn.ModuleDict({
                    "query": nn.Linear(HIDDEN, HIDDEN),
                    "key": nn.Linear(HIDDEN, HIDDEN),
                    "value": nn.Linear(HIDDEN, HIDDEN),
                    "out": nn.Linear(HIDDEN, HIDDEN),
                    "ff": nn.Sequential(nn.Linear(HIDDEN, 2 * HIDDEN), nn.GELU(),
                                        nn.Linear(2 * HIDDEN, HIDDEN))}) for _ in range(LAYERS))
            self.family = nn.Linear(HIDDEN, N_FAMILIES)
            self.subtype = nn.Linear(HIDDEN, N_SUBTYPES)
            self.orbit = nn.Linear(HIDDEN, N_ORBITS)

        def forward(self, tokens, adjacency=None):
            h = self.input(tokens)
            if adjacency is not None and self.edge is not None:
                # the real adjacency enters as an additive bias on the attention scores
                # per neighbour pair: an edge adds a learned offset to that member's token
                h = h + self.edge(adjacency.float().unsqueeze(-1)).sum(dim=2)
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
            pooled = h.mean(dim=1)
            return {"family": self.family(pooled), "subtype": self.subtype(pooled),
                    "orbit": self.orbit(h)}

    return FullAttention()


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


def run_width(width: int, states: dict, labels: dict, table: str, with_adjacency: bool,
              shuffle: bool, device) -> dict:
    import torch

    families = [family for family in FAMILIES if WIDTHS[family] == width]
    rng = np.random.default_rng(SEED + (7 if shuffle else 0))
    del rng
    train_nodes, train_roles, train_sub, train_fam, train_adj = [], [], [], [], []
    held_nodes, held_roles, held_sub, held_fam, held_adj = [], [], [], [], []
    for family in families:
        block = labels["blocks"][family]
        adjacency = labels["adjacency"][family]
        rows = np.concatenate([block["train_rows"], block["held_rows"]])
        offset = len(block["train_rows"])
        train_nodes.append(block["nodes"][block["train_rows"]])
        train_roles.append(block["roles"][block["train_rows"]])
        train_sub.append(block["subtype"][block["train_rows"]])
        train_fam.append(block["family"][block["train_rows"]])
        train_adj.append(adjacency[:offset])
        held_nodes.append(block["nodes"][block["held_rows"]])
        held_roles.append(block["roles"][block["held_rows"]])
        held_sub.append(block["subtype"][block["held_rows"]])
        held_fam.append(block["family"][block["held_rows"]])
        held_adj.append(adjacency[offset:])
    train_nodes = np.concatenate(train_nodes)
    held_nodes = np.concatenate(held_nodes)
    train_y = {"roles": np.concatenate(train_roles), "subtype": np.concatenate(train_sub),
               "family": np.concatenate(train_fam)}
    held_y = {"roles": np.concatenate(held_roles), "subtype": np.concatenate(held_sub),
              "family": np.concatenate(held_fam)}
    train_adj = np.concatenate(train_adj)
    held_adj = np.concatenate(held_adj)
    if shuffle:
        # The shuffled control must break the association the model could actually use.
        # A permutation applied inside one width group leaves a width-separable task
        # exactly as learnable as before -- that is why the first version of this control
        # reproduced the real run to three decimals.  So the family, subtype and orbit
        # labels are permuted across the *whole* training set, which breaks any
        # token-to-label mapping and any width-to-family mapping at once.
        order = np.random.default_rng(SEED + 11).permutation(len(train_nodes))
        train_y = {key: value[order] for key, value in train_y.items()}
    x_train = torch.from_numpy(states[table][train_nodes].astype(np.float32)).to(device)
    x_held = torch.from_numpy(states[table][held_nodes].astype(np.float32)).to(device)
    a_train = torch.from_numpy(train_adj).to(device) if with_adjacency else None
    a_held = torch.from_numpy(held_adj).to(device) if with_adjacency else None

    torch.manual_seed(SEED)
    model = build_model(x_train.shape[-1], int(with_adjacency)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_family = torch.from_numpy(train_y["family"]).to(device)
    y_subtype = torch.from_numpy(train_y["subtype"]).to(device)
    y_roles = torch.from_numpy(train_y["roles"]).to(device)
    for _ in range(EPOCHS):
        model.train()
        out = model(x_train, a_train)
        loss = (torch.nn.functional.cross_entropy(out["family"], y_family)
                + torch.nn.functional.cross_entropy(out["subtype"], y_subtype)
                + torch.nn.functional.cross_entropy(
                    out["orbit"].reshape(-1, N_ORBITS), y_roles.reshape(-1)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(x_held, a_held)
    family_guess = out["family"].argmax(-1).cpu().numpy()
    subtype_guess = out["subtype"].argmax(-1).cpu().numpy()
    orbit_guess = out["orbit"].argmax(-1).cpu().numpy()
    family_truth = held_y["family"]
    subtype_truth = held_y["subtype"]
    orbit_truth = held_y["roles"]
    family_prior = float(np.bincount(train_y["family"], minlength=N_FAMILIES).max()
                         / len(train_y["family"]))
    subtype_prior_hits = 0
    for family_index in range(N_FAMILIES):
        train_mask = train_y["family"] == family_index
        held_mask = family_truth == family_index
        if not train_mask.any() or not held_mask.any():
            continue
        votes = np.bincount(train_y["subtype"][train_mask], minlength=N_SUBTYPES)
        subtype_prior_hits += int((subtype_truth[held_mask] == int(votes.argmax())).sum())
    return {"width": width, "instances_train": int(len(train_nodes)),
            "instances_held": int(len(held_nodes)),
            "family_accuracy": float((family_guess == family_truth).mean()),
            "family_macro_f1": macro_f1(family_truth, family_guess, N_FAMILIES),
            "family_majority_baseline": family_prior,
            "subtype_accuracy": float((subtype_guess == subtype_truth).mean()),
            "subtype_macro_f1": macro_f1(subtype_truth, subtype_guess, N_SUBTYPES),
            "subtype_majority_baseline": subtype_prior_hits / max(len(subtype_truth), 1),
            "orbit_accuracy": float((orbit_guess == orbit_truth).mean()),
            "orbit_macro_f1": macro_f1(orbit_truth.ravel(), orbit_guess.ravel(), N_ORBITS)}


def main() -> None:
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "full_attention.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = np.load(STATES)
    states = {key: payload[key] for key in ("semantics", "z", "q")}
    labels = build_labels()
    report = {"inputs": {"states": sha256(STATES), "instances": sha256(INSTANCES),
                         "contract": sha256(CONTRACT),
                         "checkpoint": json.loads(STATES_META.read_text())},
              "labels": {"vocabulary": labels["vocabulary"],
                         "isolation": labels["isolation"],
                         "dropped_for_item_overlap": labels["dropped_for_item_overlap"],
                         "train_items": labels["train_items"],
                         "observed_edges": labels["observed_edges"]},
              "protocol": {"seed": SEED, "epochs": EPOCHS, "heads": HEADS,
                           "layers": LAYERS, "hidden": HIDDEN,
                           "attention": "full: every member attends to every member",
                           "widths_trained_separately": True,
                           "train_per_family": TRAIN_PER_FAMILY,
                           "heldout_per_family": HELDOUT_PER_FAMILY},
              "runs": {}}
    plan = [("A_semantics", "semantics", False, False),
            ("B_z", "z", False, False),
            ("C_q", "q", False, False),
            ("D_shuffled_labels", "z", False, True),
            ("E_true_adjacency_plus_z", "z", True, False)]
    for name, table, with_adjacency, shuffle in plan:
        report["runs"][name] = {"table": table, "adjacency": with_adjacency,
                                "shuffled_labels": shuffle, "widths": {}}
        for width in (3, 4):
            row = run_width(width, states, labels, table, with_adjacency, shuffle, device)
            report["runs"][name]["widths"][str(width)] = row
            print(f"  {name} w{width}: family {row['family_accuracy']:.3f} "
                  f"F1 {row['family_macro_f1']:.3f} "
                  f"(prior {row['family_majority_baseline']:.3f}) | subtype "
                  f"{row['subtype_accuracy']:.3f} F1 {row['subtype_macro_f1']:.3f} "
                  f"(prior {row['subtype_majority_baseline']:.3f}) | orbit "
                  f"{row['orbit_accuracy']:.3f} F1 {row['orbit_macro_f1']:.3f}", flush=True)
            tmp = Path(args.out + ".tmp")
            tmp.write_text(json.dumps(report, indent=1) + "\n")
            tmp.replace(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
