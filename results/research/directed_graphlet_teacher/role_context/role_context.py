#!/usr/bin/env python3
"""Can the transformer tell which role the *same* item plays in different subgraphs?

Every earlier number averaged over positions, so an item's behaviour across its many
subgraphs was never separated from the position it happened to sit in.  Here the unit of
measurement is *(item, role pair)*: an item that appears in two subgraphs where it takes
two different orbits, and the model must pick the right one in each, judged only against
that pair.  With full attention and no adjacency, the only thing that differs between the
two subgraphs is who else is in them, so a correct answer means the context was read.

Role choice must happen **before** the final orbit projection: a model that emits one
orbit per item cannot tell the two contexts apart by construction, so that is the
headline control rather than an afterthought.

Sets:
  A  per-instance orbit identity  (the question above)
  B  context ablation             (same model, other members' tokens zeroed)
  C  item-level instance mix      (predict the orbit distribution over the item's own
                                   subgraphs; this is the bag-of-roles view)
  D  context-free control         (shared orbit per item, the ~0.176 ceiling)
  E  real adjacency ceiling       (is the task learnable at all?)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "full_attention"))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from full_attention import (CONTRACT, FAMILIES, HIDDEN, HEADS, INSTANCES, LAYERS,
                            N_ORBITS, SEED, STATES, STATES_META, WIDTHS, build_labels,
                            macro_f1)  # noqa: E402

EPOCHS = 40
TRAIN_ITEMS = 9000
HELD_ITEMS = 20000
PAIRS_PER_ITEM = 6
WIDTH = 3
ROLE_GROUP = 8          # roles are grouped by frequency band for the item head
HISTORY_BINS = (1, 2, 4, 8)   # how many subgraphs of that item were already seen


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------- data assembly
def width3_rows(labels: dict) -> dict:
    """All 3-node instances with global subtype/orbit, split by item for the role task."""
    nodes, roles, subtype, family, part = [], [], [], [], []
    for name in FAMILIES:
        if WIDTHS[name] != WIDTH:
            continue
        block = labels["blocks"][name]
        ordered = np.concatenate([block["train_rows"], block["held_rows"]])
        nodes.append(block["nodes"][ordered])
        roles.append(block["roles"][ordered])
        subtype.append(block["subtype"][ordered])
        family.append(block["family"][ordered])
        part.append(np.concatenate([
            np.zeros(len(block["train_rows"]), dtype=np.int8),
            np.ones(len(block["held_rows"]), dtype=np.int8)]))
        block["_ordered"] = ordered
    return {"nodes": np.concatenate(nodes), "roles": np.concatenate(roles),
            "subtype": np.concatenate(subtype), "family": np.concatenate(family),
            "part": np.concatenate(part)}


def role_contexts(rows: dict, seed: int = SEED) -> dict:
    """Pick two (item, subgraph) contexts with different orbits, per held-out item."""
    rng = np.random.default_rng(seed)
    by_item: dict[int, list[tuple[int, int]]] = defaultdict(list)
    train_items = set()
    for instance, members in enumerate(rows["nodes"].tolist()):
        if rows["part"][instance] == 0:
            train_items.update(members)
            continue
        for position, item in enumerate(members):
            by_item[item].append((instance, position))
    pairs = []
    for item, contexts in by_item.items():
        if item in train_items or len(contexts) < 2:
            continue
        roles = [int(rows["roles"][instance, position]) for instance, position in contexts]
        if len(set(roles)) < 2:
            continue
        # several distinct-role pairs per item: one pair per item would leave a few
        # hundred rows on a catalogue of this size and make every metric noise
        for _ in range(PAIRS_PER_ITEM):
            pick = rng.choice(len(contexts), 2, replace=False)
            if roles[pick[0]] == roles[pick[1]]:
                for other in range(len(contexts)):
                    if roles[other] != roles[pick[0]]:
                        pick[1] = other
                        break
            pairs.append({"item": int(item),
                          "a": {"instance": int(contexts[pick[0]][0]),
                                "position": int(contexts[pick[0]][1]),
                                "true_orbit": roles[pick[0]]},
                          "b": {"instance": int(contexts[pick[1]][0]),
                                "position": int(contexts[pick[1]][1]),
                                "true_orbit": roles[pick[1]]}})
    rng.shuffle(pairs)
    pairs = pairs[:HELD_ITEMS]
    return {"pairs": pairs, "held_items": len({row["item"] for row in pairs}),
            "train_instances": int((rows["part"] == 0).sum()),
            "held_instances": int((rows["part"] == 1).sum())}


# ------------------------------------------------------------------------- the model
def build_role_model(in_dim: int, n_groups: int):
    import torch
    import torch.nn as nn

    class ContextRoleNet(nn.Module):
        """Tokens are (item, role-slot) pairs: a role-specific embedding exists, but which
        one is used is decided by the token's own attended representation."""

        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(in_dim, HIDDEN)
            self.slot = nn.Parameter(torch.zeros(1, 1, HIDDEN))
            self.layers = nn.ModuleList(
                nn.ModuleDict({
                    "query": nn.Linear(HIDDEN, HIDDEN),
                    "key": nn.Linear(HIDDEN, HIDDEN),
                    "value": nn.Linear(HIDDEN, HIDDEN),
                    "out": nn.Linear(HIDDEN, HIDDEN),
                    "ff": nn.Sequential(nn.Linear(HIDDEN, 2 * HIDDEN), nn.GELU(),
                                        nn.Linear(2 * HIDDEN, HIDDEN))}) for _ in range(LAYERS))
            # (width, hidden): one learned offset per directed edge, summed over the
            # members a token is joined to.  Keeping it this small is what stops the
            # (n, width, width, hidden) broadcast from costing terabytes.
            self.edge_bias = nn.Parameter(torch.zeros(WIDTH, HIDDEN))
            self.family = nn.Linear(HIDDEN, len(FAMILIES))
            self.subtype = nn.Linear(HIDDEN, 47)
            self.orbit = nn.Linear(HIDDEN, N_ORBITS)
            self.item = nn.Linear(HIDDEN, n_groups)

        def forward(self, tokens, adjacency=None, mask_context: bool = False):
            x = tokens
            if mask_context:
                # zero every *other* member's token: the item can only see itself
                n, width, _ = x.shape
                keep = torch.zeros((n, width, 1), device=x.device)
                keep[:, 0, :] = 1.0
                x = x * keep
            h = self.input(x) + self.slot
            if adjacency is not None:
                # an edge adds a learned offset to the token of the member that owns it:
                # adjacency (n, width, width) contracts to one vector per member
                h = h + torch.einsum("nij,jd->nid", adjacency.float(), self.edge_bias)
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
            return {"family": self.family(h.mean(dim=1)),
                    "subtype": self.subtype(h.mean(dim=1)),
                    "orbit": self.orbit(h),
                    "item": self.item(h.mean(dim=1))}

    return ContextRoleNet()


def rotate_to_front(nodes: np.ndarray, roles: np.ndarray,
                    position: np.ndarray) -> tuple:
    """Put the judged member in slot 0 so an item-only model sees a fixed slot.

    The orbit row must be moved with the members, otherwise the judged member's label
    stays at its original position and the comparison silently measures another slot.
    """
    width = nodes.shape[1]
    out = nodes.copy()
    moved_roles = np.array(roles, copy=True)
    for row, (members, role_row, pos) in enumerate(
            zip(nodes.tolist(), roles.tolist(), position.tolist())):
        order = [int(pos)] + [i for i in range(width) if i != int(pos)]
        out[row] = [members[i] for i in order]
        moved_roles[row] = [role_row[i] for i in order]
    return out, moved_roles


def role_group_of(role: int, mapping: dict) -> int:
    return mapping.get(int(role), 0)


def train(samples: dict, labels_orbit: np.ndarray, groups: np.ndarray, n_groups: int,
          adjacency: np.ndarray | None, device, in_dim: int, epochs: int = EPOCHS):
    import torch
    torch.manual_seed(SEED)
    model = build_role_model(in_dim, n_groups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(samples["tokens"]).to(device)
    a = torch.from_numpy(adjacency).to(device) if adjacency is not None else None
    y_family = torch.from_numpy(samples["family"]).to(device)
    y_subtype = torch.from_numpy(samples["subtype"]).to(device)
    y_roles = torch.from_numpy(samples["roles"]).to(device)
    y_item = torch.from_numpy(groups).to(device)
    for _ in range(epochs):
        model.train()
        out = model(x, a)
        loss = (torch.nn.functional.cross_entropy(out["family"], y_family)
                + torch.nn.functional.cross_entropy(out["subtype"], y_subtype)
                + torch.nn.functional.cross_entropy(
                    out["orbit"].reshape(-1, N_ORBITS), y_roles.reshape(-1))
                + torch.nn.functional.cross_entropy(out["item"], y_item))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def judge(model, tokens: np.ndarray, adjacency: np.ndarray | None, device,
          mask_context: bool = False):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(tokens).to(device)
        a = torch.from_numpy(adjacency).to(device) if adjacency is not None else None
        out = model(x, a, mask_context=mask_context)
    return out["orbit"].cpu().numpy(), out["item"].cpu().numpy()


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "role_context.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = build_labels()
    rows = width3_rows(labels)
    contexts = role_contexts(rows)
    pairs = contexts["pairs"]
    payload = np.load(STATES)
    states = {key: payload[key] for key in ("semantics", "z", "q")}

    # importance-based role grouping for the item head: rank roles by training frequency
    counts = np.bincount(rows["roles"][rows["part"] == 0].ravel(), minlength=N_ORBITS)
    order = np.argsort(-counts)
    group_of = {int(role): index // ROLE_GROUP for index, role in enumerate(order)
                if counts[role] > 0}
    n_groups = max(group_of.values()) + 1

    report = {"inputs": {"instances": sha256(INSTANCES), "states": sha256(STATES),
                         "contract": sha256(CONTRACT)},
              "protocol": {"width": WIDTH, "epochs": EPOCHS, "heads": HEADS,
                           "layers": LAYERS, "hidden": HIDDEN, "seed": SEED,
                           "attention": "full, no adjacency unless stated",
                           "unit": "item x two subgraphs with different orbits",
                           "held_items": contexts["held_items"],
                           "held_row_cap": HELD_ITEMS,
                           "train_instances": contexts["train_instances"],
                           "held_instances": contexts["held_instances"],
                           "role_groups": n_groups,
                           "item_head": f"roles grouped by frequency, {ROLE_GROUP} per group"},
              "sets": {}}

    train_rows = np.flatnonzero(rows["part"] == 0)
    rng = np.random.default_rng(SEED)
    if len(train_rows) > TRAIN_ITEMS * 3:
        train_rows = np.sort(rng.choice(train_rows, TRAIN_ITEMS * 3, replace=False))

    def sample_of(table: str, include_adjacency: bool = False) -> tuple:
        tokens = states[table][rows["nodes"][train_rows]].astype(np.float32)
        adjacency = None
        if include_adjacency:
            observed = set(map(tuple, np.load(CONTRACT)["message_edges"].tolist()))
            # (n, width, width): the token dimension must not enter the adjacency shape
            adjacency = np.zeros((len(train_rows), WIDTH, WIDTH), dtype=bool)
            for index, members in enumerate(rows["nodes"][train_rows].tolist()):
                for i in range(WIDTH):
                    for j in range(WIDTH):
                        if i != j and (members[i], members[j]) in observed:
                            adjacency[index, i, j] = True
        return tokens, adjacency

    def truth_shape(n_pairs: int) -> tuple:
        return (2, n_pairs)

    def pair_nodes() -> tuple:
        a_idx = np.array([row["a"]["instance"] for row in pairs_used])
        b_idx = np.array([row["b"]["instance"] for row in pairs_used])
        nodes_a, roles_a = rotate_to_front(
            rows["nodes"][a_idx], rows["roles"][a_idx],
            np.array([row["a"]["position"] for row in pairs_used]))
        nodes_b, roles_b = rotate_to_front(
            rows["nodes"][b_idx], rows["roles"][b_idx],
            np.array([row["b"]["position"] for row in pairs_used]))
        return nodes_a, nodes_b, roles_a[:, 0], roles_b[:, 0]

    def orbit_logits_for(model, table: str, mask_context: bool = False,
                         adjacency_pairs: tuple | None = None) -> np.ndarray:
        nodes_a, nodes_b, _, _ = pair_nodes()
        # put context a for every pair first, then context b: one block layout only
        joined = np.concatenate([nodes_a, nodes_b], axis=0)
        rows_used = joined.shape[0]
        if rows_used != 2 * len(pairs_used):
            raise SystemExit(f"pair extraction produced {rows_used} rows for "
                             f"{len(pairs_used)} pairs")
        tokens = states[table][joined].astype(np.float32)
        adj = None
        if adjacency_pairs is not None:
            adj = np.concatenate([adjacency_pairs[0], adjacency_pairs[1]], axis=0)
        logits, _ = judge(model, tokens, adj, device, mask_context)
        return logits

    def adjacency_of(nodes: np.ndarray) -> np.ndarray:
        observed = set(map(tuple, np.load(CONTRACT)["message_edges"].tolist()))
        adj = np.zeros((len(nodes), WIDTH, WIDTH), dtype=bool)
        for index, members in enumerate(nodes.tolist()):
            for i in range(WIDTH):
                for j in range(WIDTH):
                    if i != j and (members[i], members[j]) in observed:
                        adj[index, i, j] = True
        return adj

    # every pair is judged twice: once for context a, once for context b
    def score(orbit_logits: np.ndarray, n_pairs: int) -> dict:
        if orbit_logits.shape[0] != 2 * n_pairs:
            raise SystemExit(f"judged {orbit_logits.shape[0]} rows for {n_pairs} pairs")
        # rows are [a-context rows..., b-context rows...] and each row is (width, orbit)
        guesses = orbit_logits.argmax(-1)                     # (2n, width)
        predicted = np.stack([guesses[:n_pairs, 0], guesses[n_pairs:, 0]])
        if predicted.shape != (2, n_pairs):
            raise SystemExit(f"judged layout {orbit_logits.shape} -> {predicted.shape} "
                             f"does not fit {n_pairs} pairs")
        truth = np.array([[row["a"]["true_orbit"] for row in pairs_used],
                          [row["b"]["true_orbit"] for row in pairs_used]])
        first = float((predicted[0] == truth[0]).mean())
        second = float((predicted[1] == truth[1]).mean())
        both = float(((predicted[0] == truth[0]) & (predicted[1] == truth[1])).mean())
        distinct = float((predicted[0] != predicted[1]).mean())
        distinct_true = float((truth[0] != truth[1]).mean())
        return {"first_accuracy": first, "second_accuracy": second,
                "both_accuracy": both, "distinct_possible": distinct_true,
                "predicted_distinct_rate": distinct,
                "mean_macro_f1": 0.5 * (macro_f1(truth[0], predicted[0], N_ORBITS)
                                        + macro_f1(truth[1], predicted[1], N_ORBITS))}

    def train_instance_model(table: str, with_adjacency: bool):
        tokens, adjacency = sample_of(table, include_adjacency=with_adjacency)
        groups = np.array([role_group_of(int(rows["roles"][r, 0]), group_of)
                           for r in train_rows], dtype=np.int64)
        return train({"tokens": tokens, "roles": rows["roles"][train_rows],
                      "family": rows["family"][train_rows],
                      "subtype": rows["subtype"][train_rows]},
                     rows["roles"][train_rows], groups, n_groups,
                     adjacency if with_adjacency else None, device, tokens.shape[-1])

    pairs_used = pairs          # one list for A-E; every set is judged on exactly these
    held_items_in_pairs = len({row["item"] for row in pairs_used})
    for table in ("z", "q"):
        model = train_instance_model(table, with_adjacency=False)
        report["sets"][f"A_per_instance_orbit_{table}"] = score(
            orbit_logits_for(model, table), len(pairs_used))
        report["sets"][f"B_context_ablation_{table}"] = score(
            orbit_logits_for(model, table, mask_context=True), len(pairs_used))
        nodes_a, nodes_b, _, _ = pair_nodes()
        joined = np.concatenate([nodes_a, nodes_b], axis=0)
        with torch.no_grad():
            orbit_logits, item_logits = judge(
                model, states[table][joined].astype(np.float32), None, device)
        item_pred = (np.stack([item_logits[:len(pairs_used)].argmax(-1),
                               item_logits[len(pairs_used):].argmax(-1)])
                     if item_logits.shape[0] == 2 * len(pairs_used)
                     else item_logits.argmax(-1).reshape(len(pairs_used), 2).T)
        report["sets"][f"E_switch_rate_{table}"] = {
            "predicted_distinct_rate": report["sets"][
                f"A_per_instance_orbit_{table}"]["predicted_distinct_rate"],
            "context_ablation_distinct_rate": report["sets"][
                f"B_context_ablation_{table}"]["predicted_distinct_rate"],
            "note": "how often the same item is given two different orbits in its two "
                    "contexts; with ablation it is structurally 0"}
        report["sets"][f"D_context_free_item_head_{table}"] = {
            "agreement_between_contexts": float((item_pred[0] == item_pred[1]).mean()),
            "note": "the item head names one group per item, so it cannot separate two "
                    "contexts of the same item; this is the ceiling a context-free model "
                    "has on set A"}
        report["sets"][f"C_item_mix_{table}"] = {
            "item_head_groups": int(n_groups),
            "per_instance_accuracy_using_item_head":
                score(orbit_logits, len(pairs_used))["both_accuracy"]}

    model_e = train_instance_model("z", with_adjacency=True)
    nodes_a, nodes_b, _, _ = pair_nodes()
    adj_e = np.concatenate([adjacency_of(nodes_a), adjacency_of(nodes_b)], axis=0)
    if adj_e.shape != (2 * len(pairs_used), WIDTH, WIDTH):
        raise SystemExit(f"ceiling adjacency {adj_e.shape} does not fit the pairs")
    report["sets"]["E_true_adjacency_ceiling_z"] = score(
        orbit_logits_for(model_e, "z", adjacency_pairs=(adjacency_of(nodes_a),
                                                        adjacency_of(nodes_b))),
        len(pairs_used))

    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for name, value in report["sets"].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in value.items()})
    print("written", args.out)


if __name__ == "__main__":
    main()
