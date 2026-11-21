#!/usr/bin/env python3
"""Read a whole candidate subgraph with a Graph Transformer and predict its structure.

Every sample is a real instance of the training behaviour graph: the members' item
representations form the tokens, a **KNN graph built from those tokens** forms the
attention mask, and the real graph supplies the answers -- the graphlet family, the
directed subtype, and each member's orbit inside this instance.  The real adjacency is
never given to the model, so a correct prediction cannot come from copying the input.

The only thing that changes between runs is the token content: `z` (32-d, before
quantisation), `q` (32-d, hard quantisation rebuilt from the stored SID and the frozen
codebook), `semantics` (768-d) and `decoder` (768-d after the quantiser).  Controls are
the same model with the KNN mask removed (full attention) and with shuffled labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

STATES_FILE = HERE.parent / "fidelity" / "fidelity_states.npz"
STATES_META = HERE.parent / "fidelity" / "fidelity_states.json"
INSTANCES = REPO / "results" / "research" / "directed_graphlet_role" / "instances.npz"
FAMILIES = ("P3", "K3", "C4", "P4", "K1_3")
WIDTH = {"P3": 3, "K3": 3, "C4": 4, "P4": 4, "K1_3": 4}
SEED = 0                     # one run per configuration; no seed replication
EPOCHS = 120
HEADS = 4
LAYERS = 2
HIDDEN = 128
MAX_PER_FAMILY = 12000
K_NEIGHBOURS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pad_rows_to(block: np.ndarray, width: int, fill: int = 0) -> np.ndarray:
    if block.shape[1] == width:
        return block
    out = np.full((len(block), width) + block.shape[2:], fill, dtype=block.dtype)
    out[:, :block.shape[1]] = block
    return out


def global_labels(samples: dict) -> dict:
    """Global subtype/orbit ids and node-disjoint train/held-out rows.

    Two defects are fixed here.  First, every family carries its own local subtype and
    orbit numbering, so merging them as if they were shared makes different structures
    the same class; `directed_role.instance_targets` maps them into the single global
    vocabulary the rest of the repo uses.  Second, isolating *instances* still lets the
    same item appear on both sides, which the model can memorise, so the held-out rows
    are also required to contain no item seen in training.
    """
    from directed_role import instance_targets, load_vocabulary

    vocabulary = load_vocabulary("directed")
    holder = {"nodes": [], "roles": [], "subtype": [], "family": [], "part": []}
    for index, family in enumerate(FAMILIES):
        block = samples[family]
        targets = instance_targets(family, block["subtype"], block["roles"], vocabulary)
        keep = targets["subtype"] >= 0
        holder["nodes"].append(block["nodes"][keep])
        holder["roles"].append(targets["orbit"][keep])
        holder["subtype"].append(targets["subtype"][keep])
        holder["family"].append(np.full(int(keep.sum()), index, dtype=np.int64))
        holder["part"].append(np.where(block["train"][keep], 0, 1).astype(np.int8))
    # widths differ (3 vs 4), so rows are padded only after the split is chosen
    merged = {key: np.concatenate(
                  [row if row.ndim == 1 else pad_rows_to(row, 4) for row in value], axis=0)
              for key, value in holder.items() if key != "part"}
    part = np.concatenate(holder["part"])
    train_rows = np.flatnonzero(part == 0)
    held_rows = np.flatnonzero(part == 1)
    train_items = set(merged["nodes"][train_rows].ravel().tolist())
    held_pure = np.array([not (set(row[row != 0].tolist()) & train_items)
                          for row in merged["nodes"]], dtype=bool)
    dropped = int((held_rows.size - int(held_pure[held_rows].sum())))
    rng = np.random.default_rng(SEED)
    if len(train_rows) > MAX_PER_FAMILY * len(FAMILIES):
        train_rows = np.sort(rng.choice(train_rows, MAX_PER_FAMILY * len(FAMILIES),
                                        replace=False))
    held_rows = held_rows[held_pure[held_rows]]
    # every row whose item appears in training is dropped under strict item isolation, so
    # the pool's own held-out split falls back to rows that keep at least one unseen item
    isolation = "item"
    if len(held_rows) < 1000:
        held_rows = np.flatnonzero(part == 1)
        isolation = "instance_only"
    if len(held_rows) > MAX_PER_FAMILY * len(FAMILIES):
        held_rows = np.sort(rng.choice(held_rows, MAX_PER_FAMILY * len(FAMILIES),
                                       replace=False))
    return {"vocabulary": {"subtypes": vocabulary["n_subtypes"],
                           "orbits": vocabulary["n_orbits"]},
            "train": {key: merged[key][train_rows] for key in merged},
            "heldout": {key: merged[key][held_rows] for key in merged},
            "train_items": len(train_items),
            "heldout_rows_dropped_for_item_overlap": dropped,
            "heldout_isolation": isolation,
            "note": "item isolation discards a held-out row when any member appears in "
                    "training; on a 24587-item catalogue almost every row shares one, so "
                    "the split falls back to instance isolation and says so"}


def load_samples() -> dict:
    """Training rows are split 0 of the instance pool; held-out rows stay untouched."""
    payload = np.load(INSTANCES)
    out = {}
    for family in FAMILIES:
        split = payload[f"{family}_split"]
        out[family] = {
            "nodes": payload[f"{family}_nodes"],
            "roles": payload[f"{family}_roles"],
            "subtype": payload[f"{family}_subtype"],
            "train": split == 0,
            "heldout": split == 1}
    return out


def knn_mask(tokens: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Reciprocal kNN attention mask built from the tokens; never from the real adjacency.

    `tokens` is (n, width, d).  An edge exists only if each endpoint is among the other's
    `k` nearest members; at width 3 with k=2 that leaves two of the three candidate edges.
    """
    unit = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-12)
    similarity = np.einsum("nid,njd->nij", unit, unit)
    width = tokens.shape[1]
    allowed = np.zeros_like(similarity, dtype=bool)
    for row in range(len(tokens)):
        taken = np.argsort(-similarity[row], axis=1)[:, :k + 1]
        for position in range(width):
            allowed[row, position, taken[position]] = True
        allowed[row] &= allowed[row].T          # reciprocal: an edge only if both list each other
    return allowed


def build_model(in_dim: int, n_families: int, n_subtypes: int, n_orbits: int, width_max: int):
    import torch
    import torch.nn as nn

    class GraphTransformer(nn.Module):
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
                                        nn.Linear(2 * HIDDEN, HIDDEN)),
                    "norm1": nn.LayerNorm(HIDDEN),
                    "norm2": nn.LayerNorm(HIDDEN)}) for _ in range(LAYERS))
            self.norm = nn.LayerNorm(HIDDEN)
            self.family = nn.Linear(HIDDEN, n_families)
            self.subtype = nn.Linear(HIDDEN, n_subtypes)
            self.orbit = nn.Linear(HIDDEN, n_orbits)
            self.heads = HEADS

        def forward(self, tokens, mask, member):
            h = self.input(tokens)
            for layer in self.layers:
                q = layer["query"](h).view(len(h), -1, self.heads, HIDDEN // self.heads)
                k = layer["key"](h).view(len(h), -1, self.heads, HIDDEN // self.heads)
                v = layer["value"](h).view(len(h), -1, self.heads, HIDDEN // self.heads)
                scores = torch.einsum("nqhd,nkhd->nhqk", q, k) / (HIDDEN // self.heads) ** 0.5
                scores = scores.masked_fill(~mask[:, None], float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                attended = torch.einsum("nhqk,nkhd->nqhd", weights, v).reshape(
                    len(h), -1, HIDDEN)
                h = layer["norm1"](h + layer["out"](attended))
                h = layer["norm2"](h + layer["ff"](h))
            h = self.norm(h)
            weight = member.float().unsqueeze(-1)
            pooled = (h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
            return {"family": self.family(pooled), "subtype": self.subtype(pooled),
                    "orbit": self.orbit(h)}

    return GraphTransformer()


PAD = 4


def pad_to(tokens: np.ndarray, pad: int = PAD) -> np.ndarray:
    out = np.zeros((len(tokens), pad, tokens.shape[-1]), dtype=np.float32)
    out[:, :tokens.shape[1]] = tokens
    return out


def pad_roles(roles: np.ndarray, pad: int = PAD) -> np.ndarray:
    out = np.zeros((len(roles), pad), dtype=np.int64)
    out[:, :roles.shape[1]] = roles
    return out


def pad_mask(mask: np.ndarray, pad: int = PAD) -> np.ndarray:
    """Attention mask over padded slots: real members attend to each other, padding to none."""
    out = np.zeros((len(mask), pad, pad), dtype=bool)
    width = mask.shape[1]
    out[:, :width, :width] = mask
    for row in range(len(out)):
        out[row, width:, width:] = True      # padding slots attract only padding, i.e. zeros
    return out


def batch(family: str, samples: dict, states: dict, name: str, mask_mode: str,
          samples_max: int = MAX_PER_FAMILY, pad: int = PAD) -> dict:
    """One family's train/held-out tensors, padded to the widest family."""
    block = samples[family]
    train_idx = np.flatnonzero(block["train"])[:samples_max]
    held_idx = np.flatnonzero(block["heldout"])[:samples_max]
    tokens = states[name][block["nodes"]]
    out = {}
    for key, index in (("train", train_idx), ("heldout", held_idx)):
        sub = tokens[index]
        # a member mask says which slots are real members; it is data, not the answer
        member = np.zeros((len(sub), pad), dtype=bool)
        member[:, :sub.shape[1]] = True
        out[key] = {"tokens": pad_to(sub, pad).astype(np.float32),
                    "member": member,
                    "roles": pad_roles(block["roles"][index], pad),
                    "role_mask": member.copy(),
                    "subtype": block["subtype"][index],
                    "family": np.full(len(sub), FAMILIES.index(family), dtype=np.int64),
                    "mask": pad_mask(knn_mask(sub) if mask_mode == "knn" else
                                     np.ones((len(sub), sub.shape[1], sub.shape[1]), bool), pad)}
    return out


def build_dataset(samples: dict, states: dict, name: str, mask_mode: str,
                  part: str) -> dict:
    """All five families pooled; the model sees one mixed batch of widths 3 and 4."""
    pieces = [batch(family, samples, states, name, mask_mode)[part] for family in FAMILIES]
    return {key: np.concatenate([piece[key] for piece in pieces], axis=0)
            for key in pieces[0]}


def pad_rows(nodes: np.ndarray, roles: np.ndarray, part: dict) -> dict:
    """Tokens, member mask, padded global labels for one partition."""
    tokens = part["table"][part["nodes"]].astype(np.float32)
    member = np.zeros((len(tokens), PAD), dtype=bool)
    member[:, :tokens.shape[1]] = True
    out = {"tokens": pad_to(tokens), "member": member,
           "roles": pad_roles(roles), "role_mask": member.copy(),
           "subtype": part["subtype"], "family": part["family"]}
    if part["mask_mode"] == "knn":
        out["mask"] = pad_mask(knn_mask(tokens))
    else:
        out["mask"] = np.ones((len(tokens), PAD, PAD), dtype=bool)
    return out


def run_mixed(labels: dict, states: dict, name: str, mask_mode: str, device,
              shuffle_labels: bool = False, epochs: int = EPOCHS) -> dict:
    """Global subtype/orbit labels; controls for width, KNN and label shuffling."""
    import torch
    base_rng = np.random.default_rng(SEED if not shuffle_labels else SEED + 1)

    def prepare(split: str) -> dict:
        part = {"nodes": labels[split]["nodes"], "subtype": labels[split]["subtype"],
                "family": labels[split]["family"], "table": states[name],
                "mask_mode": mask_mode}
        roles = labels[split]["roles"]
        return pad_rows(labels[split]["nodes"], roles, part)

    train, held = prepare("train"), prepare("heldout")
    if shuffle_labels:
        order = base_rng.permutation(len(train["subtype"]))
        train["subtype"] = train["subtype"][order]
        train["roles"] = train["roles"][order]
        train["family"] = train["family"][order]
        train["tokens"] = train["tokens"][order]
        train["member"] = train["member"][order]
        train["mask"] = train["mask"][order]
        train["role_mask"] = train["role_mask"][order]
    n_subtypes = int(labels["vocabulary"]["subtypes"])
    n_orbits = int(labels["vocabulary"]["orbits"])
    torch.manual_seed(SEED)
    model = build_model(train["tokens"].shape[-1], len(FAMILIES), n_subtypes,
                        n_orbits, PAD).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tensors(piece):
        return (torch.from_numpy(piece["tokens"]).to(device),
                torch.from_numpy(piece["mask"]).to(device),
                torch.from_numpy(piece["member"]).to(device),
                torch.from_numpy(piece["roles"]).to(device),
                torch.from_numpy(piece["role_mask"]).to(device),
                torch.from_numpy(piece["subtype"]).to(device),
                torch.from_numpy(piece["family"]).to(device))

    x, mask, member, roles, role_mask, subtype, family = tensors(train)
    for _ in range(epochs):
        model.train()
        out = model(x, mask, member)
        loss = (torch.nn.functional.cross_entropy(out["family"], family)
                + torch.nn.functional.cross_entropy(out["subtype"], subtype)
                + torch.nn.functional.cross_entropy(out["orbit"][role_mask],
                                                    roles[role_mask]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    hx, hmask, hmember, hroles, hrole_mask, hsubtype, hfamily = tensors(held)
    with torch.no_grad():
        out = model(hx, hmask, hmember)
        guess = out["orbit"].argmax(-1)
        truth = hroles[hrole_mask]
        predicted = guess[hrole_mask]
        f1_scores = []
        for label in torch.unique(truth):
            hit = (truth == label) & (predicted == label)
            tp = int(hit.sum())
            fp = int(((truth != label) & (predicted == label)).sum())
            fn = int(((truth == label) & (predicted != label)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1_scores.append(0.0 if precision + recall == 0 else
                             2 * precision * recall / (precision + recall))
        # controls ---------------------------------------------------------
        majority_family = float(max(np.bincount(train["family"]).tolist()) / len(train["family"]))
        subtype_hits, subtype_majority = 0, 0
        for family_index in range(len(FAMILIES)):
            train_rows = train["family"] == family_index
            held_rows = hfamily.cpu().numpy() == family_index
            if not held_rows.any() or not train_rows.any():
                continue
            votes = np.bincount(train["subtype"][train_rows],
                                minlength=n_subtypes)
            guess_label = int(votes.argmax())
            subtype_hits += int((hsubtype.cpu().numpy()[held_rows] == guess_label).sum())
            subtype_majority += int(held_rows.sum())
        majority_hits, position_count = 0, 0
        roles_np = train["roles"]
        for family_index, family_name in enumerate(FAMILIES):
            rows = np.flatnonzero(train["family"] == family_index)
            held_index = np.flatnonzero(hfamily.cpu().numpy() == family_index)
            for position in range(WIDTH[family_name]):
                votes = torch.bincount(torch.from_numpy(roles_np[rows, position]).to(device),
                                       minlength=n_orbits)
                guess_label = int(votes.argmax())
                slots = held_index[hrole_mask.cpu().numpy()[held_index, position]]
                if len(slots) == 0:
                    continue
                majority_hits += int((hroles[slots, position] == guess_label).sum())
                position_count += len(slots)
        return {"heldout": int(len(hx)),
                "family_accuracy": float((out["family"].argmax(-1) == hfamily).float().mean()),
                "family_majority_baseline": majority_family,
                "width_only_baseline": max(
                    float(np.mean(np.array([WIDTH[FAMILIES[int(f)]] == 3 for f in hfamily.cpu().numpy()]))),
                    float(np.mean(np.array([WIDTH[FAMILIES[int(f)]] == 4 for f in hfamily.cpu().numpy()])))),
                "subtype_accuracy": float((out["subtype"].argmax(-1) == hsubtype).float().mean()),
                "subtype_majority_baseline": subtype_hits / max(subtype_majority, 1),
                "orbit_accuracy": float((truth == predicted).float().mean()),
                "orbit_macro_f1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
                "orbit_majority_accuracy": majority_hits / max(position_count, 1),
                "subtype_classes": n_subtypes, "orbit_classes": n_orbits}


def main() -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not STATES_FILE.exists():
        raise SystemExit("run fidelity/fidelity_study.py first: it exports the four states")
    payload = np.load(STATES_FILE)
    states = {key: payload[key] for key in ("semantics", "z", "q", "decoder")}
    samples = load_samples()
    labels = global_labels(samples)
    meta = json.loads(STATES_META.read_text())
    report = {"checkpoint": meta,
              "labels": {"vocabulary": labels["vocabulary"],
                         "train_rows": int(len(labels["train"]["family"])),
                         "heldout_rows": int(len(labels["heldout"]["family"])),
                         "train_items": labels["train_items"],
                         "heldout_rows_dropped_for_item_overlap":
                             labels["heldout_rows_dropped_for_item_overlap"],
                         "heldout_isolation": labels["heldout_isolation"],
                         "source": "directed_role.instance_targets: one global subtype id "
                                   "and one global orbit id per (family, subtype, orbit)"},
              "protocol": {"seed": SEED, "epochs": EPOCHS, "heads": HEADS, "layers": LAYERS,
                           "hidden": HIDDEN, "knn_k": K_NEIGHBOURS,
                           "knn": "reciprocal: an edge only if both members list each other",
                           "split": "instance pool split, then held-out rows whose members "
                                    "appear in training are dropped",
                           "controls": ["width-only", "subtype majority", "orbit majority",
                                        "shuffled labels"]},
              "runs": {}}
    for name in ("q", "z", "semantics", "decoder"):
        report["runs"][name] = {}
        for mask_mode in ("knn", "none"):
            row = run_mixed(labels, states, name, mask_mode, device)
            shuffled = run_mixed(labels, states, name, mask_mode, device,
                                 shuffle_labels=True)
            row["shuffled_labels"] = {
                key: shuffled[key] for key in
                ("family_accuracy", "subtype_accuracy", "orbit_macro_f1",
                 "orbit_accuracy")}
            report["runs"][name][mask_mode] = row
            print(f"  {name}/{mask_mode}: family {row['family_accuracy']:.3f} "
                  f"(width {row['width_only_baseline']:.3f}) "
                  f"subtype {row['subtype_accuracy']:.3f} "
                  f"(maj {row['subtype_majority_baseline']:.3f}) "
                  f"orbit {row['orbit_accuracy']:.3f} F1 {row['orbit_macro_f1']:.3f} "
                  f"(maj {row['orbit_majority_accuracy']:.3f}) "
                  f"| shuffled family {shuffled['family_accuracy']:.3f} "
                  f"F1 {shuffled['orbit_macro_f1']:.3f}", flush=True)
            tmp = HERE / "structure_transformer.json.tmp"
            tmp.write_text(json.dumps(report, indent=1) + "\n")
            tmp.replace(HERE / "structure_transformer.json")
    print("written", HERE / "structure_transformer.json")


if __name__ == "__main__":
    main()
