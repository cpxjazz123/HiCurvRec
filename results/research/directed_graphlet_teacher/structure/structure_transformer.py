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
    """Attention mask from the tokens themselves; never from the real adjacency.

    `tokens` is (n, width, d).  Every member attends to itself and to its `k` nearest
    members, which is all the structural hint the model gets.
    """
    unit = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-12)
    similarity = np.einsum("nid,njd->nij", unit, unit)
    width = tokens.shape[1]
    allowed = np.zeros_like(similarity, dtype=bool)
    for row in range(len(tokens)):
        taken = np.argsort(-similarity[row], axis=1)[:, :k + 1]
        for position in range(width):
            allowed[row, position, taken[position]] = True
        allowed[row] |= allowed[row].T          # symmetric: attention is not directional
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


def run_mixed(samples: dict, states: dict, name: str, mask_mode: str, device,
              epochs: int = EPOCHS) -> dict:
    """Predict family, subtype and member orbits for a mixed-width batch."""
    import torch

    pool_train = build_dataset(samples, states, name, mask_mode, "train")
    held = build_dataset(samples, states, name, mask_mode, "heldout")
    # every family keeps its own train/held-out rows (the pool's split), so the training
    # set still covers all five families; the ratio follows the pool rather than an
    # artificial balance, which is why the majority baseline is reported next to it
    train = pool_train
    subtypes = sorted(set(train["subtype"].tolist()))
    subtype_map = {value: position for position, value in enumerate(subtypes)}
    n_orbits = int(train["roles"].max()) + 1
    torch.manual_seed(SEED)
    # family and subtype heads must answer for labels never trained on
    families_seen = np.unique(train["family"])
    family_space = max(int(families_seen.max()) + 1, len(FAMILIES))
    subtype_space = max(len(subtypes), int(pool_train["subtype"].max()) + 1)
    model = build_model(train["tokens"].shape[-1], family_space, subtype_space,
                        n_orbits, PAD).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def to_tensors(piece):
        return (torch.from_numpy(piece["tokens"]).to(device),
                torch.from_numpy(piece["mask"]).to(device),
                torch.from_numpy(piece["member"]).to(device),
                torch.from_numpy(piece["roles"]).to(device),
                torch.from_numpy(piece["role_mask"]).to(device),
                torch.tensor([subtype_map.get(int(v), -1) for v in piece["subtype"]],
                             dtype=torch.long, device=device),
                torch.from_numpy(piece["family"]).to(device))

    x, mask, member, roles, role_mask, subtype, family = to_tensors(train)
    trained = subtype >= 0
    x, mask, member, roles, role_mask, subtype, family = (
        x[trained], mask[trained], member[trained], roles[trained],
        role_mask[trained], subtype[trained], family[trained])
    for _ in range(epochs):
        model.train()
        out = model(x, mask, member)
        orbit_loss = torch.nn.functional.cross_entropy(
            out["orbit"][role_mask], roles[role_mask]) if role_mask.any() else 0.0
        loss = (torch.nn.functional.cross_entropy(out["family"], family)
                + torch.nn.functional.cross_entropy(out["subtype"], subtype)
                + orbit_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    hx, hmask, hmember, hroles, hrole_mask, hsubtype, hfamily = to_tensors(held)
    with torch.no_grad():
        out = model(hx, hmask, hmember)
        prediction = out["orbit"].argmax(-1)
        truth, guess = hroles[hrole_mask], prediction[hrole_mask]
        f1_scores = []
        for label in torch.unique(truth):
            hit = (truth == label) & (guess == label)
            missed = (truth == label) & (guess != label)
            tp = int(hit.sum())
            fp = int(((truth != label) & (guess == label)).sum())
            fn = int(missed.sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1_scores.append(0.0 if precision + recall == 0 else
                             2 * precision * recall / (precision + recall))
        # every family has its own orbit vocabulary, so the control is per family slot
        roles_np = train["roles"]
        majority_hits, positions = 0, 0
        for family_index, family_name in enumerate(FAMILIES):
            rows = np.flatnonzero(train["family"] == family_index)
            held_rows = np.flatnonzero(hfamily.cpu().numpy() == family_index)
            for position in range(WIDTH[family_name]):
                votes = torch.bincount(
                    torch.from_numpy(roles_np[rows, position]).to(device),
                    minlength=n_orbits)
                guess_label = int(votes.argmax())
                slots = held_rows[hrole_mask.cpu().numpy()[held_rows, position]]
                if len(slots) == 0:
                    continue
                majority_hits += int((hroles[slots, position] == guess_label).sum())
                positions += len(slots)
        # subtype accuracy is reported on the out-of-vocabulary rows only: the head's
        # argmax cannot reach an unseen label, so the number says how often it lands on
        # a *seen* subtype, which is the honest ceiling for a closed-set head
        oov = hsubtype < 0
        seen_predictions = out["subtype"].argmax(-1)
        return {"heldout": int(hrole_mask.sum()),
                "family_accuracy": float((out["family"].argmax(-1) == hfamily).float().mean()),
                "family_majority_baseline": float(max(
                    np.bincount(train["family"]).tolist()) / len(train["family"])),
                "subtype_accuracy": float(
                    (seen_predictions[~oov] == hsubtype[~oov]).float().mean())
                    if (~oov).any() else float("nan"),

                "orbit_accuracy": float((truth == guess).float().mean()),
                "orbit_macro_f1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
                "orbit_majority_accuracy": majority_hits / max(positions, 1),
                "subtype_classes_trained": len(subtypes), "orbit_classes": n_orbits}


def main() -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not STATES_FILE.exists():
        raise SystemExit("run fidelity/fidelity_study.py first: it exports the four states")
    payload = np.load(STATES_FILE)
    states = {key: payload[key] for key in ("semantics", "z", "q", "decoder")}
    samples = load_samples()
    meta = json.loads(STATES_META.read_text())
    report = {"checkpoint": meta,
              "protocol": {"seed": SEED, "epochs": EPOCHS, "heads": HEADS, "layers": LAYERS,
                           "hidden": HIDDEN, "max_per_family": MAX_PER_FAMILY,
                           "knn_k": K_NEIGHBOURS,
                           "split": "instance pool split 0 trains, split 1 is held out",
                           "answers": "family, directed subtype, per-member orbit from the "
                                      "real training graph; adjacency never given"},
              "runs": {}}
    for name in ("q", "z", "semantics", "decoder"):
        report["runs"][name] = {}
        for mask_mode in ("knn", "none"):
            row = run_mixed(samples, states, name, mask_mode, device)
            report["runs"][name][mask_mode] = row
            print(f"  {name}/{mask_mode}: family {row['family_accuracy']:.3f} "
                  f"(prior {row['family_majority_baseline']:.3f}) "
                  f"orbit {row['orbit_accuracy']:.3f} "
                  f"F1 {row['orbit_macro_f1']:.3f} "
                  f"majority {row['orbit_majority_accuracy']:.3f}", flush=True)
            tmp = HERE / "structure_transformer.json.tmp"
            tmp.write_text(json.dumps(report, indent=1) + "\n")
            tmp.replace(HERE / "structure_transformer.json")
    print("written", HERE / "structure_transformer.json")


if __name__ == "__main__":
    main()
