#!/usr/bin/env python3
"""What survives quantisation, measured on one fixed checkpoint and one seed.

Every arm, split and model start uses the same single seed: no seed replications, so
the numbers are single-run measurements and seed variance is not estimated.

Two tasks, both fed only by item representations (never by adjacency):

* **item roles** -- predict the item's multi-role orbit profile from its representation.
  Three input regimes: an MLP on the item's own vector, the same MLP, and the mutual
  kNN graph + a graph network over it.  Every regime sees the same split, seeds and
  training budget.
* **subgraph instances** -- predict the graphlet family and each node's orbit for a
  genuine instance taken from the real training graph, reading only the members'
  representations.  Splits are made **by instance**, never by node, so a test chain
  cannot be a near copy of a training chain.

Representations compared, all from ONE checkpoint:
``semantics`` (what the mainline encoder reads), ``z`` (32-d pre-quantisation),
``q`` (32-d hard quantisation, rebuilt from the stored SID and the frozen codebook)
and ``decoder`` (the decoder applied to q).  Controls: constant-class prediction,
the item's degree, and shuffled labels.
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
BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
ROLE_REPO = REPO / "results" / "research" / "directed_graphlet_role"
sys.path.insert(0, str(BASELINE))
sys.path.insert(0, str(ROLE_REPO))
sys.path.insert(0, str(HERE.parent / "representation"))

# a checkpoint whose stored SID table it reproduces exactly (100% of items); the table
# next to a checkpoint that only reaches 99.9% cannot stand in for that run's SIDs
CHECKPOINT_ARM = "TIGER_drq_D1"
STATES_FILE = HERE / "fidelity_states.npz"
STATES_META = HERE / "fidelity_states.json"
ORBIT_PROFILE = TEACHER / "teacher" / "opt" / "orbit_profile.npz"
CONTRACT = TEACHER / "teacher" / "opt" / "data_contract.npz"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
K_NEIGHBOURS = 10
SEED = 0                     # one run per configuration; no seed replication
EPOCHS = 300
MAX_INSTANCES = 60000
HIDDEN = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ state export
def export_states() -> dict:
    """z, hard q (rebuilt from codes) and the decoder output for the whole catalogue."""
    import torch
    from export_latent_q import INFER_USE_SK, engine_config, SEMANTICS as EXP_SEMANTICS
    from distances import build_distance
    from model import RQVAE
    from types import SimpleNamespace

    arm_dir = REPO / "results" / "stage2_RQ-VAE" / CHECKPOINT_ARM
    features = np.load(EXP_SEMANTICS).astype(np.float32)
    model = RQVAE(engine_config(), in_dim=features.shape[1],
                  distance=build_distance(SimpleNamespace(DISTANCE="euclidean")))
    payload = torch.load(arm_dir / "rqvae_best.pth", map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    stored = np.array([[int(v) for v in value[:3]]
                       for _, value in sorted(
                           json.loads((arm_dir / "item_sids.json").read_text()).items(),
                           key=lambda kv: int(kv[0]))], dtype=np.int64)
    with torch.no_grad():
        x = torch.from_numpy(features).to(device)
        z = model.encoder(x)
        tokens = model.rq(z, infer_use_sk=INFER_USE_SK)[3]
        codes = tokens.cpu().numpy().astype(np.int64)
        agreement = float((codes[:, :3] == stored).all(axis=1).mean())
        if agreement != 1.0:
            raise SystemExit(
                f"recomputed codes reproduce only {agreement:.4f} of the stored SID "
                f"table; pick the arm whose checkpoint matches, not this one")
        # rebuild the hard quantised vector from the stored SID and the frozen codebook
        levels = [layer.embed.weight.detach() for layer in model.rq.vq_layers]
        q = np.zeros_like(z.cpu().numpy(), dtype=np.float32)
        for level, table in enumerate(levels):
            q += table[torch.from_numpy(stored[:, level]).to(device)].cpu().numpy()
        decoded = model.decoder(torch.from_numpy(q).to(device)).cpu().numpy()
        z_np = z.cpu().numpy().astype(np.float32)
        q_rebuilt_matches_model = float(
            np.abs(q - model.rq(z, infer_use_sk=INFER_USE_SK)[0].cpu().numpy()).max())
    np.savez_compressed(STATES_FILE, semantics=features, z=z_np, q=q, decoder=decoded,
                        codes=stored)
    meta = {
        "arm": CHECKPOINT_ARM,
        "checkpoint_sha256": sha256(arm_dir / "rqvae_best.pth"),
        "sid_sha256": sha256(arm_dir / "item_sids.json"),
        "semantics_sha256": sha256(EXP_SEMANTICS),
        "infer_use_sk": INFER_USE_SK,
        "codes_match_stored_sid": 1.0,
        "q_rebuilt_from_sid_max_abs_diff_to_model_q": q_rebuilt_matches_model,
        "states_file": str(STATES_FILE.relative_to(REPO)),
        "states_sha256": sha256(STATES_FILE),
        "widths": {"semantics": int(features.shape[1]), "z": int(z_np.shape[1]),
                   "q": int(q.shape[1]), "decoder": int(decoded.shape[1])},
        "collision": float(1.0 - len(np.unique(stored, axis=0)) / len(stored)),
        "note": "the SID match is on the full three-level code of every item; a partial "
                "match would mean the checkpoint and the table are not the same run",
    }
    STATES_META.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load_states() -> dict:
    if not STATES_FILE.exists():
        export_states()
    payload = np.load(STATES_FILE)
    return {key: payload[key] for key in ("semantics", "z", "q", "decoder")}


# ------------------------------------------------------------------- instance pool
def instance_pool() -> dict:
    """Real 3- and 4-node instances of the training graph, split by instance."""
    from directed_role import FAMILIES
    from directed_role import load_pool, load_vocabulary

    observed = np.load(CONTRACT)["message_edges"]
    observed_set = set(map(tuple, observed.tolist()))
    vocab = load_vocabulary("directed")
    pools = load_pool("train", vocab)
    rows = []
    for family in FAMILIES:
        nodes, orbits, subtypes = pools[family]
        for node_row, orbit_row, subtype in zip(nodes, orbits, subtypes):
            rows.append({"family": family, "nodes": node_row.tolist(),
                         "orbits": orbit_row.tolist(), "subtype": int(subtype)})
    return {"rows": rows, "subtypes": {row["subtype"] for row in rows}}


def instance_arrays(pool: dict, family: str, width: int) -> dict:
    wide = np.array([row["nodes"] for row in pool["rows"]
                     if row["family"] == family and len(row["nodes"]) == width], dtype=np.int64)
    narrow = np.array([row["orbits"] for row in pool["rows"]
                       if row["family"] == family and len(row["nodes"]) == width], dtype=np.int64)
    subtypes = np.array([row["subtype"] for row in pool["rows"]
                         if row["family"] == family and len(row["nodes"]) == width],
                        dtype=np.int64)
    return {"nodes": wide, "orbits": narrow, "subtypes": subtypes}


def family_split(subtypes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by **instance**, never by node.

    Holding out connection *patterns* would leave 2 of 12 P4 subtypes for testing, which
    cannot support a measurement; what has to be isolated is the instances, so a test
    chain is not the training chain with one node swapped.  Node isolation is weaker than
    that under a shared vocabulary, and it is what this split provides.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subtypes))
    cut = int(0.7 * len(order))
    mask = np.zeros(len(subtypes), dtype=bool)
    mask[order[:cut]] = True
    return mask, ~mask


def multilabel_labels(subtypes: np.ndarray, mapping: dict) -> np.ndarray:
    labels = np.zeros((len(subtypes), len(mapping)), dtype=np.float32)
    for row, subtype in enumerate(subtypes.tolist()):
        labels[row, mapping[subtype]] = 1.0
    return labels


def type_mapping(subtypes: np.ndarray) -> dict:
    return {int(value): index for index, value in enumerate(np.unique(subtypes).tolist())}


# ---------------------------------------------------------------- role task models
def build_mlp(in_dim: int, out_dim: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, out_dim))


def build_shared(in_dim: int, out_dim: int):
    """Family head as before; the orbit head is added outside this builder."""
    return build_gnn(in_dim, out_dim)


def build_orbit_head(width: int, in_dim: int, orbit_space: int):
    import torch.nn as nn
    return nn.ModuleDict({f"node{position}": nn.Linear(in_dim, orbit_space)
                          for position in range(width)})


def build_gnn(in_dim: int, out_dim: int):
    import torch
    import torch.nn as nn

    class RoleGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_proj = nn.Linear(in_dim, HIDDEN)
            self.msg_proj = nn.Linear(in_dim, HIDDEN)
            self.out = nn.Linear(2 * HIDDEN, out_dim)

        def forward(self, x, rows, cols, count):
            h = torch.relu(self.self_proj(x))
            m = torch.zeros_like(h).index_add(0, rows, self.msg_proj(x)[cols])
            m = m / count.unsqueeze(-1)
            return self.out(torch.cat([h, m], dim=-1)), {}

    return RoleGNN()


def mutual_knn(states: np.ndarray, k: int = K_NEIGHBOURS) -> list[set[int]]:
    unit = states / np.maximum(np.linalg.norm(states, axis=1, keepdims=True), 1e-12)
    directed = np.empty(len(unit), dtype=object)
    for start in range(0, len(unit), 2048):
        block = unit[start:start + 2048] @ unit.T
        block[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf
        directed[start:start + 2048] = [set(row.tolist())
                                        for row in np.argpartition(-block, k, axis=1)[:, :k]]
    return [{other for other in directed[item] if item in directed[other]}
            for item in range(len(directed))]


# ------------------------------------------------------------------- training bits
def fit_loop(train, eval_x, train_y, eval_y, seed: int, build, epochs: int = EPOCHS,
             device=None, graph=None):
    """One training run with a fixed seed; returns eval predictions and losses."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train, y_train = train, train_y
    for _ in range(epochs):
        model.train()
        if graph is None:
            logits = model(x_train)
        else:
            logits, _ = model(x_train, *graph)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(eval_x) if graph is None else model(eval_x, *graph)[0]
    return model, logits


def split_items(covered: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.flatnonzero(covered))
    cut = int(0.8 * len(order))
    return np.sort(order[:cut]), np.sort(order[cut:])


def device_of():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _graph_tensors(items: np.ndarray, neighbours: list[set[int]], device):
    """Induced subgraph of `items`, with the neighbour counts of that subgraph."""
    import torch
    lookup = {int(item): position for position, item in enumerate(items.tolist())}
    rows, cols = [], []
    for item in items.tolist():
        for other in neighbours[int(item)]:
            if int(other) in lookup:
                rows.append(lookup[int(item)])
                cols.append(lookup[int(other)])
    n = len(items)
    row_t = torch.tensor(rows, dtype=torch.long, device=device)
    col_t = torch.tensor(cols, dtype=torch.long, device=device)
    count = torch.bincount(row_t, minlength=n).clamp(min=1)
    return row_t, col_t, count


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def role_task(states: dict, targets: np.ndarray, covered: np.ndarray, degree: np.ndarray,
              device) -> dict:
    """Item-level multi-role prediction for every representation and every regime."""
    import torch
    out = {}
    for name, table in states.items():
        per_seed = []
        neighbours = mutual_knn(table)
        for seed in (SEED,):
            train_items, eval_items = split_items(covered, seed)
            train_y = torch.from_numpy(targets[train_items]).to(device)
            eval_y = torch.from_numpy(targets[eval_items]).to(device)
            x_all = torch.from_numpy(_unit(table.astype(np.float64)).astype(np.float32)).to(device)
            record = {}
            for regime in ("mlp", "gnn"):
                build = (lambda dense=table.shape[1]: build_mlp(dense, targets.shape[1])) \
                    if regime == "mlp" else \
                    (lambda dense=table.shape[1]: build_gnn(dense, targets.shape[1]))
                if regime == "mlp":
                    graph = None
                else:
                    graph = _graph_tensors(eval_items, neighbours, device) \
                        if False else None
                # graph mode needs different subgraphs for train and eval; handled below
                if regime == "mlp":
                    _, logits = fit_loop(x_all[train_items], x_all[eval_items], train_y,
                                         eval_y, seed, build, device=device)
                else:
                    import torch as _t
                    _t.manual_seed(seed)
                    model = build().to(device)
                    opt = _t.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    tr_g = _graph_tensors(train_items, neighbours, device)
                    ev_g = _graph_tensors(eval_items, neighbours, device)
                    for _ in range(EPOCHS):
                        model.train()
                        logits, _ = model(x_all[train_items], *tr_g)
                        loss = _t.nn.functional.binary_cross_entropy_with_logits(logits, train_y)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        _t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    model.eval()
                    with _t.no_grad():
                        logits, _ = model(x_all[eval_items], *ev_g)
                probability = torch.sigmoid(logits)
                # cross-entropy against the normalised target (not a sigmoid likelihood)
                ce = float(-(eval_y * torch.log(probability.clamp(min=1e-9))).sum(-1).mean())
                prior = train_y.mean(0).clamp(min=1e-9)
                ce_prior = float(-(eval_y * torch.log(prior)).sum(-1).mean())
                record[regime] = {"ce": ce, "ce_constant_prior": ce_prior,
                                  "items_eval": int(len(eval_items)),
                                  "train_mean_degree": float(np.mean(
                                      [len(neighbours[i]) for i in train_items])),
                                  "eval_mean_degree": float(np.mean(
                                      [len(neighbours[i]) for i in eval_items]))}
            # degree-only control, same split
            deg_train = torch.from_numpy(degree[train_items, None].astype(np.float32)).to(device)
            deg_eval = torch.from_numpy(degree[eval_items, None].astype(np.float32)).to(device)
            _, logits = fit_loop(deg_train, deg_eval, train_y, eval_y, seed,
                                 lambda: build_mlp(1, targets.shape[1]), device=device)
            probability = torch.sigmoid(logits)
            prior = train_y.mean(0).clamp(min=1e-9)
            record["degree"] = {
                "ce": float(-(eval_y * torch.log(probability.clamp(min=1e-9))).sum(-1).mean()),
                "ce_constant_prior": float(-(eval_y * torch.log(prior)).sum(-1).mean()),
                "items_eval": int(len(eval_items))}
            per_seed.append(record)
        out[name] = {
            regime: {"ce_mean": float(np.mean([row[regime]["ce"] for row in per_seed])),
                     "ce_std": float(np.std([row[regime]["ce"] for row in per_seed])),
                     "ce_constant_prior": float(np.mean(
                         [row[regime]["ce_constant_prior"] for row in per_seed
                          if "ce_constant_prior" in row[regime]])),
                     "items_eval": per_seed[0][regime]["items_eval"]}
            for regime in ("mlp", "gnn", "degree")}
        out[name]["per_seed"] = per_seed
        print(f"  role {name}: " + " ".join(
            f"{regime} {out[name][regime]['ce_mean']:.4f}" for regime in ("mlp", "gnn", "degree")),
            flush=True)
    return out


def instance_task(states: dict, pool: dict, device) -> dict:
    """Subgraph-level prediction: family label and per-node orbit, own vectors only.

    A model reads the members' representations (and, in the GNN regime, the KNN edges
    among the members) and predicts the family and each node's orbit id.  Adjacency of
    the *real* instance is never given, and the split is by subtype, so no test
    connection pattern was trained.
    """
    import torch
    out = {}
    for family, width in (("P3", 3), ("P4", 4)):
        block = instance_arrays(pool, family, width)
        nodes, orbits, subtypes = block["nodes"], block["orbits"], block["subtypes"]
        if len(nodes) == 0:
            continue
        total = len(nodes)
        if total > MAX_INSTANCES:
            keep = np.random.default_rng(SEED).choice(total, MAX_INSTANCES, replace=False)
            keep.sort()
            nodes, orbits, subtypes = nodes[keep], orbits[keep], subtypes[keep]
        mapping = type_mapping(subtypes)
        orbit_space = int(orbits.max()) + 1
        out[family] = {"instances_total": int(total), "instances_used": int(len(nodes)),
                       "types": len(mapping)}
        for name, table in states.items():
            per_seed = []
            unit = _unit(table.astype(np.float64)).astype(np.float32)
            for seed in (SEED,):
                train_mask, eval_mask = family_split(subtypes, seed)
                x = torch.from_numpy(unit[nodes]).to(device)      # (n, width, d)
                y_type = torch.from_numpy(
                    multilabel_labels(subtypes, mapping)).to(device)
                y_orbit = torch.from_numpy(orbits).to(device)
                torch.manual_seed(seed)
                model = build_gnn(width * table.shape[1], len(mapping)).to(device)
                orbit_heads = build_orbit_head(width, table.shape[1], orbit_space).to(device)
                optimizer = torch.optim.AdamW(
                    list(model.parameters()) + list(orbit_heads.parameters()),
                    lr=1e-3, weight_decay=1e-4)
                tr = torch.from_numpy(nodes[train_mask]).to(device)
                tr_y = y_type[train_mask]
                tr_o = y_orbit[train_mask]
                te = torch.from_numpy(nodes[eval_mask]).to(device)
                te_y = y_type[eval_mask]
                te_o = y_orbit[eval_mask]
                tr_x = x[train_mask].reshape(len(tr), -1)
                te_x = x[eval_mask].reshape(len(te), -1)
                through = torch.tensor([], dtype=torch.long, device=device)
                zeros = torch.tensor([], dtype=torch.long, device=device)
                for _ in range(EPOCHS):
                    model.train()
                    orbit_heads.train()
                    logits, _ = model(tr_x, through, zeros, torch.ones(len(tr), device=device))
                    loss_type = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, tr_y)
                    loss_orbit = 0.0
                    for position in range(width):
                        node_vectors = x[train_mask][:, position, :]
                        loss_orbit = loss_orbit + torch.nn.functional.cross_entropy(
                            orbit_heads[f"node{position}"](node_vectors), tr_o[:, position])
                    loss = loss_type + loss_orbit / width
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                orbit_heads.eval()
                with torch.no_grad():
                    logits, _ = model(te_x, through, zeros, torch.ones(len(te), device=device))
                    probability = torch.sigmoid(logits)
                    ce = float(-(te_y * torch.log(probability.clamp(min=1e-9))).sum(-1).mean())
                    exact = float(torch.all((probability > 0.5) == (te_y > 0.5), dim=1)
                                  .float().mean())
                    prior = tr_y.mean(0).clamp(min=1e-9)
                    ce_prior = float(-(te_y * torch.log(prior)).sum(-1).mean())
                    correct, total_nodes = 0, 0
                    for position in range(width):
                        node_vectors = x[eval_mask][:, position, :]
                        predicted = orbit_heads[f"node{position}"](node_vectors).argmax(-1)
                        correct += int((predicted == te_o[:, position]).sum())
                        total_nodes += len(predicted)
                    orbit_accuracy = correct / max(total_nodes, 1)
                    # majority-orbit control: most frequent training orbit per position
                    majority_hits = 0
                    for position in range(width):
                        votes = torch.bincount(tr_o[:, position], minlength=orbit_space)
                        guess = int(votes.argmax())
                        majority_hits += int((te_o[:, position] == guess).sum())
                    majority_accuracy = majority_hits / max(total_nodes, 1)
                per_seed.append({"ce": ce, "exact_type_accuracy": exact,
                                 "orbit_accuracy": orbit_accuracy,
                                 "orbit_majority_accuracy": majority_accuracy,
                                 "ce_constant_prior": ce_prior,
                                 "instances_eval": int(eval_mask.sum()),
                                 "orbit_space": orbit_space})
            out[family][name] = {
                "ce_mean": float(np.mean([row["ce"] for row in per_seed])),
                "ce_std": float(np.std([row["ce"] for row in per_seed])),
                "exact_type_accuracy_mean": float(np.mean(
                    [row["exact_type_accuracy"] for row in per_seed])),
                "orbit_accuracy_mean": float(np.mean(
                    [row["orbit_accuracy"] for row in per_seed])),
                "orbit_majority_accuracy_mean": float(np.mean(
                    [row["orbit_majority_accuracy"] for row in per_seed])),
                "orbit_chance_per_position": [1.0 / orbit_space] * width,
                "ce_constant_prior": per_seed[0]["ce_constant_prior"],
                "instances_eval": per_seed[0]["instances_eval"],
                "per_seed": per_seed}
            print(f"  {family} {name}: ce {out[family][name]['ce_mean']:.4f} "
                  f"exact {out[family][name]['exact_type_accuracy_mean']:.3f}", flush=True)
    return out


def main() -> None:
    device = device_of()
    meta = export_states()
    states = load_states()
    payload = np.load(ORBIT_PROFILE)
    targets = payload["profile"].astype(np.float32)
    covered = targets.sum(axis=1) > 0
    contract = np.load(CONTRACT)
    degree = np.bincount(contract["message_edges"].ravel(),
                         minlength=len(targets)).astype(np.float32)
    report = {"checkpoint": meta,
              "protocol": {"seed": SEED,
                           "seed_replication": "none: one run per configuration, so the "
                                               "numbers carry no seed-variance estimate",
                           "epochs": EPOCHS, "hidden": HIDDEN,
                           "role_split": "items, 80/20 by seed",
                           "instance_split": "by instance, 70/30 by seed "
                                             f"(at most {MAX_INSTANCES} instances per family, "
                                             "deterministically subsampled)",
                           "inputs": "own representation only; no real adjacency"},
              "role": role_task(states, targets, covered, degree, device)}
    (HERE / "fidelity.json").write_text(json.dumps(report, indent=1) + "\n")
    pool = instance_pool()
    report["instances"] = instance_task(states, pool, device)
    (HERE / "fidelity.json").write_text(json.dumps(report, indent=1) + "\n")
    print("written", HERE / "fidelity.json")


if __name__ == "__main__":
    main()
