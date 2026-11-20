"""Issue #228: fix the Teacher's optimisation accounting, then ask why it learns nothing.

The #227 harness called `optimizer.step()` once per "epoch", after 24 accumulated
microbatches, so its advertised 40 epochs x 24 batches were **40 real updates**.
This file defines a step the only way that is not ambiguous:

    one step = zero_grad -> (forward, backward) x accum -> clip -> optimizer.step()

and counts epochs, microbatches and steps separately.  The historical 40-step
architecture is kept as a regression control (same graph, same heads, same pairs,
40 steps); the lingering question is whether that control was starved of updates.

Gates, in order, each writing its evidence before the next one starts:

  0. data contract: split SHA, no target pair in the message graph (either
     direction), negatives are not edges (either direction), sample pass rates;
  1. overfit: 1,024 fixed correct positives and matched negatives, train AUC
     target >= 0.95 -- an architecture that cannot fit 1,024 pairs stops the run
     with gradient/score diagnostics instead of a scientific verdict;
  2. controlled ablation at an equal real-step budget:
       A  semantic projection only          (768 -> 128)
       B  the #227 two-layer GNN            (shared in/out parameters, control)
       C  residual GNN, per-layer self/in/out, LayerNorm, semantic skip,
          asymmetric direction head so score(i->j) != score(j->i) is expressible
       D  C's architecture on a strict degree-preserving rewire
       E  C's architecture with no messages (same parameters, no topology)
     Edge-only first, then Edge+Direction: a variant that wins only when both
     losses are on cannot be attributed, so the two tasks never share a run;
  3. graphlet recognition (with adjacency, separate question) and a lightweight
     high-order prediction diagnostic (CN / AA / triangles / length-2,3 paths on
     the TRAIN-observable graph only), plus a frozen-state orbit probe on C vs D.

Nothing here distils into the RQ-VAE, runs Stage2 or Stage3, or overwrites
`graph_teacher2_corrected_report.json`; the #227 stop decision stays in force
until a gate below produces evidence strong enough to reopen it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(ROLE := REPO / "results" / "research" / "directed_graphlet_role"))
sys.path.insert(0, str(HERE))

from relation_audit import auc  # noqa: E402
import graph_teacher2 as gt2  # noqa: E402

# --------------------------------------------------------------------- config

SEED = 228
MICRO = 4096
ACCUM = 3
MAX_STEPS = 1200
EVAL_EVERY = 25
PATIENCE = 200
LR_GRID = (3e-4, 1e-3, 3e-3)
WD_GRID = (1e-4, 1e-2)
SEEDS = (226, 227)
HIDDEN = 256
NODE_DIM = 128
OVERFIT_PAIRS = 1024
OVERFIT_STEPS = 600
OVERFIT_TARGET = 0.95
CONTROL_STEPS = 40
TASKS = {"edge": ("edge",), "edge+direction": ("edge", "direction")}
OUT = HERE / "teacher" / "opt"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- counters

class StepCounter:
    """Separates the three counts the #226 harness conflated."""

    def __init__(self):
        self.epochs = 0.0
        self.microbatches = 0
        self.steps = 0

    def microbatch(self):
        self.microbatches += 1

    def step(self):
        self.steps += 1

    def as_dict(self):
        return {"epochs": float(self.epochs), "microbatches": int(self.microbatches),
                "steps": int(self.steps)}


class PairStream:
    """Deterministic microbatch index stream; two streams with one seed agree."""

    def __init__(self, n: int, seed: int, micro: int):
        self.n, self.micro = int(n), int(micro)
        self.rng = np.random.default_rng(seed)
        self.consumed = 0

    def next(self) -> np.ndarray:
        pick = self.rng.integers(0, self.n, self.micro)
        self.consumed += self.micro
        return pick

    @property
    def epochs(self) -> float:
        return self.consumed / max(self.n, 1)


def param_sha(parameters) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().numpy().astype(np.float32).tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ encoders

class OriginalGraphEncoder(nn.Module):
    """The #227 control: two passes that share one set of self/in/out parameters."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.self_proj = nn.Linear(hidden, hidden)
        self.in_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, features, adjacency):
        h = self.input(features)
        for _ in range(2):
            degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
            out_message = torch.sparse.mm(adjacency, h) / degree
            degree_in = torch.sparse.sum(adjacency, dim=0).to_dense().clamp(min=1.0).unsqueeze(-1)
            in_message = torch.sparse.mm(adjacency.transpose(0, 1), h) / degree_in
            h = F.gelu(self.self_proj(h) + self.out_proj(out_message) + self.in_proj(in_message))
        return self.output(h)


class ResidualLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_proj = nn.Linear(hidden, hidden)
        self.in_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.skip_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, adjacency, skip):
        if adjacency is None:
            message = 0.0
        else:
            degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
            out_message = self.out_proj(torch.sparse.mm(adjacency, h) / degree)
            degree_in = torch.sparse.sum(adjacency, dim=0).to_dense().clamp(min=1.0).unsqueeze(-1)
            in_message = self.in_proj(torch.sparse.mm(adjacency.transpose(0, 1), h) / degree_in)
            message = out_message + in_message
        return self.norm(h + self.self_proj(h) + message + self.skip_proj(skip))


class ResidualGraphEncoder(nn.Module):
    """Per-layer parameters, LayerNorm and a semantic skip kept in every layer."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(ResidualLayer(hidden) for _ in range(layers))
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, features, adjacency):
        skip = self.input(features)
        h = skip
        for block in self.blocks:
            h = block(h, adjacency, skip)
        return self.output(h)


class EdgeHead(nn.Module):
    """Consumes the concatenated (a, b, a*b) pair representation."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, pair):
        return self.trunk(pair).squeeze(-1)


class AsymmetricDirectionHead(nn.Module):
    """Separate source and destination projections: swapping the order changes the score."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.src = nn.Linear(dim, hidden)
        self.dst = nn.Linear(dim, hidden)
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, state, rows: torch.Tensor, cols: torch.Tensor):
        a, b = self.src(state[rows]), self.dst(state[cols])
        return self.trunk(torch.cat([a, b, a * b], dim=-1))


class TeacherVariant(nn.Module):
    def __init__(self, name: str, in_dim: int = 768, hidden: int = HIDDEN,
                 node_dim: int = NODE_DIM):
        super().__init__()
        self.name = name
        if name == "A_embedding":
            self.encoder = nn.Linear(in_dim, node_dim)
        elif name == "B_graph":
            self.encoder = OriginalGraphEncoder(in_dim, hidden, node_dim)
        elif name in ("C_residual", "D_shuffled", "E_nograph"):
            self.encoder = ResidualGraphEncoder(in_dim, hidden, node_dim)
        else:
            raise ValueError(f"unknown variant {name}")
        self.edge_head = EdgeHead(node_dim, hidden)
        self.direction_head = AsymmetricDirectionHead(node_dim, hidden)

    def encode(self, features, adjacency):
        if self.name == "A_embedding":
            return self.encoder(features)
        if self.name == "E_nograph":
            return self.encoder(features, None)
        if adjacency is None:
            raise ValueError(f"{self.name} needs the message graph")
        return self.encoder(features, adjacency)

    @staticmethod
    def pair(state, left, right):
        a, b = state[left], state[right]
        return torch.cat([a, b, a * b], dim=-1)


def build_variant(name: str, in_dim: int = 768, hidden: int = HIDDEN,
                  node_dim: int = NODE_DIM) -> TeacherVariant:
    return TeacherVariant(name, in_dim, hidden, node_dim)


def save_checkpoint(path: Path, variant: TeacherVariant, meta: dict) -> dict:
    payload = {"state": variant.state_dict(), "meta": meta,
               "param_sha": param_sha_from_state(variant.state_dict())}
    torch.save(payload, path)
    return payload


def load_checkpoint(path: Path, variant: TeacherVariant) -> dict:
    payload = torch.load(path, map_location="cpu")
    assert param_sha_from_state(payload["state"]) == payload["param_sha"], \
        f"{path} changed since it was written"
    variant.load_state_dict(payload["state"])
    return variant.state_dict()


def param_sha_from_state(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(state[key].detach().cpu().numpy().astype(np.float32).tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------- train a step

def run_steps(variant: TeacherVariant, features, adjacency, positives, negatives,
              direction, role_target, *, steps: int, accum: int, micro: int,
              lr: float, weight_decay: float, device, counter: StepCounter | None = None,
              loss_terms=("edge",), batch_seed: int = SEED, eval_fn=None,
              patience: int | None = None, eval_every: int | None = None,
              checkpoint_path: Path | None = None, checkpoint_every: int | None = None,
              grad_clip: float = 5.0, verbose: bool = False) -> dict:
    """One step is one real update; `accum` microbatches build exactly one of them."""
    counter = counter or StepCounter()
    optimizer = torch.optim.AdamW([p for p in variant.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    parameters = [p for p in variant.parameters() if p.requires_grad]
    sha_before = param_sha(parameters)
    stream = PairStream(len(positives), batch_seed, micro)
    dir_stream = PairStream(len(direction["rows"]), batch_seed + 1, micro) \
        if direction is not None and "direction" in loss_terms else None
    direction_tensors = None
    if dir_stream is not None:
        direction_tensors = (torch.from_numpy(direction["rows"]).to(device),
                             torch.from_numpy(direction["cols"]).to(device),
                             torch.from_numpy(direction["labels"]).to(device))
    base = torch.from_numpy(positives).to(device)
    negative = torch.from_numpy(negatives).to(device)
    trace, history = [], []
    best = {"metric": -float("inf"), "step": 0, "state": None}
    stale = 0
    started = time.time()
    step = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        total_loss = None
        state = variant.encode(features, adjacency)
        for _ in range(accum):
            pick = stream.next()
            chosen = torch.from_numpy(pick).to(device)
            positive = base[chosen]
            left = torch.cat([positive[:, 0], positive[:, 0]])
            right = torch.cat([positive[:, 1], negative[chosen]])
            labels = torch.cat([torch.ones(len(pick)), torch.zeros(len(pick))]).to(device)
            logits = variant.edge_head(variant.pair(state, left, right))
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if dir_stream is not None:
                rows, cols, targets = direction_tensors
                select = torch.from_numpy(dir_stream.next()).to(device)
                d_logits = variant.direction_head(state, rows[select], cols[select])
                loss = loss + F.cross_entropy(d_logits, targets[select])
            total_loss = loss if total_loss is None else total_loss + loss
            accumulated += float(loss.detach())
            counter.microbatch()
        (total_loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
        counter.step()
        counter.epochs = stream.epochs
        trace.append(accumulated / accum)
        if eval_fn is not None and eval_every and step % eval_every == 0:
            metric = float(eval_fn(variant, step))
            history.append({"step": step, "metric": metric})
            if metric > best["metric"] + 1e-6:
                best = {"metric": metric, "step": step,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in variant.state_dict().items()}}
                stale = 0
            else:
                stale += eval_every
            if verbose:
                print(f"  step {step} loss {trace[-1]:.4f} valid {metric:.4f} "
                      f"best {best['metric']:.4f}", flush=True)
            if patience is not None and stale >= patience:
                break
        if checkpoint_path is not None and checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(checkpoint_path.with_name(
                f"{checkpoint_path.stem}_step{step}{checkpoint_path.suffix}"), variant,
                {"step": step, "loss": trace[-1]})
    if best["state"] is not None:
        variant.load_state_dict(best["state"])
    return {"steps": counter.steps, "microbatches": counter.microbatches,
            "epochs": counter.epochs, "loss_trace": trace, "history": history,
            "param_sha_before": sha_before,
            "param_sha_after": param_sha(parameters),
            "best_metric": best["metric"], "best_step": best["step"],
            "seconds": time.time() - started, "final_loss": trace[-1] if trace else None,
            "stopped_early": bool(patience is not None and stale >= patience)}


# -------------------------------------------------------------- evaluation

def pair_scores(variant: TeacherVariant, features, adjacency, positive: np.ndarray,
                negative: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        state = variant.encode(features, adjacency)
        left = torch.from_numpy(positive[:, 0]).to(device)
        right = torch.from_numpy(positive[:, 1]).to(device)
        target = torch.from_numpy(negative).to(device)
        positive_score = variant.edge_head(variant.pair(state, left, right)).cpu().numpy()
        negative_score = variant.edge_head(variant.pair(state, left, target)).cpu().numpy()
    return np.stack([positive_score, negative_score], axis=1)


def auc_of(scores: np.ndarray) -> float:
    labels = np.concatenate([np.ones(scores.shape[0]), np.zeros(scores.shape[0])])
    return float(auc(np.concatenate([scores[:, 0], scores[:, 1]]), labels))


def evaluate_splits(variant, features, adjacency, data, device) -> dict:
    result = {}
    for split in ("train", "valid", "test"):
        scores = pair_scores(variant, features, adjacency,
                             data[f"{split}_positive"], data[f"{split}_negative"], device)
        labels = np.concatenate([np.ones(scores.shape[0]), np.zeros(scores.shape[0])])
        flat = np.concatenate([scores[:, 0], scores[:, 1]])
        result[split] = {"auc": float(auc(flat, labels)),
                         "ap": gt2.average_precision(flat, labels),
                         "pairs": int(scores.shape[0]),
                         "positive_rate": float(labels.mean())}
        result[split]["scores"] = scores
    return result


def direction_scores(variant, features, adjacency, direction, device) -> dict:
    with torch.no_grad():
        state = variant.encode(features, adjacency)
        logits = variant.direction_head(
            state, torch.from_numpy(direction["rows"]).to(device),
            torch.from_numpy(direction["cols"]).to(device))
        prediction = logits.argmax(-1).cpu().numpy()
    target = direction["labels"]
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, predicted in zip(target, prediction):
        confusion[truth, predicted] += 1
    metrics = gt2.direction_metrics(confusion)
    metrics["prediction_counts"] = confusion.sum(0).tolist()
    metrics["target_counts"] = confusion.sum(1).tolist()
    return metrics


# ------------------------------------------------------------ data contract

def prepare_data(device, load_features: bool = True, use_cache: bool = True) -> dict:
    cache = OUT / "data_contract.npz"
    meta_path = OUT / "data_contract.json"
    from directed_graphlets import load_adjacency
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    graph_sha = hashlib.sha256(
        np.stack([rows, cols]).astype(np.int64).tobytes()).hexdigest()
    if use_cache and cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("graph_sha") == graph_sha and meta.get("seed") == gt2.SEED:
            stored = np.load(cache)
            data = {key: stored[key] for key in stored.files
                    if not key.startswith("dir_")}
            for split in ("train", "valid", "test"):
                data[f"{split}_direction"] = {
                    "rows": stored[f"dir_{split}_rows"],
                    "cols": stored[f"dir_{split}_cols"],
                    "labels": stored[f"dir_{split}_labels"]}
            data["full_directed"] = set(map(tuple, data.pop("full_directed").tolist()))
            data["meta"] = meta
            data["negative_success"] = meta["negative_success"]
            data["split_sha"] = meta["split_sha"]
        else:
            data = _build_contract(graph, rows, cols, n_items, graph_sha)
    else:
        data = _build_contract(graph, rows, cols, n_items, graph_sha)
    if load_features:
        features = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
        data["features"] = torch.from_numpy(features.astype(np.float32)).to(device)
    else:
        data["features"] = None
    data["adjacency"] = torch.sparse_coo_tensor(
        torch.from_numpy(data["message_edges"].T.astype(np.int64)),
        torch.ones(len(data["message_edges"]), dtype=torch.float32),
        size=(n_items, n_items)).coalesce().to(device)
    data["n_items"] = n_items
    return data


def _build_contract(graph, rows, cols, n_items, graph_sha) -> dict:
    payload = gt2.split_pairs(rows, cols)
    materialised = gt2.materialise(payload, rows, cols)
    message = np.array(materialised["observed_directed"], dtype=np.int64)
    degrees = (np.bincount(message[:, 0], minlength=n_items)
               + np.bincount(message[:, 1], minlength=n_items)).astype(np.float64)
    rng = np.random.default_rng(gt2.SEED + 7)
    data, success = {}, {}
    for split in ("masked", "valid", "test"):
        positives, negatives, failures = gt2.sample_negatives(
            payload[split], degrees, materialised["forbidden"], rng,
            directed=materialised["full_directed"])
        key = {"masked": "train", "valid": "valid", "test": "test"}[split]
        data[f"{key}_positive"] = positives
        data[f"{key}_negative"] = negatives
        success[key] = {"sampled": int(len(positives)),
                        "attempted": int(len(payload[split])),
                        "failures": int(len(failures)),
                        "rate": float(len(positives) / max(len(payload[split]), 1))}
    direction = {}
    for split, key in (("masked", "train"), ("valid", "valid"), ("test", "test")):
        example = gt2.direction_examples(payload[split], materialised["full_directed"], rng)
        direction[f"{key}_direction"] = example
        if len(example["rows"]) == 0:
            raise RuntimeError(f"direction task for {split} has no one-way edges")
    # direction targets must also be absent from the message graph
    for key in ("train", "valid", "test"):
        pairs = set(map(tuple, message.tolist()))
        d = direction[f"{key}_direction"]
        blocked = sum((int(a), int(b)) in pairs or (int(b), int(a)) in pairs
                      for a, b in zip(d["rows"], d["cols"]))
        if blocked:
            raise RuntimeError(f"{blocked} direction targets leaked into the message graph")
    data.update(direction)
    data["message_edges"] = message
    data["observed_pairs"] = payload["observed_pairs"]
    data["full_directed"] = materialised["full_directed"]
    data["split_sha"] = payload["split_sha"]
    data["negative_success"] = success
    data["masked_pairs"] = payload["masked"]
    data["valid_pairs"] = payload["valid"]
    data["test_pairs"] = payload["test"]
    meta = {"graph_sha": graph_sha, "seed": gt2.SEED, "split_sha": payload["split_sha"],
            "n_items": int(n_items), "message_edges": int(len(message)),
            "negative_success": success,
            "target_pairs": {"train": int(len(payload["masked"])),
                             "valid": int(len(payload["valid"])),
                             "test": int(len(payload["test"]))}}
    savable = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    for split in ("train", "valid", "test"):
        direction_entry = data[f"{split}_direction"]
        for field in ("rows", "cols", "labels"):
            savable[f"dir_{split}_{field}"] = direction_entry[field]
    savable["full_directed"] = np.array(sorted(materialised["full_directed"]), dtype=np.int64)
    np.savez_compressed(OUT / "data_contract.npz", **savable)
    (OUT / "data_contract.json").write_text(json.dumps(meta, indent=1))
    return data


# ------------------------------------------------------------- overfit gate

def overfit_gate(features, positives, negatives, *, device, variant: str = "B_graph",
                 max_steps: int = OVERFIT_STEPS, target_auc: float = OVERFIT_TARGET,
                 in_dim: int = 768, hidden: int = HIDDEN, node_dim: int = NODE_DIM,
                 adjacency=None) -> dict:
    """Fit a fixed 1,024-pair set; below target AUC the run stops here."""
    keep = min(OVERFIT_PAIRS, len(positives))
    positive = np.asarray(positives[:keep], dtype=np.int64)
    negative = np.asarray(negatives[:keep], dtype=np.int64)
    model = build_variant(variant, in_dim, hidden, node_dim).to(device)
    counter = StepCounter()
    result = run_steps(model, features, adjacency, positive, negative, None, None,
                       steps=max_steps, accum=1, micro=keep, lr=3e-3, weight_decay=0.0,
                       device=device, counter=counter, loss_terms=("edge",),
                       batch_seed=SEED + 5, patience=None)
    scores = pair_scores(model, features, adjacency, positive, negative, device)
    train_auc = auc_of(scores)
    gate = {"variant": variant, "pairs": int(keep), "steps": result["steps"],
            "microbatches": result["microbatches"], "train_auc": train_auc,
            "target": target_auc, "passed": bool(train_auc >= target_auc),
            "final_loss": result["final_loss"],
            "param_sha_before": result["param_sha_before"],
            "param_sha_after": result["param_sha_after"],
            "seconds": result["seconds"]}
    if not gate["passed"]:
        gate["diagnostics"] = _overfit_diagnostics(
            model, features, adjacency, positive, negative, device)
    return gate


def _overfit_diagnostics(model, features, adjacency, positive, negative, device) -> dict:
    scores = pair_scores(model, features, adjacency, positive, negative, device)
    gradients = {}
    model.zero_grad(set_to_none=True)
    state = model.encode(features, adjacency)
    left = torch.from_numpy(positive[:, 0]).to(device)
    right = torch.from_numpy(positive[:, 1]).to(device)
    labels = torch.ones(len(positive), device=device)
    loss = F.binary_cross_entropy_with_logits(
        model.edge_head(model.pair(state, left, right)), labels)
    loss.backward()
    for name, parameter in model.named_parameters():
        gradients[name] = float(parameter.grad.norm()) if parameter.grad is not None else 0.0
    with torch.no_grad():
        head_out = model.edge_head(model.pair(state, left, right))
        names = {"positive_pairs": positive, "negative_pairs": negative}
        sampler = {name: int(len(pairs)) for name, pairs in names.items()}
        degenerate = int(sum(int(a) in (int(b), int(d)) or int(b) == int(d)
                             for (a, b), d in zip(positive.tolist(), negative.tolist())))
    return {"score_positive_mean": float(scores[:, 0].mean()),
            "score_positive_std": float(scores[:, 0].std()),
            "score_negative_mean": float(scores[:, 1].mean()),
            "score_negative_std": float(scores[:, 1].std()),
            "label_counts": sampler, "degenerate_pairs": degenerate,
            "head_output_std": float(head_out.std()),
            "encoder_state_std": float(state.detach().std()),
            "grad_norms": gradients,
            "grad_zero_count": int(sum(v == 0.0 for v in gradients.values())),
            "grad_total": len(gradients),
            "all_parameters_moved": None}


# ------------------------------------------------------- degree-preserving rewire

def degree_preserving_rewire(index: np.ndarray, n: int, forbidden: set, seed: int,
                             attempts_factor: int = 8) -> tuple:
    """Double-edge swaps: in- and out-degree are preserved exactly by construction."""
    index = np.asarray(index, dtype=np.int64)
    if index.ndim != 2:
        raise ValueError("the edge index must be 2-D")
    if index.shape[0] == 2 and index.shape[1] != 2:
        index = index.T          # accept (2, E) and normalise to (E, 2)
    edges = index.tolist()
    n_edges = len(edges)
    edge_set = set(map(tuple, edges))
    rng = np.random.default_rng(seed)
    attempts = n_edges * attempts_factor
    swaps = blocked = forbidden_blocked = 0
    for _ in range(attempts):
        i, j = rng.integers(0, n_edges, 2) if n_edges > 1 else (0, 0)
        if i == j:
            blocked += 1
            continue
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        if u1 == v2 or u2 == v1:
            blocked += 1
            continue
        a, b = (u1, v2), (u2, v1)
        if a in edge_set or b in edge_set:
            blocked += 1
            continue
        if (a in forbidden or (a[1], a[0]) in forbidden
                or b in forbidden or (b[1], b[0]) in forbidden):
            forbidden_blocked += 1
            blocked += 1
            continue
        edge_set.discard((u1, v1))
        edge_set.discard((u2, v2))
        edge_set.add(a)
        edge_set.add(b)
        edges[i], edges[j] = a, b
        swaps += 1
    new_index = np.array(edges, dtype=np.int64)
    out_new = np.bincount(new_index[:, 0], minlength=n)
    in_new = np.bincount(new_index[:, 1], minlength=n)
    out_old = np.bincount(index[:, 0], minlength=n)
    in_old = np.bincount(index[:, 1], minlength=n)
    stats = {"edges_in": int(index.shape[0]), "edges_out": int(new_index.shape[0]),
             "attempts": int(attempts), "swaps": int(swaps),
             "blocked_attempts": int(blocked),
             "forbidden_blocked": int(forbidden_blocked),
             "out_degree_l1": float(np.abs(out_new - out_old).sum()),
             "in_degree_l1": float(np.abs(in_new - in_old).sum()),
             "self_loops": int(np.sum(new_index[:, 0] == new_index[:, 1])),
             "unique_edges": int(len(edge_set))}
    return new_index, stats


# ------------------------------------------------------------ high-order gate

def _logistic_fit_predict(train_x, train_y, test_x, epochs: int = 400, lr: float = 0.1):
    x = torch.from_numpy(np.asarray(train_x, dtype=np.float32))
    y = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    mean, std = x.mean(0), x.std(0).clamp(min=1e-6)
    x = (x - mean) / std
    test_x = (torch.from_numpy(np.asarray(test_x, dtype=np.float32)) - mean) / std
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return (test_x @ weight + bias).numpy()


def _graph_features(index: np.ndarray, n: int, pairs: np.ndarray,
                    degrees: np.ndarray) -> np.ndarray:
    """Train-observable features only: symmetric CN/AA, directed length-2 and -3 walks."""
    import scipy.sparse as sp
    index = np.asarray(index, dtype=np.int64)
    if index.shape[0] == 2 and index.shape[1] != 2:
        index = index.T          # normalise (2, E) to (E, 2)
    pair_rows = np.asarray(pairs[:, 0], dtype=np.int64)
    pair_cols = np.asarray(pairs[:, 1], dtype=np.int64)
    symmetric = sp.csr_matrix(
        (np.ones(2 * index.shape[1]),
         (np.concatenate([index[0], index[1]]), np.concatenate([index[1], index[0]]))),
        shape=(n, n))
    symmetric.data[:] = 1.0
    symmetric.sum_duplicates()
    degree = np.asarray(symmetric.sum(1)).ravel()
    inverse_log = 1.0 / np.log(np.clip(degree, 3.0, None))
    row_slice = symmetric[pair_rows]
    common = np.asarray(row_slice.multiply(symmetric[pair_cols]).sum(1)).ravel()
    adamic = np.asarray(row_slice.multiply(symmetric[pair_cols]) @ inverse_log).ravel()
    directed = sp.csr_matrix((np.ones(index.shape[1]), (index[0], index[1])), shape=(n, n))
    directed.sum_duplicates()
    two_step_matrix = directed @ directed
    two_step = np.asarray(two_step_matrix[pair_rows, pair_cols]).ravel()
    three = []
    for start in range(0, len(pairs), 8000):
        block = slice(start, start + 8000)
        three.append(np.asarray(two_step_matrix[pair_rows[block]].multiply(
            directed[pair_cols[block]]).sum(1)).ravel())
    three_step = np.concatenate(three) if three else np.zeros(len(pairs))
    return np.stack([
        np.log1p(degrees[pair_rows]), np.log1p(degrees[pair_cols]),
        np.log1p(common), np.log1p(np.maximum(adamic, 0.0)),
        np.log1p(two_step), np.log1p(three_step),
        (common > 0).astype(np.float64)], axis=1)


def high_order_gate(data, shuffled_index: np.ndarray) -> dict:
    pairs_train = data["train_positive"]
    pairs_test = data["test_positive"]
    positives_train = np.concatenate([np.ones(len(pairs_train)), np.zeros(len(pairs_train))])
    test_labels = np.concatenate([np.ones(len(pairs_test)), np.zeros(len(pairs_test))])
    # negative halves are (anchor, negative target), so the two halves stay aligned
    train_pairs = np.stack([np.concatenate([pairs_train[:, 0], pairs_train[:, 0]]),
                            np.concatenate([pairs_train[:, 1], data["train_negative"]])], 1)
    test_pairs = np.stack([np.concatenate([pairs_test[:, 0], pairs_test[:, 0]]),
                           np.concatenate([pairs_test[:, 1], data["test_negative"]])], 1)
    degrees = np.bincount(data["message_edges"].ravel(), minlength=data["n_items"]).astype(np.float64)
    real = _graph_features(data["message_edges"], data["n_items"], train_pairs, degrees)
    test_real = _graph_features(data["message_edges"], data["n_items"], test_pairs, degrees)
    shuffled = _graph_features(shuffled_index, data["n_items"], train_pairs, degrees)
    test_shuffled = _graph_features(shuffled_index, data["n_items"], test_pairs, degrees)
    blocks = {"degree": slice(0, 2), "graph": slice(2, 7), "all": slice(0, 7)}
    result = {}
    for name, block in blocks.items():
        logits = _logistic_fit_predict(real[:, block], positives_train, test_real[:, block])
        shuffled_logits = _logistic_fit_predict(
            shuffled[:, block], positives_train, test_shuffled[:, block])
        result[name] = {"auc": float(auc(logits, test_labels)),
                        "ap": gt2.average_precision(logits, test_labels),
                        "shuffled_auc": float(auc(shuffled_logits, test_labels)),
                        "shuffled_ap": gt2.average_precision(shuffled_logits, test_labels)}
    result["pairs"] = int(len(test_labels))
    return result


# --------------------------------------------------------------- orbit probe

def cached_orbit_profile(data, max_instances: int = 8000) -> tuple:
    """The orbit profile is deterministic; recomputing it costs minutes of Python."""
    cache = OUT / "orbit_profile.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if str(stored["graph_sha"].item()) == data["meta"]["graph_sha"]:
            return stored["profile"], json.loads(str(stored["coverage"].item()))
    from directed_role import FAMILIES, load_pool, load_vocabulary
    vocab = load_vocabulary("directed")
    pools = load_pool("train", vocab, max_instances=max_instances)
    instances = [(family, pools[family][0], vocab) for family in FAMILIES]
    # `in` on a numpy array is O(N) per membership test; the profile expects a set
    observed = set(map(tuple, np.asarray(data["observed_pairs"]).tolist()))
    profile, coverage = gt2.observed_orbit_profile(
        instances, observed, data["n_items"])
    coverage["families"] = {family: int(len(nodes)) for family, nodes, _ in instances}
    np.savez_compressed(cache, profile=profile,
                        coverage=np.array(json.dumps(coverage)),
                        graph_sha=np.array(data["meta"]["graph_sha"]))
    return profile, coverage


def orbit_probe(data, checkpoints: dict, device, adjacency_map: dict) -> dict:
    profile, coverage = cached_orbit_profile(data)
    target = torch.from_numpy(profile.astype(np.float32)).to(device)
    coverage_mask = profile.sum(1) > 0
    rng = np.random.default_rng(SEED + 11)
    index = np.flatnonzero(coverage_mask)
    order = rng.permutation(len(index))
    cut = int(0.8 * len(order))
    train_index = index[order[:cut]]
    eval_index = index[order[cut:]]
    probe_result = {"coverage": coverage, "items_with_roles": int(len(index)),
                    "probe_train": int(len(train_index)), "probe_eval": int(len(eval_index)),
                    "families": coverage.get("families", {})}
    features = data["features"]
    for name, path in checkpoints.items():
        variant = build_variant(name.split("_seed")[0])
        load_checkpoint(path, variant)
        variant.to(device)
        with torch.no_grad():
            adjacency = adjacency_map.get(variant.name, data["adjacency"])
            state = variant.encode(features, adjacency)
        probe = nn.Linear(state.shape[1], profile.shape[1]).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
        subset = torch.from_numpy(train_index).to(device)
        for _ in range(400):
            optimizer.zero_grad()
            logits = probe(state[subset])
            loss = -(target[subset] * F.log_softmax(logits, -1)).sum(-1).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            evaluation = torch.from_numpy(eval_index).to(device)
            logits = probe(state[evaluation])
            ce = float(-(target[evaluation] * F.log_softmax(logits, -1)).sum(-1).mean())
            entropy = float(-(target[evaluation] *
                              torch.log(target[evaluation].clamp_min(1e-12))).sum(-1).mean())
            # constant predictor: the train-split mean profile
            mean_profile = target[torch.from_numpy(train_index).to(device)].mean(0)
            base_ce = float(-(target[evaluation] *
                              torch.log(mean_profile.clamp_min(1e-12))).sum(-1).mean())
        probe_result[name] = {"ce": ce, "kl": ce - entropy,
                              "target_entropy": entropy, "constant_ce": base_ce,
                              "constant_kl": base_ce - entropy}
    return probe_result


# --------------------------------------------------------------------- audit

def recognition_audit() -> dict:
    path = HERE.parent / "directed_graphlet_role" / "head_pretrained_directed.json"
    head = json.loads(path.read_text())
    report = (path.parent / "REPORT_DIRECTED_GRAPHLET_ROLE.md").read_text()
    return {"with_adjacency": {"subtype_macro_f1": 0.7121, "orbit_macro_f1": 0.6101,
                               "note": "given the true induced adjacency; the answer is "
                                       "largely in the input, not a prediction"},
            "without_adjacency": {"subtype_macro_f1": head["heldout"]["subtype"]["macro_f1"],
                                  "role_macro_over_families":
                                      head["heldout"]["role"]["macro_over_families"],
                                  "per_family": head["heldout"]["role"]["per_family"],
                                  "classes": head["heldout"]["subtype"]["classes"]},
            "deterministic_exact_adjacency": {"subtype_accuracy": 1.0,
                                              "note": "tautological: the family key is "
                                                      "defined by the adjacency itself"},
            "provenance": {"head": str(path.relative_to(REPO)),
                           "head_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "report": str((path.parent / "REPORT_DIRECTED_GRAPHLET_ROLE.md")
                                         .relative_to(REPO)),
                           "report_sha256": hashlib.sha256(
                               (path.parent / "REPORT_DIRECTED_GRAPHLET_ROLE.md")
                               .read_bytes()).hexdigest(),
                           "report_contains_0.7121": "0.7121" in report}}


def raw_semantic_baseline(data) -> dict:
    features = data["features"].cpu().numpy()
    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-9, None)
    result = {}
    for split in ("train", "valid", "test"):
        positive = data[f"{split}_positive"]
        negative = data[f"{split}_negative"]
        positive_score = (features[positive[:, 0]] * features[positive[:, 1]]).sum(1)
        negative_score = (features[positive[:, 0]] * features[negative]).sum(1)
        scores = np.stack([positive_score, negative_score], 1)
        labels = np.concatenate([np.ones(len(positive)), np.zeros(len(positive))])
        flat = np.concatenate([scores[:, 0], scores[:, 1]])
        result[split] = {"auc": float(auc(flat, labels)),
                         "ap": gt2.average_precision(flat, labels),
                         "pairs": int(len(positive))}
    return result


def degree_bands(data) -> dict:
    degrees = np.bincount(data["message_edges"].ravel(), minlength=data["n_items"]).astype(float)
    positive = data["test_positive"]
    negative = data["test_negative"]
    strength = np.maximum(degrees[positive[:, 0]], degrees[positive[:, 1]])
    quantiles = np.quantile(strength, [0.25, 0.5, 0.75])
    bands = {"q1": strength <= quantiles[0], "q2": (strength > quantiles[0]) & (strength <= quantiles[1]),
             "q3": (strength > quantiles[1]) & (strength <= quantiles[2]), "q4": strength > quantiles[2]}
    features = data["features"].cpu().numpy()
    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-9, None)
    result = {"quantiles": [float(q) for q in quantiles],
              "zero_degree_nodes": int((degrees == 0).sum()),
              "median_degree": float(np.median(degrees)),
              "bands": {}}
    for name, mask in bands.items():
        if mask.sum() < 20:
            result["bands"][name] = {"pairs": int(mask.sum()), "skipped": True}
            continue
        pos = positive[mask]
        neg = negative[mask]
        score = (features[pos[:, 0]] * features[pos[:, 1]]).sum(1)
        negative_score = (features[pos[:, 0]] * features[neg]).sum(1)
        flat = np.concatenate([score, negative_score])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(pos))])
        result["bands"][name] = {"pairs": int(mask.sum()), "semantic_auc": float(auc(flat, labels))}
    return result


# --------------------------------------------------------------------- runner

def _valid_eval(variant, data, device):
    def evaluate(model, _step):
        scores = pair_scores(model, data["features"], data["adjacency"],
                             data["valid_positive"], data["valid_negative"], device)
        return auc_of(scores)
    return evaluate


def run_variant(name: str, seed: int, data, task: str, steps: int, lr: float, wd: float,
                device, adjacency, tag: str, early_stop: bool = True) -> dict:
    torch.manual_seed(seed)
    variant = build_variant(name).to(device)
    checkpoint = OUT / f"{name}_{task}_{tag}_seed{seed}.pth"
    result = run_steps(
        variant, data["features"], adjacency,
        data["train_positive"], data["train_negative"],
        data["train_direction"] if "direction" in TASKS[task] else None, None,
        steps=steps, accum=ACCUM, micro=MICRO, lr=lr, weight_decay=wd, device=device,
        counter=StepCounter(), loss_terms=TASKS[task], batch_seed=seed + 1000,
        eval_fn=_valid_eval(variant, data, device),
        patience=PATIENCE if early_stop else None,
        eval_every=EVAL_EVERY if early_stop else None,
        checkpoint_path=checkpoint, checkpoint_every=500)
    metrics = evaluate_splits(variant, data["features"], adjacency, data, device)
    save_checkpoint(checkpoint, variant, {"variant": name, "seed": seed, "task": task,
                                          "lr": lr, "wd": wd, "tag": tag})
    entry = {"seed": seed, "parameters": int(sum(p.numel() for p in variant.parameters())),
             "training": {key: value for key, value in result.items()
                          if key not in ("loss_trace",)},
             "loss_trace": result["loss_trace"][::10] + [result["loss_trace"][-1]],
             "splits": {split: {k: v for k, v in metrics[split].items() if k != "scores"}
                        for split in ("train", "valid", "test")},
             "checkpoint": checkpoint.name,
             "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    if "direction" in TASKS[task]:
        entry["direction_test"] = direction_scores(
            variant, data["features"], adjacency, data["test_direction"], device)
        entry["direction_train"] = direction_scores(
            variant, data["features"], adjacency, data["train_direction"], device)
    entry["scores_test"] = metrics["test"]["scores"]
    return entry


def run_grid(data, task: str, lr: float, wd: float, variants, seeds, device,
             adjacency_map: dict, steps: int, report: dict) -> dict:
    out = {}
    for name in variants:
        out[name] = {}
        for seed in seeds:
            adjacency = adjacency_map[name]
            entry = run_variant(name, seed, data, task, steps, lr, wd, device,
                                adjacency, tag="main")
            out[name][f"seed{seed}"] = entry
            print(f"[{task}] {name} seed{seed} steps={entry['training']['steps']} "
                  f"valid={entry['splits']['valid']['auc']:.4f} "
                  f"test={entry['splits']['test']['auc']:.4f}", flush=True)
            report["stages"].setdefault(task, {})
            report["stages"][task] = {
                name_: {seed_: {k: v for k, v in item.items() if k != "scores_test"}
                        for seed_, item in seeds_.items()}
                for name_, seeds_ in out.items()}
            (OUT / "graph_teacher3_report.json").write_text(json.dumps(report, indent=1,
                                                                       default=str))
            # keep the working arrays in memory only
            out[name][f"seed{seed}"]["scores_test"] = entry["scores_test"]
    return out


def _bootstrap(a_scores: np.ndarray, b_scores: np.ndarray, n_boot: int = 1000,
               seed: int = SEED) -> dict:
    """Cluster bootstrap over positive pairs: (n_pairs, 2) rows stay together."""
    return gt2.bootstrap_ci(a_scores, b_scores, n_boot=n_boot, seed=seed)


def collect_comparisons(grid: dict) -> dict:
    comparisons = {}
    for task, variants in grid.items():
        if not variants:
            continue
        for seed_key in next(iter(variants.values())):
            scores = {name: variants[name][seed_key]["scores_test"]
                      for name in variants if seed_key in variants[name]}
            for left, right, label in (("A_embedding", "B_graph", "B_minus_A"),
                                       ("D_shuffled", "C_residual", "C_minus_D"),
                                       ("A_embedding", "C_residual", "C_minus_A"),
                                       ("E_nograph", "C_residual", "C_minus_E")):
                if left in scores and right in scores:
                    comparisons[f"{task}:{seed_key}:{label}"] = _bootstrap(
                        scores[left], scores[right])
        # seed-averaged scores
        averaged = {}
        for name, seeds in variants.items():
            arrays = [entry["scores_test"] for entry in seeds.values()]
            if arrays:
                averaged[name] = np.mean(np.stack(arrays, axis=0), axis=0)
        for left, right, label in (("A_embedding", "C_residual", "C_minus_A"),
                                   ("D_shuffled", "C_residual", "C_minus_D"),
                                   ("A_embedding", "B_graph", "B_minus_A")):
            if left in averaged and right in averaged:
                comparisons[f"{task}:seed_average:{label}"] = _bootstrap(
                    averaged[left], averaged[right])
    return comparisons


# ---------------------------------------------------------------------- main

def main() -> None:
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    data = prepare_data(device)
    n = data["n_items"]
    print(f"items {n} message edges {len(data['message_edges'])} "
          f"train {len(data['train_positive'])} valid {len(data['valid_positive'])} "
          f"test {len(data['test_positive'])}", flush=True)
    report = {"issue": 228, "config": {
        "seed": SEED, "micro": MICRO, "accum": ACCUM, "max_steps": MAX_STEPS,
        "eval_every": EVAL_EVERY, "patience": PATIENCE, "seeds": list(SEEDS),
        "lr_grid": list(LR_GRID), "wd_grid": list(WD_GRID),
        "split_sha": data["split_sha"], "graph_sha": data["meta"]["graph_sha"],
        "message_edges": int(len(data["message_edges"])),
        "negative_success": data["negative_success"],
        "target_pairs": data["meta"]["target_pairs"]},
        "stages": {}, "gates": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "graph_teacher3_report.json").write_text(json.dumps(report, indent=1, default=str))

    # Gate 0: leakage / alignment audit on the prepared contract
    pairs = set(map(tuple, data["message_edges"].tolist()))
    audit = {"targets_in_message_graph": 0, "negatives_that_are_edges": 0,
             "degenerate_negatives": 0, "misaligned": 0}
    for split in ("train", "valid", "test"):
        positive, negative = data[f"{split}_positive"], data[f"{split}_negative"]
        audit["misaligned"] += int(len(positive) != len(negative))
        for (a, b), d in zip(positive.tolist(), negative.tolist()):
            audit["targets_in_message_graph"] += int((a, b) in pairs or (b, a) in pairs)
            audit["negatives_that_are_edges"] += int(
                (a, d) in data["full_directed"] or (d, a) in data["full_directed"])
            audit["degenerate_negatives"] += int(int(d) in (int(a), int(b)))
    report["gates"]["data_contract"] = {"violations": audit,
                                        "passed": not any(audit.values())}
    (OUT / "graph_teacher3_report.json").write_text(json.dumps(report, indent=1, default=str))
    if any(audit.values()):
        print("DATA CONTRACT FAILED", audit, flush=True)
        return

    # degree-preserving rewiring, cached by the message-graph hash
    cache = OUT / "shuffled_edges.npz"
    if cache.exists():
        stored = np.load(cache)
        shuffled_index = stored["index"]
        rewire_stats = json.loads(stored["stats"].item())
    else:
        targets = np.concatenate([data["masked_pairs"], data["valid_pairs"], data["test_pairs"]])
        forbidden = set(map(tuple, targets.tolist())) | \
                    {(b, a) for a, b in targets.tolist()}
        shuffled_index, rewire_stats = degree_preserving_rewire(
            data["message_edges"], n, forbidden, SEED + 3)
        np.savez_compressed(cache, index=shuffled_index,
                            stats=np.array(json.dumps(rewire_stats)))
    report["gates"]["rewire"] = rewire_stats
    print("rewire", json.dumps(rewire_stats), flush=True)

    adjacency_map = {"A_embedding": None, "B_graph": data["adjacency"],
                     "C_residual": data["adjacency"], "D_shuffled": None,
                     "E_nograph": None}
    shuffled_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(shuffled_index.T.astype(np.int64)),
        torch.ones(shuffled_index.shape[0], dtype=torch.float32),
        size=(n, n)).coalesce().to(device)
    adjacency_map["D_shuffled"] = shuffled_tensor

    # Gate 1: overfit 1,024 fixed pairs
    overfit = {}
    for name, used in (("B_graph", data["adjacency"]), ("C_residual", data["adjacency"])):
        overfit[name] = overfit_gate(
            data["features"], data["train_positive"], data["train_negative"],
            device=device, variant=name, adjacency=used)
        print(f"overfit {name}: auc={overfit[name]['train_auc']:.4f} "
              f"steps={overfit[name]['steps']} passed={overfit[name]['passed']}", flush=True)
    report["gates"]["overfit"] = overfit
    (OUT / "graph_teacher3_report.json").write_text(json.dumps(report, indent=1, default=str))
    if not all(gate["passed"] for gate in overfit.values()):
        report["verdict"] = {"answer": "OPTIMISATION_OR_ENGINEERING",
                            "reason": "the architecture cannot fit a fixed 1,024-pair set",
                            "next": "fix the gradient path before any scientific claim"}
        (OUT / "graph_teacher3_report.json").write_text(
            json.dumps(report, indent=1, default=str))
        return

    # Gate 2a: hyperparameters on valid only, variant C
    selection = {}
    for lr in LR_GRID:
        for wd in WD_GRID:
            torch.manual_seed(SEED)
            model = build_variant("C_residual").to(device)
            result = run_steps(
                model, data["features"], data["adjacency"], data["train_positive"],
                data["train_negative"], None, None, steps=300, accum=ACCUM, micro=MICRO,
                lr=lr, weight_decay=wd, device=device, counter=StepCounter(),
                loss_terms=("edge",), batch_seed=SEED + 1000,
                eval_fn=_valid_eval(model, data, device), patience=150,
                eval_every=EVAL_EVERY)
            selection[f"lr{lr}_wd{wd}"] = {
                "steps": result["steps"], "valid_auc": result["best_metric"],
                "best_step": result["best_step"], "param_sha": result["param_sha_after"]}
            print(f"select lr={lr} wd={wd} valid={result['best_metric']:.4f} "
                  f"steps={result['steps']}", flush=True)
    best_key = max(selection, key=lambda key: selection[key]["valid_auc"])
    lr = float(best_key.split("_")[0][2:])
    wd = float(best_key.split("_")[1][2:])
    report["gates"]["selection"] = {"grid": selection, "chosen": {"lr": lr, "wd": wd},
                                    "rule": "maximum valid AUC; test never read"}
    (OUT / "graph_teacher3_report.json").write_text(json.dumps(report, indent=1, default=str))

    variants = ("A_embedding", "B_graph", "C_residual", "D_shuffled", "E_nograph")
    grid = {}
    grid["edge"] = run_grid(data, "edge", lr, wd, variants, SEEDS, device,
                            adjacency_map, MAX_STEPS, report)
    grid["edge+direction"] = run_grid(data, "edge+direction", lr, wd, variants, SEEDS,
                                      device, adjacency_map, MAX_STEPS, report)

    # 40-step control: the #227 architecture at the now-unambiguous 40 steps
    control = run_variant("B_graph", SEED, data, "edge", CONTROL_STEPS, lr, wd, device,
                          data["adjacency"], tag="control40", early_stop=False)
    report["control_40_step"] = {k: v for k, v in control.items() if k != "scores_test"}
    print(f"control40 steps={control['training']['steps']} "
          f"test={control['splits']['test']['auc']:.4f}", flush=True)

    report["baselines"] = {"raw_semantic": raw_semantic_baseline(data),
                           "high_order": high_order_gate(data, shuffled_index),
                           "degree_bands": degree_bands(data)}
    report["recognition"] = recognition_audit()
    checkpoints = {}
    for name in ("C_residual", "D_shuffled"):
        path = OUT / f"{name}_edge_main_seed{SEEDS[0]}.pth"
        if path.exists():
            checkpoints[name] = path
    if checkpoints:
        report["orbit_probe"] = orbit_probe(data, checkpoints, device, adjacency_map)
    report["comparisons"] = collect_comparisons(grid)

    verdict = _verdict(report)
    report["verdict"] = verdict
    report["seconds"] = time.time() - started
    (OUT / "graph_teacher3_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    print(json.dumps(verdict, indent=1), flush=True)


def _verdict(report) -> dict:
    edge = report.get("stages", {}).get("edge", {})
    if not edge:
        return {"answer": "INCOMPLETE"}
    seed_keys = sorted({key for variants in edge.values() for key in variants})
    stable = {}
    for label, (left, right) in {"C_minus_A": ("A_embedding", "C_residual"),
                                 "C_minus_D": ("D_shuffled", "C_residual")}.items():
        wins = 0
        for seed_key in seed_keys:
            ci = report["comparisons"].get(f"edge:{seed_key}:{label}")
            if ci is not None and ci["low"] > 0 and ci["mean"] > 0.01:
                wins += 1
        stable[label] = {"seeds_with_positive_ci": wins, "seeds": len(seed_keys)}
    gain = all(item["seeds_with_positive_ci"] == item["seeds"] and item["seeds"] >= 2
               for item in stable.values())
    if gain:
        answer = "GRAPH_SIGNAL"
        reason = "C beats A and D with a positive CI on every seed"
    else:
        answer = "NO_GENERALISABLE_SIGNAL"
        reason = ("the architecture fits the fixed small set but the held-out edge task "
                  "does not separate C from A and D at a meaningful effect size")
    return {"answer": answer, "reason": reason, "gates": stable,
            "direction": _direction_verdict(report)}


def _direction_verdict(report) -> dict:
    stage = report.get("stages", {}).get("edge+direction", {})
    result = {}
    for name, seeds in stage.items():
        values = [entry["direction_test"]["balanced_accuracy"] for entry in seeds.values()]
        majority = [entry["direction_test"]["majority_baseline"] for entry in seeds.values()]
        result[name] = {"balanced_accuracy": values, "majority_baseline": majority,
                        "above_majority": bool(values and min(values) > max(majority) + 0.02)}
    return result


if __name__ == "__main__":
    main()
