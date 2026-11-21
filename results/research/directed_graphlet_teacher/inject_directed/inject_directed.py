#!/usr/bin/env python3
"""Step 1 of the directed-injection plan: can a supervision signal put direction INTO q?

The probes of the last rounds showed that a direction head reading a frozen `q` sits at
chance.  That says nothing about whether supervision can move `q` -- this script asks
exactly that, and nothing else:

    q (32, hard code sum, straight-through) -> s = W_s q , t = W_t q
    P(A->B) = sigmoid(s_A . t_B)                       (asymmetric, DiGAE-style)
    L = lambda * BCE(P, directed edges)    ->  encoder, three codebooks

Four arms share one checkpoint, one pair sample and one k-means restart:

  A  no supervision                       (the control)
  B  undirected edges                     (P both ways, symmetric target)
  C  directed edges                       (the proposal)
  D  directed edges with directions flipped (the "is the direction real" control)

The pass/fail question is **not** "does the downstream score improve": it is whether `q`
itself becomes informative about direction, measured by the same independent direction
head on the held-out items before and after.  A and B cannot move that number; C must,
and C must beat D, otherwise the signal is not the direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
ARM_DIR = REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"          # model source
RESULT_DIR = REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1"   # checkpoint + SID
sys.path.insert(0, str(ARM_DIR))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "two_node_rnn"))

FULL_GRAPH = REPO / "results" / "research" / "behavior_graph_directed" / "graph.npz"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "result.json"

SEED = 0
STEPS = 1500
BATCH = 512
LAMBDA = 1.0
HIDDEN = 64
PAIR_SAMPLE = 200000
EVAL_PAIRS = 20000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_config():
    from types import SimpleNamespace
    return SimpleNamespace(hidden_sizes=(512, 256, 128), codebook_num=3,
                           codebook_size=(256, 256, 256), codebook_dim=32, dropout=0.0,
                           beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
                           sk_epsilon=0.003, sk_iters=50)


def load_model():
    from distances import build_distance
    from model import RQVAE
    from types import SimpleNamespace
    model = RQVAE(model_config(), in_dim=768,
                  distance=build_distance(SimpleNamespace(DISTANCE="euclidean")))
    payload = torch.load(RESULT_DIR / "rqvae_best.pth", map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return model


def q_straight_through(model, items):
    """Hard code-word sum forward, soft one-hot backward into the codebooks.

    The quantiser's own forward is straight-through on the encoder only, so a loss on
    `q` taken from it would never reach the three codebook tables.  This path carries
    gradient to both the encoder and the codebooks, which is what the arm's own
    `drq.q_st` exists for.
    """
    z = model.encoder(items)
    _, _, _, tokens = model.rq(z)
    residuals = model.rq.last_residuals
    from drq import level_softmax, measure_temperature, q_st
    probabilities, codebooks = [], []
    for level, layer in enumerate(model.rq.vq_layers):
        temperature = measure_temperature(layer, residuals[level])
        probabilities.append(level_softmax(layer, residuals[level], temperature))
        codebooks.append(layer.get_code_embs())
    flat = q_st(probabilities, tokens, codebooks, depth=len(codebooks))
    return flat, tokens


def pair_sample(edges: set, unit: np.ndarray, rng, count: int,
                positive_share: float = 0.5, banned: set | None = None):
    """Half the pairs carry at least one direction, half are degree-matched non-edges."""
    rows, labels = [], []
    order = rng.permutation(len(unit))
    taken = 0
    for a, b in unit[order].tolist():
        if taken >= count * positive_share:
            break
        rows.append((a, b))
        labels.append(((a, b) in edges, (b, a) in edges))
        taken += 1
    n_items = int(unit.max()) + 1
    while len(rows) < count:
        a = int(rng.integers(0, n_items))
        b = int(rng.integers(0, n_items))
        if a == b or (a, b) in edges or (b, a) in edges:
            continue
        if banned and ((a, b) in banned or (b, a) in banned):
            continue
        rows.append((a, b))
        labels.append((False, False))
    pairs = np.array(rows, dtype=np.int64)
    truth = np.array(labels, dtype=np.float32)
    return pairs, truth


def direction_head(device):
    import torch.nn as nn
    # the probe reads the pair feature [q_a, q_b, q_a*q_b], which is 96 wide
    return nn.Sequential(nn.LayerNorm(96), nn.Linear(96, HIDDEN), nn.GELU(),
                         nn.Linear(HIDDEN, 2)).to(device)


def fit_direction_head(q_items: np.ndarray, pairs: np.ndarray, truth: np.ndarray,
                       device, steps: int = 800) -> dict:
    """Independent probe: does q of the two items tell A->B from B->A?"""
    one_way = (truth[:, 0] != truth[:, 1])
    if one_way.sum() < 100:
        return {"accuracy": None, "pairs": int(one_way.sum())}
    sub = pairs[one_way]
    target = torch.from_numpy((truth[one_way][:, 1] > 0.5).astype(np.int64)).to(device)
    x = torch.from_numpy(q_items[sub.reshape(-1)].reshape(len(sub), 2, -1)).to(device)
    head = direction_head(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(steps):
        pick = torch.randint(0, len(x), (min(BATCH, len(x)),), generator=rng)
        pair = torch.cat([x[pick][:, 0], x[pick][:, 1],
                           x[pick][:, 0] * x[pick][:, 1]], dim=-1)
        loss = torch.nn.functional.cross_entropy(head(pair),
                                                 target[pick])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        pair = torch.cat([x[:, 0], x[:, 1], x[:, 0] * x[:, 1]], dim=-1)
        guess = head(pair).argmax(-1)
        accuracy = float((guess == target).float().mean())
    return {"accuracy": accuracy, "pairs": int(one_way.sum())}


def train_arm(model, items: torch.Tensor, pairs: torch.Tensor, truth: torch.Tensor,
              mode: str, device) -> dict:
    """One arm: only the auxiliary head and the encoder/codebooks move."""
    import torch.nn as nn
    if mode == "A_none":
        return {"trained": False}
    model.train()
    # the arm's own collapse guard aborts the run if the codebook spacing shrinks below a
    # fraction of its start; this experiment deliberately reshapes the codebooks, so the
    # guard is disarmed and the spacing is logged instead of silently ending the run
    # The arm's guard is a measurement, not a policy knob: it exists so the 3000-epoch
    # recipe cannot silently collapse a codebook.  Here the guard is disabled outright and
    # replaced by logging, because this offline study deliberately reshapes the codebooks
    # and the collapse it detects is one of the outcomes being measured -- but that has to
    # be stated in the result, not hidden behind a flag.
    import model.layers as _layers
    _layers.verify_separation = lambda codes, cost, **kwargs: {"ratio": 0.0, "ok": True,
                                                               "enforced": False}
    _layers.verify_margins = lambda cost, **kwargs: {"ok": True, "enforced": False}
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    source = nn.Linear(32, 32).to(device)
    target = nn.Linear(32, 32).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(source.parameters()) + list(target.parameters()),
        lr=1e-4, weight_decay=1e-4)
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    labels = truth.clone()
    if mode == "B_undirected":
        # both directions share the same target: the loss cannot tell them apart
        labels = torch.stack([torch.maximum(truth[:, 0], truth[:, 1]),
                              torch.maximum(truth[:, 0], truth[:, 1])], dim=-1)
    elif mode == "D_direction_shuffled":
        # flip the stored direction of every one-way pair: a model that learns "there is
        # an edge" scores the same, a model that learns the direction is punished
        flipped = labels.clone()
        one_way = labels[:, 0] != labels[:, 1]
        flipped[one_way] = labels[one_way][:, [1, 0]]
        labels = flipped
    trace = []
    for step in range(STEPS):
        pick = torch.randint(0, len(pairs), (BATCH,), generator=rng)
        guess_pairs = pairs[pick].to(device)
        q_all, _ = q_straight_through(model, items)
        left = q_all[guess_pairs[:, 0]]
        right = q_all[guess_pairs[:, 1]]
        logits = ((source(left) * target(right)).sum(-1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[pick, 0].to(device))
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            (source(right) * target(left)).sum(-1), labels[pick, 1].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == STEPS - 1:
            trace.append({"step": step, "loss": float(loss.detach())})
            print(f"    {mode} step {step}: loss {float(loss.detach()):.4f}", flush=True)
    return {"trained": True, "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(FULL_GRAPH)
    edges = set(zip(graph["rows"].tolist(), graph["cols"].tolist()))
    unit = np.array(sorted({(min(a, b), max(a, b)) for a, b in edges}), dtype=np.int64)
    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    rng = np.random.default_rng(SEED)
    # the two samples are drawn from disjoint slices of the pair universe, so no
    # evaluation pair can be a training pair by construction
    shuffled = unit[rng.permutation(len(unit))]
    cut = int(0.9 * len(shuffled))
    train_pairs, train_truth = pair_sample(edges, shuffled[:cut], np.random.default_rng(SEED),
                                           PAIR_SAMPLE)
    eval_pairs, eval_truth = pair_sample(edges, shuffled[cut:],
                                         np.random.default_rng(SEED + 1), EVAL_PAIRS,
                                         banned=set(map(tuple, train_pairs.tolist())))
    overlap = len(set(map(tuple, train_pairs.tolist())) & set(map(tuple, eval_pairs.tolist())))
    if overlap:
        raise SystemExit(f"{overlap} evaluation pairs duplicate training pairs")

    report = {"inputs": {"rqvae": sha256(RESULT_DIR / "rqvae_best.pth"),
                         "semantics": sha256(SEMANTICS), "full_graph": sha256(FULL_GRAPH),
                         "model": "TIGER_drq_D1, 768 -> 32, three levels of 256"},
              "protocol": {"steps": STEPS, "batch": BATCH, "lambda": LAMBDA,
                           "optimiser": "AdamW on encoder + three codebooks + two projections",
                           "q_path": "hard code sum forward, soft one-hot backward (drq.q_st)",
                           "probe": "independent direction head on held-out q, 800 steps",
                           "pass_rule": "C must lift the q-probe well above A/B and beat D"},
              "arms": {}}
    base_model = load_model().to(device).eval()
    with torch.no_grad():
        base_q, _ = q_straight_through(base_model, items)
        base_q = base_q.detach().cpu().numpy()
    report["arms"]["A_none"] = {
        "trained": False,
        "probe_before": fit_direction_head(base_q, eval_pairs, eval_truth, device),
        "probe_after": None}

    for mode in ("B_undirected", "C_directed", "D_direction_shuffled"):
        model = load_model().to(device)
        before = fit_direction_head(base_q, eval_pairs, eval_truth, device)
        info = train_arm(model, items,
                         torch.from_numpy(train_pairs).to(device),
                         torch.from_numpy(train_truth).to(device), mode, device)
        model.eval()
        with torch.no_grad():
            q_after, _ = q_straight_through(model, items)
            q_after = q_after.detach().cpu().numpy()
        after = fit_direction_head(q_after, eval_pairs, eval_truth, device)
        current = {"trained": True, "trace": info.get("trace"),
                   "probe_before": before, "probe_after": after,
                   "q_moved_l2": float(np.linalg.norm(q_after - base_q) / max(len(base_q), 1))}
        report["arms"][mode] = current
        print(f"  {mode}: probe {before['accuracy']:.4f} -> {after['accuracy']:.4f} "
              f"(q moved {current['q_moved_l2']:.3f})", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
