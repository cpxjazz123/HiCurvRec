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
                       device, steps: int = 800, holdout: float = 0.3) -> dict:
    """Independent probe: does q of the two items tell A->B from B->A?

    The probe trains on one slice of the pairs and reports the other, so its number is a
    generalisation result.  Training and evaluating on the same pairs inflates every arm
    and makes them indistinguishable.
    """
    one_way = (truth[:, 0] != truth[:, 1])
    if one_way.sum() < 100:
        return {"accuracy": None, "pairs": int(one_way.sum())}
    ordered = pairs[one_way]
    labels_all = truth[one_way][:, 1] > 0.5
    rng_split = np.random.default_rng(SEED + 3)
    order = rng_split.permutation(len(ordered))
    cut = int((1.0 - holdout) * len(order))
    train_idx, eval_idx = order[:cut], order[cut:]
    sub = ordered
    sub_train, sub_eval = ordered[train_idx], ordered[eval_idx]
    target = torch.from_numpy(labels_all[train_idx].astype(np.int64)).to(device)
    target_eval = torch.from_numpy(labels_all[eval_idx].astype(np.int64)).to(device)
    x = torch.from_numpy(q_items[sub_train.reshape(-1)].reshape(len(sub_train), 2, -1)).to(device)
    x_eval = torch.from_numpy(q_items[sub_eval.reshape(-1)].reshape(len(sub_eval), 2, -1)).to(device)
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
        pair = torch.cat([x_eval[:, 0], x_eval[:, 1], x_eval[:, 0] * x_eval[:, 1]], dim=-1)
        guess = head(pair).argmax(-1)
        accuracy = float((guess == target_eval).float().mean())
    return {"accuracy": accuracy, "pairs_eval": int(len(sub_eval)),
            "pairs_train": int(len(sub_train))}


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
    # the arm's collapse guard stays armed: if it fires, the arm failed and that is the
    # result.  Switching it off would make a collapsed codebook look like a finding.
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
    elif mode == "D_direction_random":
        # a random orientation per one-way pair: "reversal" keeps a perfect, learnable
        # relation (the complement), so it is not a null.  A coin flip destroys exactly
        # the direction while keeping the edge structure.
        randomised = labels.clone()
        one_way = labels[:, 0] != labels[:, 1]
        flip = torch.from_numpy(
            np.random.default_rng(SEED + 2).random(len(labels)) < 0.5).to(labels.device)
        chosen = one_way & flip
        randomised[chosen] = labels[chosen][:, [1, 0]]
        labels = randomised
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text())
                           if (ARM_DIR / "config" / "train_config.json").exists() else
                           {"BETA": 0.25})
    trace = []
    for step in range(STEPS):
        pick = torch.randint(0, len(pairs), (BATCH,), generator=rng)
        guess_pairs = pairs[pick].to(device)
        # the reconstruction term is kept: an auxiliary loss alone reshapes the codebook
        # with nothing holding it to the data, which is what collapsed it last time
        recon, quant_parts, _, _ = model(items)
        loss_rqvae, _ = criterion(recon, items, quant_parts)
        q_all, _ = q_straight_through(model, items)
        left = q_all[guess_pairs[:, 0]]
        right = q_all[guess_pairs[:, 1]]
        logits = ((source(left) * target(right)).sum(-1))
        loss_directed = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[pick, 0].to(device))
        loss_directed = loss_directed + torch.nn.functional.binary_cross_entropy_with_logits(
            (source(right) * target(left)).sum(-1), labels[pick, 1].to(device))
        loss = loss_rqvae + LAMBDA * loss_directed
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == STEPS - 1:
            trace.append({"step": step, "loss": float(loss.detach()),
                          "rqvae": float(loss_rqvae.detach()),
                          "directed": float(loss_directed.detach())})
            print(f"    {mode} step {step}: total {float(loss.detach()):.4f} "
                  f"(rqvae {float(loss_rqvae.detach()):.4f}, "
                  f"directed {float(loss_directed.detach()):.4f})", flush=True)
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

    for mode in ("B_undirected", "C_directed", "D_direction_random"):
        model = load_model().to(device)
        before = fit_direction_head(base_q, eval_pairs, eval_truth, device)
        from assignment_floor import CodebookCollapse
        try:
            info = train_arm(model, items,
                             torch.from_numpy(train_pairs).to(device),
                             torch.from_numpy(train_truth).to(device), mode, device)
        except CodebookCollapse as collapse:
            # the arm failed inside its own guard: record it and keep the run going so the
            # other arms are still measured instead of being aborted with it
            info = {"trained": False, "collapsed": True, "reason": str(collapse)[:300]}
            report["arms"][mode] = {"trained": False,
                                    "probe_before": fit_direction_head(
                                        base_q, eval_pairs, eval_truth, device),
                                    "probe_after": None, "codebook_collapsed": True,
                                    "reason": str(collapse)[:300]}
            print(f"  {mode}: CODEBOOK COLLAPSE -- {str(collapse)[:120]}", flush=True)
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
            continue
        model.eval()
        with torch.no_grad():
            q_after, _ = q_straight_through(model, items)
            q_after = q_after.detach().cpu().numpy()
        after = fit_direction_head(q_after, eval_pairs, eval_truth, device)
        with torch.no_grad():
            _, _, _, tokens = model.rq(model.encoder(items))
        codes = tokens.cpu().numpy().astype(np.int64)
        current = {"trained": True, "trace": info.get("trace"),
                   "probe_before": before, "probe_after": after,
                   "q_moved_l2": float(np.linalg.norm(q_after - base_q) / max(len(base_q), 1)),
                   "sid_collision": float(1.0 - len(np.unique(codes[:, :3], axis=0)) / len(codes)),
                   "level1_codes_used": int(len(np.unique(codes[:, 0]))),
                   "codeword_spacing": float(np.min(np.linalg.norm(
                       np.asarray(model.rq.vq_layers[0].get_code_embs().detach().cpu())[:, None]
                       - np.asarray(model.rq.vq_layers[0].get_code_embs().detach().cpu())[None],
                       axis=-1) + np.eye(256) * 1e9))}
        report["arms"][mode] = current
        print(f"  {mode}: probe {before['accuracy']:.4f} -> {after['accuracy']:.4f} "
              f"(q moved {current['q_moved_l2']:.3f})", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
