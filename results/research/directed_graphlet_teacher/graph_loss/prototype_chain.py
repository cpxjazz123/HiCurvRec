#!/usr/bin/env python3
"""Chain-only prototype: can a single margin loss move the chain code distribution?

One structure only: A -> B -> C, strictly induced (no reverse edge, no A->C shortcut, B and
C not adjacent, A and B not adjacent beyond the required edge).

The soft statistic is the expected number of shared leading codes,

    S(u,v) = p1(u).p1(v) + p1p2 + p1p2p3,

and the loss is the margin form the owner proposed,

    L = max(0, m + S(A,C) - (S(A,B) + S(B,C)) / 2),

so nothing forces *which* adjacent pair shares L2 -- only that the adjacent pairs jointly share
more than the two-hop pair.  Evaluation always uses the **discrete** SID, never the soft value.

Controls kept from the earlier work: codebooks frozen, the chain term's encoder gradient is
budgeted against the reconstruction gradient every step, and collision plus semantic R^2 are
reported so a win cannot come from wrecking the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inject_directed"))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
from inject_directed import (ARM_DIR, SEMANTICS, load_model,  # noqa: E402
                             q_straight_through)
import augment_gnn as base  # noqa: E402
import sid_alignment_v2 as chains  # noqa: E402

OUT = HERE / "prototype_chain.json"
STEPS = 400
BATCH_CHAINS = 256
MARGIN = 0.5
GRAD_SHARE = 0.5
LR = 1e-3
SEED = 42
TRAIN_FRACTION = 0.6


def soft_prefix(model, items: torch.Tensor, temperatures):
    """p1.p1, p1.p2 and p1.p2.p3 style levels: return the three per-level agreements."""
    from drq import level_softmax
    z = model.encoder(items)
    residual = z
    probabilities = []
    for level, layer in enumerate(model.rq.vq_layers):
        if level >= 3:
            break
        temperature = temperatures[level] if temperatures[level] > 0 else 1.0
        probabilities.append(level_softmax(layer, residual, temperature))
        residual = residual - probabilities[-1] @ layer.get_code_embs()
    return probabilities


def soft_share(probabilities, left, right) -> torch.Tensor:
    """Expected number of shared leading codes between paired rows."""
    agree = []
    running = None
    for probability in probabilities:
        dot = (probability[left] * probability[right]).sum(-1)
        running = dot if running is None else running * dot
        agree.append(running)
    return torch.stack(agree).sum(0)


def discrete_share(codes: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    equal = codes[left] == codes[right]
    return np.cumprod(equal, axis=1).sum(1)


def split_chains(bank: np.ndarray) -> tuple:
    """Deterministic 60/40 split by a hash of the chain's own node triple."""
    import hashlib
    keys = ["|".join(str(int(v)) for v in row) for row in bank]
    train = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100
                      < TRAIN_FRACTION * 100 for k in keys])
    return bank[train], bank[~train]


def chain_delta(codes: np.ndarray, bank: np.ndarray) -> dict:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    adjacent = (discrete_share(codes, a, b) + discrete_share(codes, b, c)) / 2.0
    far = discrete_share(codes, a, c)
    delta = float(adjacent.mean() - far.mean())
    ordered = float((adjacent > far).mean())
    return {"delta": delta, "ordered_share": ordered,
            "adjacent_mean": float(adjacent.mean()), "far_mean": float(far.mean()),
            "instances": int(len(bank))}


def permutation_null(codes: np.ndarray, bank: np.ndarray, rng, rounds: int = 200) -> dict:
    deltas = []
    for _ in range(rounds):
        shuffled = codes[rng.permutation(len(codes))]
        a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
        adjacent = (discrete_share(shuffled, a, b) + discrete_share(shuffled, b, c)) / 2.0
        deltas.append(float(adjacent.mean() - discrete_share(shuffled, a, c).mean()))
    deltas = np.asarray(deltas)
    return {"low": float(np.percentile(deltas, 2.5)),
            "high": float(np.percentile(deltas, 97.5))}


def semantic_probe(codes: np.ndarray, semantics: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(codes))
    cut = int(0.8 * len(order))
    train, evaluate = order[:cut], order[cut:]
    x = codes[train].astype(np.float64)
    xc = x - x.mean(0, keepdims=True)
    lam = 1e-3 * float(np.trace(xc.T @ xc)) / xc.shape[1]
    weights = np.linalg.solve(xc.T @ xc + lam * np.eye(xc.shape[1]),
                              xc.T @ (semantics[train] - semantics[train].mean(0)))
    predicted = (codes[evaluate] - x.mean(0)) @ weights + semantics[train].mean(0)
    return float(1.0 - ((predicted - semantics[evaluate]) ** 2).mean()
                 / ((semantics[evaluate] - semantics[evaluate].mean(0)) ** 2).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    rows, cols, _ = chains.loaded_edges()
    out_map, in_map = chains.adjacency(rows, cols)
    bank = chains.chain3(out_map, np.stack([rows, cols], axis=1), rng, chains.CAP)
    train_bank, eval_bank = split_chains(bank)

    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    model = load_model().to(device)
    for layer in model.rq.vq_layers:
        layer.embed.requires_grad_(False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    from drq import measure_temperature
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    def evaluate(label: str) -> dict:
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            codes = tokens.cpu().numpy().astype(np.int64)
            q_np = q.cpu().numpy().astype(np.float32)
        stats = chain_delta(codes, eval_bank)
        null = permutation_null(codes, eval_bank, rng)
        collision = float(1.0 - len(np.unique(codes[:, :3], axis=0)) / len(codes))
        result = {**stats, "null_low": null["low"], "null_high": null["high"],
                  "collision": collision,
                  "semantic_r2": semantic_probe(q_np, semantics), "codes": codes}
        print(f"  {label}: chain delta {stats['delta']:+.4f} "
              f"(ordered {stats['ordered_share']:.3f}, null "
              f"[{null['low']:+.3f}, {null['high']:+.3f}]) collision {collision:.4f} "
              f"semR2 {result['semantic_r2']:.3f}", flush=True)
        return result

    before = evaluate("before")

    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    temperatures = []
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(train_bank[:1024].reshape(-1)).to(device)
        z = model.encoder(items[sample])
        residual = z
        for layer in model.rq.vq_layers:
            temperatures.append(float(measure_temperature(layer, residual)))
            residual = residual - layer.get_code_embs()[
                torch.cdist(residual, layer.get_code_embs()).argmin(-1)]
    print("temperatures", temperatures, flush=True)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    log = []
    for step in range(STEPS):
        pick = torch.randint(0, len(train_bank), (BATCH_CHAINS,), generator=generator)
        batch_chains = train_bank[pick.numpy()]
        flat = torch.from_numpy(batch_chains.reshape(-1)).to(device)
        left = torch.arange(0, len(flat), 3, device=device)
        middle = left + 1
        right = left + 2
        model.train()
        recon, quant_parts, _, _ = model(items[flat])
        loss_rq, _ = criterion(recon, items[flat], quant_parts)
        optimizer.zero_grad(set_to_none=True)
        loss_rq.backward(retain_graph=True)
        rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                            if p.grad is not None) ** 0.5)
        probabilities = soft_prefix(model, items[flat], temperatures)
        s_ab = soft_share(probabilities, left, middle)
        s_bc = soft_share(probabilities, middle, right)
        s_ac = soft_share(probabilities, left, right)
        hinge = torch.clamp(MARGIN + s_ac - (s_ab + s_bc) / 2.0, min=0.0).mean()
        chain_grads = torch.autograd.grad(hinge, encoder_parameters, retain_graph=False,
                                          allow_unused=True)
        chain_norm = float(sum(float(g.pow(2).sum()) for g in chain_grads
                               if g is not None) ** 0.5)
        scale = min(1.0, GRAD_SHARE * rq_norm / max(chain_norm, 1e-12))
        for parameter, grad in zip(encoder_parameters, chain_grads):
            if grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = scale * grad
            else:
                parameter.grad = parameter.grad + scale * grad
        optimizer.step()
        if step % 50 == 0 or step == STEPS - 1:
            log.append({"step": step, "hinge": float(hinge), "scale": scale,
                        "rq_norm": rq_norm, "chain_norm": chain_norm})
            print(f"    step {step}: hinge {float(hinge):.4f} |g_chain| {chain_norm:.3e} "
                  f"|g_rq| {rq_norm:.3e} scale {scale:.3f}", flush=True)

    after = evaluate("after")
    report = {"protocol": {
        "structure": "A->B->C strictly induced in the directed TRAIN graph",
        "chains": {"total": int(len(bank)), "train": int(len(train_bank)),
                   "eval": int(len(eval_bank))},
        "loss": "max(0, m + S(A,C) - (S(A,B)+S(B,C))/2), m = 0.5",
        "S": "soft: p1.p1 + p1.p2p2 + p1p2p3 agreements; evaluation uses the discrete SID",
        "gradient_budget": f"chain encoder-grad <= {GRAD_SHARE} x reconstruction, measured "
                           f"every step",
        "codebook": "frozen", "steps": STEPS, "seed": SEED},
        "before": {k: v for k, v in before.items() if k != "codes"},
        "after": {k: v for k, v in after.items() if k != "codes"},
        "log": log}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    np.save(HERE / "prototype_chain_codes_after.npy", after["codes"])
    print("written", args.out)


if __name__ == "__main__":
    main()
