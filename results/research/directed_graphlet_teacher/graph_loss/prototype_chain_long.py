#!/usr/bin/env python3
"""4000-step run of the current weak chain supervision against a budget-matched control.

Only one question: does longer training keep improving the chain alignment while holding the
collision near the control's?

Protocol fixed before the run:

* chains A->B->C, strictly induced on the directed TRAIN graph;
* **pair-disjoint split**: every unordered pair gets a stable hash bucket, a chain is train
  only if all its pairs are below the cut and eval only if all are above it, so no item pair
  can appear in both sides;
* **no duplicate chain inside a batch** (sampled without replacement), and every per-item
  statistic is aggregated on unique item ids;
* one batch sequence, planned once and replayed by both arms;
* L1 offset inside ``VQLayer._distances`` at weight 0.001, L2 margin at a 10% gradient
  budget, codebooks frozen, all anti-collapse guards on;
* snapshots every 400 steps, metrics computed on the **discrete** SID only.
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
sys.path.insert(0, str(HERE.parent / "inject_directed"))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
from inject_directed import (ARM_DIR, SEMANTICS, load_model,  # noqa: E402
                             q_straight_through)
import sid_alignment_v2 as chains  # noqa: E402
import prototype_chain_wired2 as base  # noqa: E402

OUT = HERE / "prototype_chain_long.json"
STEPS = 4000
SNAPSHOT = 400
BATCH_CHAINS = 256
LR = 1e-3
SEED = 42
TRAIN_FRACTION = 60
MARGIN = 0.5
ALIGN = 0.001
BUDGET = 0.10
PENDING_OFFSET = None


def pair_bucket(a: int, b: int) -> int:
    key = f"{min(a, b)}-{max(a, b)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100


def split_pair_disjoint(bank: np.ndarray):
    """Train and eval chains may not share a single item pair."""
    train, evaluate = [], []
    for a, b, c in bank.tolist():
        buckets = {pair_bucket(a, b), pair_bucket(b, c), pair_bucket(a, c)}
        if all(value < TRAIN_FRACTION for value in buckets):
            train.append((a, b, c))
        elif all(value >= TRAIN_FRACTION for value in buckets):
            evaluate.append((a, b, c))
    return np.array(train, dtype=np.int64), np.array(evaluate, dtype=np.int64)


def plan_batches(bank: np.ndarray, steps: int, size: int, seed: int):
    """Sampled without replacement: a batch never contains the same chain twice."""
    generator = np.random.default_rng(seed)
    batches = []
    for _ in range(steps):
        pick = generator.choice(len(bank), size=size, replace=False)
        batches.append(bank[pick])
    return batches


def evaluate(codes, bank, rng, semantics, q_np) -> dict:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    cumulative = lambda left, right: np.cumprod(codes[left] == codes[right], axis=1).sum(1)
    adjacent = (cumulative(a, b) + cumulative(b, c)) / 2.0
    far = cumulative(a, c)
    delta = float(adjacent.mean() - far.mean())
    l1_adj = ((codes[a, 0] == codes[b, 0]).astype(np.float64)
              + (codes[b, 0] == codes[c, 0]).astype(np.float64)).mean() / 2.0
    l1_far = float((codes[a, 0] == codes[c, 0]).mean())
    l2_adj = ((codes[a, :2] == codes[b, :2]).all(1).astype(np.float64)
              + (codes[b, :2] == codes[c, :2]).all(1).astype(np.float64)).mean() / 2.0
    l2_far = float((codes[a, :2] == codes[c, :2]).all(1).mean())
    null = []
    for _ in range(100):
        shuffled = codes[rng.permutation(len(codes))]
        shuffled_cum = lambda left, right: np.cumprod(
            shuffled[left] == shuffled[right], axis=1).sum(1)
        null.append(float((shuffled_cum(a, b).mean() + shuffled_cum(b, c).mean()) / 2.0
                          - shuffled_cum(a, c).mean()))
    return {"chain_delta": delta,
            "chain_delta_null": [float(np.percentile(null, 2.5)),
                                 float(np.percentile(null, 97.5))],
            "L1_adjacent": float(l1_adj), "L1_far": l1_far,
            "L2_adjacent": float(l2_adj), "L2_far": l2_far,
            "collision": float(1.0 - len(np.unique(codes, axis=0)) / len(codes)),
            "semantic_r2": base.semantic_probe(q_np, semantics)}


def install_offset(layers, align: float) -> None:
    """Route this module's pending offset into the arm's level-1 assignment."""
    first = layers[0]
    original = first._distances

    def patched(latent):
        base_cost = original(latent)
        if PENDING_OFFSET is None or PENDING_OFFSET.shape != base_cost.shape:
            return base_cost
        spread = base_cost.std(dim=1, keepdim=True).clamp(min=1e-6)
        offset = PENDING_OFFSET
        offset = (offset - offset.mean(dim=1, keepdim=True)) / offset.std(
            dim=1, keepdim=True).clamp(min=1e-6)
        return base_cost + align * offset * spread

    first._distances = patched


def neighbour_offset(code_table, flat, partners, codes: int = 256):
    prior = np.full((len(flat), codes), 1.0, dtype=np.float64)
    for row, item in enumerate(flat.tolist()):
        seen = 0
        for partner in partners.get(int(item), ()):
            code = int(code_table[partner, 0])
            if code >= 0:
                prior[row, code] += 1.0
                seen += 1
        if seen:
            prior[row] /= prior[row].sum()
    prior /= prior.sum(1, keepdims=True)
    return torch.from_numpy(-np.log(prior).astype(np.float32))


def main() -> None:
    global PENDING_OFFSET
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "wired", "l2only"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--budget", type=float, default=BUDGET)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows, cols, _ = chains.loaded_edges()
    out_map, _ = chains.adjacency(rows, cols)
    full_bank = chains.chain3(out_map, np.stack([rows, cols], axis=1), rng, chains.CAP)
    train_bank, eval_bank = split_pair_disjoint(full_bank)
    batches = plan_batches(train_bank, args.steps, BATCH_CHAINS, SEED)
    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    model = load_model().to(device)
    for layer in model.rq.vq_layers:
        layer.embed.requires_grad_(False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    layers = list(model.rq.vq_layers)
    if args.arm == "wired":
        install_offset(layers, ALIGN)
    from drq import level_softmax, measure_temperature
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    def snapshot(label):
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            codes = tokens.cpu().numpy().astype(np.int64)
            q_np = q.cpu().numpy().astype(np.float32)
        stats = evaluate(codes, eval_bank, rng, semantics, q_np)
        print(f"  {label}: chainD {stats['chain_delta']:+.4f} "
              f"null[{stats['chain_delta_null'][0]:+.4f},{stats['chain_delta_null'][1]:+.4f}] "
              f"L1 {stats['L1_adjacent']:.4f}/{stats['L1_far']:.4f} "
              f"L2 {stats['L2_adjacent']:.4f}/{stats['L2_far']:.4f} "
              f"coll {stats['collision']:.4f} semR2 {stats['semantic_r2']:.3f}", flush=True)
        return stats

    report = {"protocol": {"arm": args.arm, "steps": args.steps, "seed": SEED,
                           "chains": {"total": int(len(full_bank)),
                                      "train": int(len(train_bank)),
                                      "eval": int(len(eval_bank))},
                           "split": "pair-disjoint by hash of the unordered item pair",
                           "batch": "256 chains sampled without replacement, replayed "
                                    "identically for both arms",
                           "align": ALIGN, "budget": args.budget, "margin": MARGIN,
                           "codebook": "frozen"}, "curve": {}}
    report["curve"]["0"] = snapshot("step 0")
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    code_table = np.full((len(semantics), 3), -1, dtype=np.int64)
    partners = base.build_partners(train_bank)
    for step, batch_chains in enumerate(batches, start=1):
        flat = batch_chains.reshape(-1)
        flat_tensor = torch.from_numpy(flat).to(device)
        left = torch.arange(0, len(flat), 3, device=device)
        middle, right = left + 1, left + 2
        detail = {}
        if args.arm == "wired":
            PENDING_OFFSET = neighbour_offset(code_table, flat, partners).to(device)
            with torch.no_grad():
                probe = model.encoder(items[flat_tensor])
                plain = layers[0].distance(probe, layers[0].get_code_embs())
                centered = PENDING_OFFSET - PENDING_OFFSET.mean(dim=1, keepdim=True)
                centered = centered / centered.std(dim=1, keepdim=True).clamp(min=1e-6)
                shifted = plain + ALIGN * centered * plain.std(dim=1, keepdim=True).clamp(min=1e-6)
                changed = float((layers[0]._indices(plain, True)
                                 != layers[0]._indices(shifted, True)).float().mean())
        else:
            PENDING_OFFSET = None
            changed = 0.0
        model.train()
        recon, quant_parts, _, tokens_batch = model(items[flat_tensor])
        code_table[flat] = tokens_batch.detach().cpu().numpy().astype(np.int64)
        loss_rq, _ = criterion(recon, items[flat_tensor], quant_parts)
        optimizer.zero_grad(set_to_none=True)
        loss_rq.backward(retain_graph=True)
        rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                            if p.grad is not None) ** 0.5)
        if args.arm != "control":
            z = model.encoder(items[flat_tensor])
            temperature = measure_temperature(layers[0], z)
            p1 = level_softmax(layers[0], z, temperature)
            residual = z - p1 @ layers[0].get_code_embs()
            temperature2 = measure_temperature(layers[1], residual)
            p2 = level_softmax(layers[1], residual, temperature2)
            s2_ab = (p1[left] * p1[middle]).sum(-1) * (p2[left] * p2[middle]).sum(-1)
            s2_bc = (p1[middle] * p1[right]).sum(-1) * (p2[middle] * p2[right]).sum(-1)
            s2_ac = (p1[left] * p1[right]).sum(-1) * (p2[left] * p2[right]).sum(-1)
            loss_l2 = torch.clamp(MARGIN + s2_ac - (s2_ab + s2_bc) / 2.0, min=0.0).mean()
            gradients = torch.autograd.grad(loss_l2, encoder_parameters, retain_graph=False,
                                            allow_unused=True)
            norm = float(sum(float(g.pow(2).sum()) for g in gradients
                             if g is not None) ** 0.5)
            scale = min(1.0, args.budget * rq_norm / max(norm, 1e-12))
            for parameter, grad in zip(encoder_parameters, gradients):
                if grad is None:
                    continue
                parameter.grad = scale * grad if parameter.grad is None \
                    else parameter.grad + scale * grad
            detail = {"l2_loss": float(loss_l2.detach()), "grad_scale": scale,
                      "changed": changed, "temperature": [temperature, temperature2]}
        optimizer.step()
        if step % SNAPSHOT == 0 or step == args.steps:
            stats = snapshot(f"step {step}")
            stats.update(detail)
            report["curve"][str(step)] = stats
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
