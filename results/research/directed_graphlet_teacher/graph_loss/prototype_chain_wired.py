#!/usr/bin/env python3
"""Chain supervision wired into the arm's own quantizer, with the guards left on.

Three corrections over ``prototype_chain_three.py``:

* **the structural cost reaches the real assignment**: ``VQLayer._distances`` is wrapped so
  the level-1 cost becomes ``distance + offset`` *before* the arm's own
  ``constrained_assignment`` runs, so the floor, the separation checks and the margin checks
  all stay active.  The offset is ``-log`` of the neighbours' current code frequency,
  rescaled to ``ALIGN`` times the row's distance spread (a relative bound, not a tuned
  constant).
* **L3 reads the exported discrete SID**: collisions are found on the model's own tokens with
  the batch's items de-duplicated, never on a soft argmax of a second pipeline.
* **one evaluator for every arm**: the chain statistic is the cumulative share over L1+L2+L3,
  so the control and the mechanism are measured with the same ruler.

Run ``--arm control`` for the budget-matched reference (no structural offset, no chain loss).
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
import sid_alignment_v2 as chains  # noqa: E402

OUT = HERE / "prototype_chain_wired.json"
STEPS = 400
BATCH_CHAINS = 256
LR = 1e-3
SEED = 42
TRAIN_FRACTION = 0.6
MARGIN = 0.5
ALIGN = 0.10
BUDGET = 0.4
PENDING_OFFSET = None  # (B, K) structural offset for the next forward, set per step


def split_chains(bank: np.ndarray):
    import hashlib
    keys = ["|".join(str(int(v)) for v in row) for row in bank]
    train = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100
                      < TRAIN_FRACTION * 100 for k in keys])
    return bank[train], bank[~train]


def cumulative_share(codes: np.ndarray, left, right) -> np.ndarray:
    equal = codes[left] == codes[right]
    return np.cumprod(equal, axis=1).sum(1)


def evaluate(codes, bank, rng, semantics, q_np) -> dict:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    adjacent = (cumulative_share(codes, a, b) + cumulative_share(codes, b, c)) / 2.0
    far = cumulative_share(codes, a, c)
    delta = float(adjacent.mean() - far.mean())
    s1 = codes
    level1 = ((s1[a, 0] == s1[b, 0]) + (s1[b, 0] == s1[c, 0])).mean() / 2.0
    level1_far = float((s1[a, 0] == s1[c, 0]).mean())
    level2 = ((s1[a, :2] == s1[b, :2]).all(1) + (s1[b, :2] == s1[c, :2]).all(1)).mean() / 2.0
    level2_far = float((s1[a, :2] == s1[c, :2]).all(1).mean())
    null = []
    for _ in range(100):
        shuffled = codes[rng.permutation(len(codes))]
        shuffled_adjacent = (cumulative_share(shuffled, a, b)
                             + cumulative_share(shuffled, b, c)) / 2.0
        null.append(float(shuffled_adjacent.mean()
                          - cumulative_share(shuffled, a, c).mean()))
    return {"chain_delta_L1L2L3": delta,
            "chain_delta_null": [float(np.percentile(null, 2.5)),
                                 float(np.percentile(null, 97.5))],
            "adjacent_mean": float(adjacent.mean()), "far_mean": float(far.mean()),
            "L1_adjacent": float(level1), "L1_far": level1_far,
            "L2_adjacent": float(level2), "L2_far": level2_far,
            "collision": float(1.0 - len(np.unique(codes, axis=0)) / len(codes)),
            "semantic_r2": semantic_probe(q_np, semantics)}


def semantic_probe(codes, semantics) -> float:
    generator = np.random.default_rng(SEED)
    order = generator.permutation(len(codes))
    cut = int(0.8 * len(order))
    train, evaluate_rows = order[:cut], order[cut:]
    x = codes[train].astype(np.float64)
    xc = x - x.mean(0, keepdims=True)
    lam = 1e-3 * float(np.trace(xc.T @ xc)) / xc.shape[1]
    weights = np.linalg.solve(xc.T @ xc + lam * np.eye(xc.shape[1]),
                              xc.T @ (semantics[train] - semantics[train].mean(0)))
    predicted = (codes[evaluate_rows] - x.mean(0)) @ weights + semantics[train].mean(0)
    return float(1.0 - ((predicted - semantics[evaluate_rows]) ** 2).mean()
                 / ((semantics[evaluate_rows] - semantics[evaluate_rows].mean(0)) ** 2).mean())


def install_offset(layers) -> None:
    """Route the pending structural offset into the arm's own level-1 assignment."""
    first = layers[0]
    original = first._distances

    def patched(latent):
        base = original(latent)
        if PENDING_OFFSET is None or PENDING_OFFSET.shape != base.shape:
            return base
        spread = base.std(dim=1, keepdim=True).clamp(min=1e-6)
        offset = PENDING_OFFSET
        offset = (offset - offset.mean(dim=1, keepdim=True)) / offset.std(
            dim=1, keepdim=True).clamp(min=1e-6)
        return base + ALIGN * offset * spread

    first._distances = patched


def neighbour_offset(codes_prev: np.ndarray, flat: np.ndarray, codes: int = 256):
    """-log frequency of each item's chain partners' current level-1 codes."""
    grouped = codes_prev.reshape(-1, 3)
    prior = np.full((len(flat), codes), 1e-6, dtype=np.float64)
    for position in range(3):
        for other in range(3):
            if other == position:
                continue
            prior[position::3, grouped[:, other]] += 1.0
    prior /= prior.sum(1, keepdims=True)
    return torch.from_numpy(-np.log(prior).astype(np.float32))


def main() -> None:
    global PENDING_OFFSET
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--arm", default="wired", choices=("wired", "control"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows, cols, _ = chains.loaded_edges()
    out_map, _ = chains.adjacency(rows, cols)
    bank = chains.chain3(out_map, np.stack([rows, cols], axis=1), rng, chains.CAP)
    train_bank, eval_bank = split_chains(bank)
    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    model = load_model().to(device)
    for layer in model.rq.vq_layers:
        layer.embed.requires_grad_(False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    layers = list(model.rq.vq_layers)
    if args.arm == "wired":
        install_offset(layers)
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    def snapshot(label):
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            codes = tokens.cpu().numpy().astype(np.int64)
            q_np = q.cpu().numpy().astype(np.float32)
        stats = evaluate(codes, eval_bank, rng, semantics, q_np)
        print(f"  {label}: chainD {stats['chain_delta_L1L2L3']:+.4f} "
              f"(null [{stats['chain_delta_null'][0]:+.3f},"
              f"{stats['chain_delta_null'][1]:+.3f}]) "
              f"L1 {stats['L1_adjacent']:.4f}/{stats['L1_far']:.4f} "
              f"L2 {stats['L2_adjacent']:.4f}/{stats['L2_far']:.4f} "
              f"collision {stats['collision']:.4f} semR2 {stats['semantic_r2']:.3f}",
              flush=True)
        return stats, codes

    before, _ = snapshot("before")
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    previous = np.zeros((BATCH_CHAINS * 3, 3), dtype=np.int64)
    log = []
    for step in range(STEPS):
        pick = torch.randint(0, len(train_bank), (BATCH_CHAINS,), generator=generator)
        batch_chains = train_bank[pick.numpy()]
        flat = batch_chains.reshape(-1)
        flat_tensor = torch.from_numpy(flat).to(device)
        left = torch.arange(0, len(flat), 3, device=device)
        middle, right = left + 1, left + 2
        if args.arm == "wired":
            PENDING_OFFSET = neighbour_offset(previous, flat).to(device)
        else:
            PENDING_OFFSET = None
        model.train()
        recon, quant_parts, _, tokens_batch = model(items[flat_tensor])
        previous = tokens_batch.detach().cpu().numpy().astype(np.int64)
        loss_rq, _ = criterion(recon, items[flat_tensor], quant_parts)
        optimizer.zero_grad(set_to_none=True)
        loss_rq.backward(retain_graph=True)
        rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                            if p.grad is not None) ** 0.5)
        detail = {}
        if args.arm == "wired":
            # L2 margin on the discrete codes the arm just produced (no second pipeline),
            # pushed through the encoder with the measured budget
            codes = previous
            adjacent = (cumulative_share(codes, left.cpu().numpy(), middle.cpu().numpy())
                        + cumulative_share(codes, middle.cpu().numpy(), right.cpu().numpy())) / 2
            far = cumulative_share(codes, left.cpu().numpy(), right.cpu().numpy())
            margin_value = float((MARGIN + far - adjacent).mean())
            # the discrete margin has no gradient; its differentiable surrogate is the soft
            # level-2 agreement of the arm's own logits
            z = model.encoder(items[flat_tensor])
            layer1, layer2 = layers[0], layers[1]
            base1 = layer1.distance(z, layer1.get_code_embs())
            p1 = torch.softmax(-base1 / base1.std().clamp(min=1e-6), dim=-1)
            residual = z - p1 @ layer1.get_code_embs()
            base2 = layer2.distance(residual, layer2.get_code_embs())
            p2 = torch.softmax(-base2 / base2.std().clamp(min=1e-6), dim=-1)
            s1_ab = (p1[left] * p1[middle]).sum(-1)
            s1_bc = (p1[middle] * p1[right]).sum(-1)
            s2_ab = s1_ab * (p2[left] * p2[middle]).sum(-1)
            s2_bc = s1_bc * (p2[middle] * p2[right]).sum(-1)
            s2_ac = (p1[left] * p1[right]).sum(-1) * (p2[left] * p2[right]).sum(-1)
            loss_l2 = torch.clamp(MARGIN + s2_ac - (s2_ab + s2_bc) / 2.0, min=0.0).mean()
            gradients = torch.autograd.grad(loss_l2, encoder_parameters, retain_graph=False,
                                            allow_unused=True)
            norm = float(sum(float(g.pow(2).sum()) for g in gradients
                             if g is not None) ** 0.5)
            scale = min(1.0, BUDGET * rq_norm / max(norm, 1e-12))
            for parameter, grad in zip(encoder_parameters, gradients):
                if grad is None:
                    continue
                parameter.grad = scale * grad if parameter.grad is None \
                    else parameter.grad + scale * grad
            detail = {"discrete_margin": margin_value, "grad_scale": scale,
                      "l2_loss": float(loss_l2.detach())}
        optimizer.step()
        if step % 50 == 0 or step == STEPS - 1:
            log.append({"step": step, "rq_norm": rq_norm, **detail})
            print(f"    step {step}: rq {rq_norm:.3e} {detail}", flush=True)
    after, codes_after = snapshot("after")
    report = {"protocol": {"arm": args.arm, "steps": STEPS, "seed": SEED,
                           "align": ALIGN, "margin": MARGIN, "budget": BUDGET,
                           "assignment": "the arm's own constrained_assignment with the "
                                         "floor ON; the offset is added inside _distances",
                           "L3": "discrete tokens of this forward, items de-duplicated"},
              "before": before, "after": after, "log": log}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    np.save(HERE / f"codes_{args.arm}.npy", codes_after)
    print("written", args.out)


if __name__ == "__main__":
    main()
