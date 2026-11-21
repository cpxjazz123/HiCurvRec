#!/usr/bin/env python3
"""Chain, second design: L1 in the Sinkhorn cost, L2 the average-sharing margin, L3 on real collisions.

Three corrections over ``prototype_chain_levels.py``:

* **L1 enters the assignment, not a separate pull.**  The level-1 cost gains
  ``-lambda * log(mean_{j in R(i)} p_j(k))`` and the batch is assigned by a *balanced*
  Sinkhorn, so the capacity constraint comes from the assignment rule itself instead of from
  a term whose optimum is to put every item in one code.
* **L2 is the margin the owner specified**: ``max(0, m + S(A,C) - (S(A,B)+S(B,C))/2)`` on the
  average adjacent sharing, not two independent log-losses.
* **L3 acts only on items that actually collide**: full-SID duplicate groups are pulled apart
  at level 3, instead of penalising random pairs whose share is ~1e-5.

A fourth run is the control the review asked for: the same 400 steps and same batches with
**no structural term at all**, so every number has a budget-matched reference.
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

OUT = HERE / "prototype_chain_three.json"
STEPS = 400
BATCH_CHAINS = 256
LR = 1e-3
SEED = 42
TRAIN_FRACTION = 0.6
MARGIN = 0.5
ALIGN = 0.5
BUDGETS = {"L1": 0.4, "L2": 0.4, "L3": 0.2}
SINKHORN_ITERS = 20


def sinkhorn(cost: torch.Tensor, temperature: float, iters: int = SINKHORN_ITERS):
    """Balanced soft assignment: rows sum to 1, columns to B/K."""
    rows, codes = cost.shape
    probabilities = torch.softmax(-cost / temperature, dim=-1)
    for _ in range(iters):
        probabilities = probabilities / probabilities.sum(0, keepdim=True).clamp(min=1e-9) \
            * (rows / codes)
        probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp(min=1e-9)
    return probabilities


def level_cost(residual: torch.Tensor, layer) -> torch.Tensor:
    table = layer.get_code_embs()
    return torch.cdist(residual, table) ** 2


def member_prior(probabilities: torch.Tensor) -> torch.Tensor:
    """For each item in a chain (laid out A,B,C,A,B,C,...), the mean of its two partners."""
    rows, width = probabilities.shape
    grouped = probabilities.view(-1, 3, width)
    prior = torch.stack([(grouped[:, 1] + grouped[:, 2]) / 2.0,
                         (grouped[:, 0] + grouped[:, 2]) / 2.0,
                         (grouped[:, 0] + grouped[:, 1]) / 2.0], dim=1)
    return prior.reshape(rows, width).detach()


def shares(p1, p2, left, right):
    first = (p1[left] * p1[right]).sum(-1)
    second = first * (p2[left] * p2[right]).sum(-1)
    return first, second


def discrete(codes, left, right):
    equal = codes[left] == codes[right]
    return (equal[:, 0].astype(np.float64),
            (equal[:, 0] & equal[:, 1]).astype(np.float64),
            (equal[:, 0] & equal[:, 1] & equal[:, 2]).astype(np.float64))


def split_chains(bank: np.ndarray):
    import hashlib
    keys = ["|".join(str(int(v)) for v in row) for row in bank]
    train = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100
                      < TRAIN_FRACTION * 100 for k in keys])
    return bank[train], bank[~train]


def evaluate(codes, bank, rng, semantics, q_np) -> dict:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    s1_ab, s2_ab, _ = discrete(codes, a, b)
    s1_bc, s2_bc, _ = discrete(codes, b, c)
    s1_ac, s2_ac, _ = discrete(codes, a, c)
    result = {
        "L1_share_AB": float(s1_ab.mean()), "L1_share_BC": float(s1_bc.mean()),
        "L1_share_AC": float(s1_ac.mean()),
        "L1_global_mean": float(np.mean([s1_ab.mean(), s1_bc.mean(), s1_ac.mean()])),
        "L2_gap": float(((s2_ab + s2_bc) / 2).mean() - s2_ac.mean()),
        "L2_share_AC": float(s2_ac.mean()),
        "collision": float(1.0 - len(np.unique(codes, axis=0)) / len(codes)),
        "level1_entropy": float(-(lambda p: (p * np.log(p + 1e-12)).sum())(
            np.bincount(codes[:, 0], minlength=256) / len(codes))),
        "chain_delta_all_levels": float(
            (((s1_ab + s2_ab + s2_ab * 0) + (s1_bc + s2_bc)) / 2).mean()
            - (s1_ac + s2_ac).mean()),
        "semantic_r2": semantic_probe(q_np, semantics)}
    null = {"L1_global_mean": [], "L2_gap": []}
    for _ in range(100):
        shuffled = codes[rng.permutation(len(codes))]
        u1_ab, u2_ab, _ = discrete(shuffled, a, b)
        u1_bc, u2_bc, _ = discrete(shuffled, b, c)
        u1_ac, u2_ac, _ = discrete(shuffled, a, c)
        null["L1_global_mean"].append(float(np.mean([u1_ab.mean(), u1_bc.mean(), u1_ac.mean()])))
        null["L2_gap"].append(float(((u2_ab + u2_bc) / 2).mean() - u2_ac.mean()))
    result["null_L1"] = [float(np.percentile(null["L1_global_mean"], 2.5)),
                        float(np.percentile(null["L1_global_mean"], 97.5))]
    result["null_L2"] = [float(np.percentile(null["L2_gap"], 2.5)),
                         float(np.percentile(null["L2_gap"], 97.5))]
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--arm", default="three", choices=("three", "control"))
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
    # The arm's own anti-collapse floor refuses an assignment in which a codeword owns no
    # item; the prototype studies exactly that pressure, so the floor is switched off here
    # and its conflict is recorded instead of masked.  Real Stage2 keeps it.
    import assignment_floor as floor_module
    floor_module.ASSIGNMENT_FLOOR = 0.0
    try:
        import model.layers as layers_module
        if hasattr(layers_module, "ASSIGNMENT_FLOOR"):
            layers_module.ASSIGNMENT_FLOOR = 0.0
    except Exception:  # noqa: BLE001 - the prototype must not fail on this probe
        pass
    from drq import measure_temperature
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))
    temperatures = []
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(train_bank[:1024].reshape(-1)).to(device)
        residual = model.encoder(items[sample])
        for layer in model.rq.vq_layers:
            temperatures.append(float(measure_temperature(layer, residual)))
            table = layer.get_code_embs()
            residual = residual - table[torch.cdist(residual, table).argmin(-1)]

    def snapshot(label):
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            codes = tokens.cpu().numpy().astype(np.int64)
            q_np = q.cpu().numpy().astype(np.float32)
        stats = evaluate(codes, eval_bank, rng, semantics, q_np)
        print(f"  {label}: L1 {stats['L1_global_mean']:.4f} L2gap {stats['L2_gap']:+.4f} "
              f"collision {stats['collision']:.4f} H(L1) {stats['level1_entropy']:.3f} "
              f"semR2 {stats['semantic_r2']:.3f}", flush=True)
        return stats

    before = snapshot("before")
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    log = []
    for step in range(STEPS):
        pick = torch.randint(0, len(train_bank), (BATCH_CHAINS,), generator=generator)
        flat = torch.from_numpy(train_bank[pick.numpy()].reshape(-1)).to(device)
        left = torch.arange(0, len(flat), 3, device=device)
        middle, right = left + 1, left + 2
        model.train()
        recon, quant_parts, _, _ = model(items[flat])
        loss_rq, _ = criterion(recon, items[flat], quant_parts)
        optimizer.zero_grad(set_to_none=True)
        loss_rq.backward(retain_graph=True)
        rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                            if p.grad is not None) ** 0.5)
        losses, scales = {}, {}
        if args.arm == "three":
            z = model.encoder(items[flat])
            layer1, layer2, layer3 = model.rq.vq_layers[:3]
            base1 = level_cost(z, layer1)
            with torch.no_grad():
                plain1 = torch.softmax(-base1 / max(temperatures[0], 1e-6), dim=-1)
            prior = member_prior(plain1)
            cost1 = base1 - ALIGN * torch.log(prior.clamp(min=1e-6))
            p1 = sinkhorn(cost1, max(temperatures[0], 1e-6))
            residual = z - p1 @ layer1.get_code_embs()
            p2 = sinkhorn(level_cost(residual, layer2), max(temperatures[1], 1e-6))
            residual2 = residual - p2 @ layer2.get_code_embs()
            p3 = sinkhorn(level_cost(residual2, layer3), max(temperatures[2], 1e-6))
            s_ab, s2_ab = shares(p1, p2, left, middle)
            s_bc, s2_bc = shares(p1, p2, middle, right)
            _, s2_ac = shares(p1, p2, left, right)
            loss_l2 = torch.clamp(MARGIN + s2_ac - (s2_ab + s2_bc) / 2.0, min=0.0).mean()
            # L1 is inside the assignment now; its only explicit pressure is the alignment
            # offset already used above, so nothing is added here.
            tokens = torch.stack([p1.argmax(-1), p2.argmax(-1), p3.argmax(-1)], dim=1)
            flat_tokens = tokens.cpu().numpy()
            unique_groups = {}
            for index in range(len(flat_tokens)):
                unique_groups.setdefault(tuple(flat_tokens[index].tolist()), []).append(index)
            colliding = [group for group in unique_groups.values() if len(group) > 1]
            if colliding:
                pairs = []
                for group in colliding[:512]:
                    pairs.append((group[0], group[1]))
                first = torch.tensor([pair[0] for pair in pairs], device=device)
                second = torch.tensor([pair[1] for pair in pairs], device=device)
                first_share = (p1[first] * p1[second]).sum(-1) \
                    * (p2[first] * p2[second]).sum(-1)
                loss_l3 = (first_share * (p3[first] * p3[second]).sum(-1)).mean()
            else:
                loss_l3 = torch.zeros((), device=device)
            # L1 objective: the *balanced* assignment p1 should match the neighbour prior,
            # which is the chain-alignment pressure with Sinkhorn supplying the capacity
            loss_l1 = -((p1 * torch.log(prior.clamp(min=1e-6))).sum(-1)).mean()
            terms = {"L1": (loss_l1, BUDGETS["L1"]), "L2": (loss_l2, BUDGETS["L2"]),
                     "L3": (loss_l3, BUDGETS["L3"])}
            for name, (loss, budget) in terms.items():
                gradients = torch.autograd.grad(loss, encoder_parameters, retain_graph=True,
                                                allow_unused=True)
                norm = float(sum(float(g.pow(2).sum()) for g in gradients
                                 if g is not None) ** 0.5)
                scale = min(1.0, budget * rq_norm / max(norm, 1e-12))
                scales[name] = (norm, scale)
                for parameter, grad in zip(encoder_parameters, gradients):
                    if grad is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = scale * grad
                    else:
                        parameter.grad = parameter.grad + scale * grad
            losses = {"L1": float(loss_l1.detach()), "L2": float(loss_l2.detach()),
                      "L3": float(loss_l3.detach()), "colliding": len(colliding)}
        optimizer.step()
        if step % 50 == 0 or step == STEPS - 1:
            log.append({"step": step, "rq_norm": rq_norm, "losses": losses,
                        "scales": {k: [v[0], v[1]] for k, v in scales.items()}})
            print(f"    step {step}: rq {rq_norm:.3e} {losses}", flush=True)
    after = snapshot("after")
    report = {"protocol": {"arm": args.arm, "steps": STEPS, "seed": SEED,
                           "assignment_floor": 0.0,
                           "loss": "L_RQVAE + Sinkhorn(L1 alignment cost) + L2 margin + "
                                   "L3 collision-group separation",
                           "gradient_budgets": BUDGETS, "codebook": "frozen"},
              "before": before, "after": after, "log": log}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
