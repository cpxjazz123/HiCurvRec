#!/usr/bin/env python3
"""Three-level chain supervision: L1 global, L2 local, L3 identity.

The owner's design, implemented on the chain A->B->C only:

    L = L_RQVAE + lambda_1 L_global + lambda_2 L_local + lambda_3 L_unique

* ``L_global`` pushes all three nodes of a chain to share L1 (coarse structure);
* ``L_local`` pushes the *adjacent* pairs to share L1+L2 while keeping the two-hop pair
  apart at that depth -- soft, so the model chooses which adjacent pair groups together
  instead of being forced to align both and thereby also the ends;
* ``L_unique`` keeps full three-code collisions down, since the first two levels now
  encourage sharing.

Each term gets its own measured gradient budget against the reconstruction gradient, so
"three losses" cannot silently turn into one dominant term.  Codebooks stay frozen and
every reported number is measured on the **discrete** SID.
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

OUT = HERE / "prototype_chain_levels.json"
STEPS = 400
BATCH_CHAINS = 256
LR = 1e-3
SEED = 42
TRAIN_FRACTION = 0.6
BUDGETS = {"global": 0.4, "local": 0.4, "unique": 0.2}
UNIQUE_PAIRS = 2048


def probabilities(model, items: torch.Tensor, temperatures):
    from drq import level_softmax
    residual = model.encoder(items)
    result = []
    for level, layer in enumerate(model.rq.vq_layers):
        if level >= 3:
            break
        temperature = temperatures[level] if temperatures[level] > 0 else 1.0
        probability = level_softmax(layer, residual, temperature)
        result.append(probability)
        residual = residual - probability @ layer.get_code_embs()
    return result


def shares(probabilities, left, right):
    """(share L1, share L1+L2, share L1+L2+L3) between paired rows."""
    first = (probabilities[0][left] * probabilities[0][right]).sum(-1)
    second = first * (probabilities[1][left] * probabilities[1][right]).sum(-1)
    third = second * (probabilities[2][left] * probabilities[2][right]).sum(-1)
    return first, second, third


def discrete(codes: np.ndarray, left, right):
    equal = codes[left] == codes[right]
    s1 = equal[:, 0].astype(np.float64)
    s2 = (equal[:, 0] & equal[:, 1]).astype(np.float64)
    s3 = (equal[:, 0] & equal[:, 1] & equal[:, 2]).astype(np.float64)
    return s1, s2, s3


def split_chains(bank: np.ndarray):
    import hashlib
    keys = ["|".join(str(int(v)) for v in row) for row in bank]
    train = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100
                      < TRAIN_FRACTION * 100 for k in keys])
    return bank[train], bank[~train]


def evaluate(codes: np.ndarray, bank: np.ndarray, rng, semantics, q_np) -> dict:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    s1_ab, s2_ab, s3_ab = discrete(codes, a, b)
    s1_bc, s2_bc, s3_bc = discrete(codes, b, c)
    s1_ac, s2_ac, s3_ac = discrete(codes, a, c)
    global_share = float(np.mean([s1_ab.mean(), s1_bc.mean(), s1_ac.mean()]))
    local_gap = float(((s2_ab + s2_bc) / 2).mean() - s2_ac.mean())
    collision = float(1.0 - len(np.unique(codes, axis=0)) / len(codes))
    result = {
        "L1_share_AB": float(s1_ab.mean()), "L1_share_BC": float(s1_bc.mean()),
        "L1_share_AC": float(s1_ac.mean()), "L1_global_mean": global_share,
        "L2_share_AB": float(s2_ab.mean()), "L2_share_BC": float(s2_bc.mean()),
        "L2_share_AC": float(s2_ac.mean()), "L2_local_gap": local_gap,
        "ordered_share": float((((s2_ab + s2_bc) / 2) > s2_ac).mean()),
        "collision": collision,
        "distinct_codes": [int(len(np.unique(codes[:, k]))) for k in range(3)],
        "semantic_r2": semantic_probe(q_np, semantics)}
    nulls = {"L1_global_mean": [], "L2_local_gap": []}
    for _ in range(200):
        shuffled = codes[rng.permutation(len(codes))]
        u1_ab, u2_ab, _ = discrete(shuffled, a, b)
        u1_bc, u2_bc, _ = discrete(shuffled, b, c)
        u1_ac, u2_ac, _ = discrete(shuffled, a, c)
        nulls["L1_global_mean"].append(float(np.mean([u1_ab.mean(), u1_bc.mean(), u1_ac.mean()])))
        nulls["L2_local_gap"].append(float(((u2_ab + u2_bc) / 2).mean() - u2_ac.mean()))
    result["null_L1"] = [float(np.percentile(nulls["L1_global_mean"], 2.5)),
                         float(np.percentile(nulls["L1_global_mean"], 97.5))]
    result["null_L2_gap"] = [float(np.percentile(nulls["L2_local_gap"], 2.5)),
                             float(np.percentile(nulls["L2_local_gap"], 97.5))]
    return result


def semantic_probe(codes: np.ndarray, semantics: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(codes))
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
    parser.add_argument("--variant", default="absolute", choices=("absolute", "relative"))
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--usage-budget", type=float, default=0.0)
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
    from drq import measure_temperature
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    def snapshot(label: str) -> dict:
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            codes = tokens.cpu().numpy().astype(np.int64)
            q_np = q.cpu().numpy().astype(np.float32)
        stats = evaluate(codes, eval_bank, rng, semantics, q_np)
        print(f"  {label}: L1 {stats['L1_global_mean']:.4f} "
              f"(AB {stats['L1_share_AB']:.4f} BC {stats['L1_share_BC']:.4f} "
              f"AC {stats['L1_share_AC']:.4f}) L2 gap {stats['L2_local_gap']:+.4f} "
              f"(AC share2 {stats['L2_share_AC']:.4f}) collision {stats['collision']:.4f} "
              f"semR2 {stats['semantic_r2']:.3f}", flush=True)
        return stats

    before = snapshot("before")
    temperatures = []
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(train_bank[:1024].reshape(-1)).to(device)
        residual = model.encoder(items[sample])
        for layer in model.rq.vq_layers:
            temperatures.append(float(measure_temperature(layer, residual)))
            table = layer.get_code_embs()
            residual = residual - table[torch.cdist(residual, table).argmin(-1)]

    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    log = []
    for step in range(STEPS):
        pick = torch.randint(0, len(train_bank), (BATCH_CHAINS,), generator=generator)
        flat = torch.from_numpy(train_bank[pick.numpy()].reshape(-1)).to(device)
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
        p1, p2, p3 = probabilities(model, items[flat], temperatures)
        first, second, third = shares((p1, p2, p3), left, middle)
        first_bc, second_bc, _ = shares((p1, p2, p3), middle, right)
        first_ac, second_ac, _ = shares((p1, p2, p3), left, right)

        eps = 1e-6
        chain_first = torch.stack([first, first_bc, first_ac]).mean()
        if args.variant == "relative":
            # absolute sharing is satisfiable by collapsing L1; compare against the same
            # statistic on random item pairs, which a collapsed L1 also raises
            random_left = torch.randint(0, len(flat), (UNIQUE_PAIRS,), device=device)
            random_right = torch.randint(0, len(flat), (UNIQUE_PAIRS,), device=device)
            keep_random = random_left != random_right
            random_first = shares((p1, p2, p3), random_left[keep_random],
                                  random_right[keep_random])[0]
            loss_global = torch.clamp(args.margin + random_first.mean() - chain_first,
                                      min=0.0)
        else:
            loss_global = -(torch.stack([first, first_bc, first_ac]).clamp(min=eps).log().mean())
        loss_local = -(torch.stack([second, second_bc]).clamp(min=eps).log().mean()) \
            - (1.0 - second_ac).clamp(min=eps).log().mean()
        pair_left = torch.randint(0, len(flat), (UNIQUE_PAIRS,), device=device)
        pair_right = torch.randint(0, len(flat), (UNIQUE_PAIRS,), device=device)
        keep = pair_left != pair_right
        _, _, third_pairs = shares((p1, p2, p3), pair_left[keep], pair_right[keep])
        loss_unique = third_pairs.mean()

        terms = {"global": (loss_global, BUDGETS["global"]),
                 "local": (loss_local, BUDGETS["local"]),
                 "unique": (loss_unique, BUDGETS["unique"])}
        if args.usage_budget > 0:
            marginal = p1.mean(0).clamp(min=1e-9)
            loss_usage = (marginal * marginal.log()).sum()  # minimise -> maximise entropy
            terms["usage"] = (loss_usage, args.usage_budget)
        scales = {}
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
        optimizer.step()
        if step % 50 == 0 or step == STEPS - 1:
            values = {name: float(loss.detach()) for name, (loss, _) in terms.items()}
            log.append({"step": step, "rq_norm": rq_norm, "losses": values,
                        "gradients": {name: {"norm": value[0], "scale": value[1]}
                                      for name, value in scales.items()}})
            print(f"    step {step}: rq {rq_norm:.3e} "
                  + " ".join(f"{name} {value:.4f} (|g| {scales[name][0]:.2e} "
                             f"x{scales[name][1]:.4f})" for name, value in values.items()),
                  flush=True)

    after = snapshot("after")
    report = {"protocol": {
        "structure": "A->B->C strictly induced in the directed TRAIN graph",
        "chains": {"total": int(len(bank)), "train": int(len(train_bank)),
                   "eval": int(len(eval_bank))},
        "loss": "L_RQVAE + L_global(L1 all pairs) + L_local(L1+L2 adjacent vs AC) "
                "+ L_unique(full-SID agreement between random pairs)",
        "variant": args.variant, "margin": args.margin,
        "usage_budget": args.usage_budget,
        "gradient_budgets": BUDGETS, "codebook": "frozen", "steps": STEPS, "seed": SEED},
        "before": before, "after": after, "log": log}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
