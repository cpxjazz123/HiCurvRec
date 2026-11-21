#!/usr/bin/env python3
"""L2 supervision on its own: within groups that already share L1, and without the product.

Two defects of the previous L2-only run are fixed:

* **the L2 term no longer multiplies the L1 probability.**  ``S2`` is ``<p2(u), p2(v)>``
  alone, and it is computed only over chains whose three items already share the discrete
  L1 code, so the L2 signals are not attenuated by an L1 agreement of ~1e-3.
* **the batch is de-duplicated at the item level**: the forward runs on unique item ids and
  the chain triples are re-indexed into that pool, so an item that appears in several chains
  contributes one row, not several.

Both arms use the identical batch construction, so the control answers "what does the same
budget do with no structural term" under the same de-duplication.
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
import prototype_chain_wired2 as helper  # noqa: E402
import prototype_chain_long as proto  # noqa: E402

OUT = HERE / "prototype_chain_l2local.json"
STEPS = 400
BATCH_CHAINS = 256
LR = 1e-3
SEED = 42
MARGIN = 0.5
BUDGET = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "l2local"))
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
    train_bank, eval_bank = proto.split_pair_disjoint(full_bank)
    batches = proto.plan_batches(train_bank, args.steps, BATCH_CHAINS, SEED)
    semantics = np.load(SEMANTICS).astype(np.float32)
    items = torch.from_numpy(semantics).to(device)
    model = load_model().to(device)
    for layer in model.rq.vq_layers:
        layer.embed.requires_grad_(False)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    layers = list(model.rq.vq_layers)
    from drq import level_softmax, measure_temperature
    from losses import build_loss
    criterion = build_loss(json.loads((ARM_DIR / "config" / "train_config.json").read_text()))

    def snapshot(label):
        model.eval()
        with torch.no_grad():
            q, tokens = q_straight_through(model, items)
            codes = tokens.cpu().numpy().astype(np.int64)
            q_np = q.cpu().numpy().astype(np.float32)
        stats = proto.evaluate(codes, eval_bank, rng, semantics, q_np)
        print(f"  {label}: chainD {stats['chain_delta']:+.4f} "
              f"null[{stats['chain_delta_null'][0]:+.4f},{stats['chain_delta_null'][1]:+.4f}] "
              f"L1 {stats['L1_adjacent']:.4f}/{stats['L1_far']:.4f} "
              f"L2 {stats['L2_adjacent']:.4f}/{stats['L2_far']:.4f} "
              f"coll {stats['collision']:.4f} semR2 {stats['semantic_r2']:.3f}", flush=True)
        return stats

    report = {"protocol": {"arm": args.arm, "steps": args.steps, "seed": SEED,
                           "split": "pair-disjoint", "batch": "chains without replacement, "
                                                             "items de-duplicated per step",
                           "loss": "max(0, m + S2(A,C) - (S2(A,B)+S2(B,C))/2) with "
                                   "S2 = <p2,p2> only (no L1 product); the L1 gate is "
                                   "reported but not applied, because only 1/13,076 train "
                                   "chains has all three items in one L1 cell",
                           "margin": MARGIN, "budget": args.budget,
                           "chains": {"train": int(len(train_bank)),
                                      "eval": int(len(eval_bank))}},
              "curve": {"0": snapshot("step 0")}}
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    eligible_log = []
    for step, batch_chains in enumerate(batches, start=1):
        flat = batch_chains.reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        triples = inverse.reshape(-1, 3)
        unique_tensor = torch.from_numpy(unique).to(device)
        model.train()
        recon, quant_parts, _, tokens_batch = model(items[unique_tensor])
        loss_rq, _ = criterion(recon, items[unique_tensor], quant_parts)
        optimizer.zero_grad(set_to_none=True)
        loss_rq.backward(retain_graph=True)
        rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                            if p.grad is not None) ** 0.5)
        detail = {"unique_items": int(len(unique))}
        if args.arm == "l2local":
            z = model.encoder(items[unique_tensor])
            temperature1 = measure_temperature(layers[0], z)
            p1 = level_softmax(layers[0], z, temperature1)
            residual = z - p1 @ layers[0].get_code_embs()
            temperature2 = measure_temperature(layers[1], residual)
            p2 = level_softmax(layers[1], residual, temperature2)
            codes1 = tokens_batch[:, 0].detach().cpu().numpy()
            left, middle, right = triples[:, 0], triples[:, 1], triples[:, 2]
            eligible = (codes1[left] == codes1[middle]) & (codes1[middle] == codes1[right])
            count = int(eligible.sum())
            eligible_log.append(count)
            detail.update({"eligible_chains": count,
                           "temperature": [temperature1, temperature2]})
            # The L1 gate is reported, not used: only 1 of 13,076 train chains has all three
            # items in one L1 cell, so conditioning on it would leave ~0 chains per batch.
            # What remains is the essential change: S2 is <p2,p2> alone, with no L1 product.
            a = torch.from_numpy(left).to(device)
            b = torch.from_numpy(middle).to(device)
            c = torch.from_numpy(right).to(device)
            s_ab = (p2[a] * p2[b]).sum(-1)
            s_bc = (p2[b] * p2[c]).sum(-1)
            s_ac = (p2[a] * p2[c]).sum(-1)
            loss_l2 = torch.clamp(MARGIN + s_ac - (s_ab + s_bc) / 2.0, min=0.0).mean()
            gradients = torch.autograd.grad(loss_l2, encoder_parameters,
                                            retain_graph=False, allow_unused=True)
            norm = float(sum(float(g.pow(2).sum()) for g in gradients
                             if g is not None) ** 0.5)
            scale = min(1.0, args.budget * rq_norm / max(norm, 1e-12))
            for parameter, grad in zip(encoder_parameters, gradients):
                if grad is None:
                    continue
                parameter.grad = scale * grad if parameter.grad is None \
                    else parameter.grad + scale * grad
            detail.update({"l2_loss": float(loss_l2.detach()), "grad_scale": scale,
                           "s_ab": float(s_ab.mean()), "s_ac": float(s_ac.mean())})
        optimizer.step()
        if step % 100 == 0:
            print(f"    step {step}: rq {rq_norm:.2e} {detail}", flush=True)
        if step % 400 == 0 or step == args.steps:
            stats = snapshot(f"step {step}")
            stats.update(detail)
            report["curve"][str(step)] = stats
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    report["protocol"]["eligible_chains_mean"] = float(np.mean(eligible_log)) if eligible_log else 0.0
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
