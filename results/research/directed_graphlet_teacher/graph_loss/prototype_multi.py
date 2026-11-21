#!/usr/bin/env python3
"""One rule per structure: chain, fork and merge supervised with their own margin.

Each rule states its own ordering of the pairwise L2 sharing, and they are applied together
so the run also shows whether the rules leak into each other:

* chain  A->B->C     : S2(A,B), S2(B,C) above S2(A,C)
* fork   A->{B,C}    : S2(B,C) above S2(A,B), S2(A,C)
* merge  {B,C}->A    : S2(B,C) above S2(B,A), S2(C,A)

S2 is ``<p2,p2>`` alone -- the L1 product that attenuated the earlier runs is gone -- and the
codebooks stay frozen with every anti-collapse guard on.  No L1 structural offset is used:
the ε-ball capacity question is separate and is not mixed into this comparison.
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
from inject_directed import ARM_DIR, SEMANTICS, load_model, q_straight_through  # noqa: E402
import sid_alignment_v2 as chains  # noqa: E402
import prototype_chain_wired2 as helper  # noqa: E402
import prototype_chain_long as proto  # noqa: E402

OUT = HERE / "prototype_multi.json"
STEPS = 400
CHAINS, FORKS, MERGES = 128, 64, 64
LR = 1e-3
SEED = 42
MARGIN = 0.5
BUDGET = 0.10


def eval_structure(codes, bank, rng, kind):
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    cumulative = lambda left, right: np.cumprod(codes[left] == codes[right], axis=1).sum(1)
    if kind == "chain":
        near = (cumulative(a, b) + cumulative(b, c)) / 2.0
        far = cumulative(a, c)
    elif kind == "fork":
        near = cumulative(b, c)
        far = (cumulative(a, b) + cumulative(a, c)) / 2.0
    else:
        near = cumulative(b, c)
        far = (cumulative(b, a) + cumulative(c, a)) / 2.0
    delta = float(near.mean() - far.mean())
    null = []
    for _ in range(100):
        shuffled = codes[rng.permutation(len(codes))]
        shuffled_cum = lambda left, right: np.cumprod(
            shuffled[left] == shuffled[right], axis=1).sum(1)
        if kind == "chain":
            near_s = (shuffled_cum(a, b) + shuffled_cum(b, c)) / 2.0
            far_s = shuffled_cum(a, c)
        elif kind == "fork":
            near_s = shuffled_cum(b, c)
            far_s = (shuffled_cum(a, b) + shuffled_cum(a, c)) / 2.0
        else:
            near_s = shuffled_cum(b, c)
            far_s = (shuffled_cum(b, a) + shuffled_cum(c, a)) / 2.0
        null.append(float(near_s.mean() - far_s.mean()))
    return {"delta": delta,
            "null": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "multi"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--budget", type=float, default=BUDGET)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows, cols, _ = chains.loaded_edges()
    out_map, in_map = chains.adjacency(rows, cols)
    edges = np.stack([rows, cols], axis=1)
    banks = {
        "chain": chains.chain3(out_map, edges, rng, chains.CAP),
        "fork": chains.fan(out_map, in_map, rng, chains.CAP, "fork"),
        "merge": chains.fan(out_map, in_map, rng, chains.CAP, "merge"),
    }
    train, evaluation = {}, {}
    for kind, bank in banks.items():
        tr, ev = proto.split_pair_disjoint(bank)
        train[kind], evaluation[kind] = tr, ev
    plans = {
        "chain": proto.plan_batches(train["chain"], args.steps, CHAINS, SEED),
        "fork": proto.plan_batches(train["fork"], args.steps, FORKS, SEED + 1),
        "merge": proto.plan_batches(train["merge"], args.steps, MERGES, SEED + 2),
    }
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
        stats = {"collision": float(1.0 - len(np.unique(codes, axis=0)) / len(codes)),
                 "semantic_r2": helper.semantic_probe(q_np, semantics)}
        for kind, bank in evaluation.items():
            stats[kind] = eval_structure(codes, bank, rng, kind)
        print(f"  {label}: " + " ".join(
            f"{kind} {stats[kind]['delta']:+.4f}"
            for kind in ("chain", "fork", "merge"))
              + f" coll {stats['collision']:.4f} semR2 {stats['semantic_r2']:.3f}",
              flush=True)
        return stats

    report = {"protocol": {"arm": args.arm, "steps": args.steps, "seed": SEED,
                           "batch": f"{CHAINS} chains + {FORKS} forks + {MERGES} merges, "
                                    f"sampled without replacement, items de-duplicated",
                           "loss": "three per-structure margins on S2 = <p2,p2> (no L1 "
                                   "product), margin 0.5, budget split evenly",
                           "margin": MARGIN, "budget": args.budget,
                           "chrome": "no L1 structural offset; guards on; codebook frozen"},
              "curve": {"0": snapshot("step 0")}}
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=LR, weight_decay=1e-4)
    encoder_parameters = list(model.encoder.parameters())
    for step in range(1, args.steps + 1):
        batch = np.concatenate([plans["chain"][step - 1], plans["fork"][step - 1],
                                plans["merge"][step - 1]], axis=0)
        kinds = (["chain"] * CHAINS) + (["fork"] * FORKS) + (["merge"] * MERGES)
        flat = batch.reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        triples = inverse.reshape(-1, 3)
        unique_tensor = torch.from_numpy(unique).to(device)
        model.train()
        recon, quant_parts, _, _ = model(items[unique_tensor])
        loss_rq, _ = criterion(recon, items[unique_tensor], quant_parts)
        optimizer.zero_grad(set_to_none=True)
        loss_rq.backward(retain_graph=True)
        rq_norm = float(sum(float(p.grad.pow(2).sum()) for p in encoder_parameters
                            if p.grad is not None) ** 0.5)
        detail = {"unique_items": int(len(unique))}
        if args.arm == "multi":
            z = model.encoder(items[unique_tensor])
            temperature = measure_temperature(layers[0], z)
            p1 = level_softmax(layers[0], z, temperature)
            residual = z - p1 @ layers[0].get_code_embs()
            temperature2 = measure_temperature(layers[1], residual)
            p2 = level_softmax(layers[1], residual, temperature2)
            a = torch.from_numpy(triples[:, 0]).to(device)
            b = torch.from_numpy(triples[:, 1]).to(device)
            c = torch.from_numpy(triples[:, 2]).to(device)
            share = lambda left, right: (p2[left] * p2[right]).sum(-1)
            s_ab, s_bc, s_ac = share(a, b), share(b, c), share(a, c)
            index = 0
            losses = []
            for kind in ("chain", "fork", "merge"):
                count = kinds.count(kind)
                slice_ = slice(index, index + count)
                index += count
                if kind == "chain":
                    hinge = torch.clamp(MARGIN + s_ac[slice_]
                                        - (s_ab[slice_] + s_bc[slice_]) / 2.0, min=0.0)
                elif kind == "fork":
                    hinge = torch.clamp(MARGIN + (s_ab[slice_] + s_ac[slice_]) / 2.0
                                        - s_bc[slice_], min=0.0)
                else:
                    # merge triple is (B, C, centre): siblings are A-B, parents are A-C, B-C
                    hinge = torch.clamp(MARGIN + (s_ac[slice_] + s_bc[slice_]) / 2.0
                                        - s_ab[slice_], min=0.0)
                losses.append(hinge.mean())
            loss_struct = torch.stack(losses).sum()
            gradients = torch.autograd.grad(loss_struct, encoder_parameters,
                                            retain_graph=False, allow_unused=True)
            norm = float(sum(float(g.pow(2).sum()) for g in gradients
                             if g is not None) ** 0.5)
            scale = min(1.0, args.budget * rq_norm / max(norm, 1e-12))
            for parameter, grad in zip(encoder_parameters, gradients):
                if grad is None:
                    continue
                parameter.grad = scale * grad if parameter.grad is None \
                    else parameter.grad + scale * grad
            detail.update({"struct_loss": float(loss_struct.detach()),
                           "grad_scale": scale,
                           "hinges": [float(value) for value in losses]})
        optimizer.step()
        if step % 100 == 0:
            print(f"    step {step}: rq {rq_norm:.2e} {detail}", flush=True)
        if step % 400 == 0 or step == args.steps:
            stats = snapshot(f"step {step}")
            stats.update(detail)
            report["curve"][str(step)] = stats
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
