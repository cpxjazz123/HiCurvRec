#!/usr/bin/env python3
"""Hard negatives: the reverse of a true edge, so similarity alone cannot answer.

For a true edge A->B the sample set becomes

    A -> B   label 1   positive
    B -> A   label 0   hard negative   (same two items, opposite direction)
    A -> D   label 0   regular negative (degree-matched non-neighbour)

The pair (A, B) then appears twice with opposite labels, so a model that only learns "these
two items are related" scores 0.5 on the batch; only a direction-sensitive score can do
better.

Two defects of the previous rounds are addressed here as well:

* the pipeline's own run-to-run noise was 0.01-0.02, larger than every effect being
  compared, so this script turns on deterministic kernels and a fixed candidate sampler;
* the evaluation set is enlarged to every held-out one-way edge in both orders, which
  brings the probe's binomial standard error down to about 0.004.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import augment_gnn as base  # noqa: E402
from direction_arms import build_model  # noqa: E402

OUT = HERE / "hard_negatives.json"
STEPS = base.STEPS
BATCH = base.BATCH
ALPHA = base.ALPHA


def make_deterministic() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def build_samples(msg_rows, msg_cols, train_unit, n_items, with_hard: bool, seed: int):
    """Positive edges; hard negatives = the reverse pair; regular negatives degree-matched."""
    rng = np.random.default_rng(seed)
    directed = set(zip(msg_rows.tolist(), msg_cols.tolist()))
    positives = np.stack([msg_rows, msg_cols], axis=1)
    samples = [(int(a), int(b), 1.0) for a, b in positives.tolist()]
    if with_hard:
        # every one-way edge contributes its reverse as a negative with the same endpoints
        for a, b in positives.tolist():
            if (b, a) not in directed:
                samples.append((int(b), int(a), 0.0))
    in_degree = np.bincount(msg_cols, minlength=n_items)
    out_degree = np.bincount(msg_rows, minlength=n_items)
    degree = out_degree + in_degree
    buckets = np.argsort(degree)
    # regular negatives: pick an anchor, then a partner of comparable degree
    target = len(positives) // 2
    added = 0
    while added < target:
        a = int(rng.integers(0, n_items))
        position = int(np.searchsorted(degree[buckets], degree[a]))
        window = buckets[max(0, position - 64): position + 64]
        b = int(rng.choice(window))
        if a == b or (a, b) in directed or (b, a) in directed:
            continue
        samples.append((a, b, 0.0))
        added += 1
    rng.shuffle(samples)
    pairs = torch.tensor([[s[0], s[1]] for s in samples], dtype=torch.long, device="cpu")
    labels = torch.tensor([s[2] for s in samples], dtype=torch.float32, device="cpu")
    return pairs, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    make_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(base.FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    semantics = np.load(base.SEMANTICS).astype(np.float32)
    x = torch.from_numpy(semantics).to(device)
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    edge_index = torch.from_numpy(np.stack([msg_rows, msg_cols]).astype(np.int64)).to(device)
    edges = {"out": (edge_index[0], edge_index[1], None),
             "in": (edge_index[1], edge_index[0], None)}

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]

    report = {"split": {"message_edges": int(len(msg_rows)), "held_edges": int(len(held_rows)),
                        "held_one_way_for_probe": int(len(one_way))},
              "protocol": {"steps": STEPS, "batch": BATCH, "seed": base.SEED,
                           "aggregation": "unweighted mean",
                           "deterministic": "torch.use_deterministic_algorithms(True) and "
                                            "cudnn deterministic",
                           "negatives": {"regular": "degree-matched non-neighbour",
                                         "hard": "reverse of a true one-way edge"},
                           "probe": "every held-out one-way edge in both orders, "
                                    "item-disjoint, fixed initialisation"},
              "arms": {}}
    for arm in ("A_regular_negatives", "B_plus_hard_negatives"):
        pairs, labels = build_samples(msg_rows, msg_cols, train_unit, n_items,
                                      with_hard=arm.startswith("B"), seed=base.SEED)
        positive_share = float(labels.mean())
        torch.manual_seed(base.SEED)
        model = build_model(semantics.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(base.SEED)
        pairs_dev, labels_dev = pairs.to(device), labels.to(device)
        for step in range(STEPS):
            model.train()
            pick = torch.randint(0, len(pairs_dev), (BATCH,), generator=generator)
            out = model(x, edges, n_items, ALPHA)
            left, right = pairs_dev[pick, 0], pairs_dev[pick, 1]
            target = labels_dev[pick]
            forward_score = (out["s"][left] * out["t"][right]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward_score, target)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                (out["s2"][left] * out["t2"][right]).sum(-1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == STEPS - 1:
                print(f"    {arm} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, edges, n_items, ALPHA)
            fused = out["fused"].cpu().numpy().astype(np.float32)
            pair_t = torch.from_numpy(one_way).to(device)
            head = float((((out["s"][pair_t[:, 0]] * out["t"][pair_t[:, 1]]).sum(-1))
                          > ((out["s"][pair_t[:, 1]] * out["t"][pair_t[:, 0]]).sum(-1))
                          ).float().mean())
        probe = base.probe(fused, one_way, device, steps=2500)
        report["arms"][arm] = {"positive_share": positive_share,
                               "edge_head_direction": head,
                               "probe": probe["direction_accuracy"],
                               "probe_pairs_eval": probe["pairs_eval"]}
        print(f"  {arm}: positive share {positive_share:.3f} head {head:.4f} "
              f"probe {probe['direction_accuracy']:.4f} "
              f"(n={probe['pairs_eval']})", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    se = 0.5 / math.sqrt(report["arms"]["A_regular_negatives"]["probe_pairs_eval"])
    delta = (report["arms"]["B_plus_hard_negatives"]["probe"]
             - report["arms"]["A_regular_negatives"]["probe"])
    report["readings"] = {"se": round(se, 4), "B_minus_A": round(delta, 4),
                          "B_minus_A_sigma": round(delta / se, 2)}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["readings"], indent=1))
    print("written", args.out)


if __name__ == "__main__":
    main()
