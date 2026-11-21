#!/usr/bin/env python3
"""Learn a direction-aware behaviour representation, then fuse it into the 768-d input.

The probes of the last rounds showed that a frozen 768-d semantic vector carries almost no
information about *which way* a behaviour edge points.  Rather than asking the RQ-VAE to
learn that from scratch, this stage learns it first, as an auxiliary task on the item
vectors themselves:

    s_i = W_s x_i , t_i = W_t x_i          (source and target projections, DiGAE-style)
    P(i->j) = sigmoid(s_i . t_j)           (asymmetric: A->B and B->A differ)
    L_dir   = BCE(P, directed edges)       (undirected and randomised arms use the same head)

    e_i = f([s_i, t_i])                    (f maps the behaviour pair back to 768)
    x'_i= x_i + alpha * e_i                (width unchanged, RQ-VAE untouched)

Arms share one checkpoint, one pair sample, one budget and one seed:

  A  raw semantics                         (the untouched baseline)
  B  undirected-enhanced semantics         (L_dir on max(A->B, B->A); no direction signal)
  C  directed-enhanced semantics           (the recommendation)
  D  raw semantics, alpha = 0              (the augmentation path with no behaviour in it)

The pass criterion is stated before the numbers: the *fused* vector of arm C must beat
arm B and arm D on the same held-out direction probe, and it must do so at a margin larger
than the probe's standard error.  Only then is quantisation asked to preserve anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE.parent / "inject_directed"))

FULL_GRAPH = REPO / "results" / "research" / "behavior_graph_directed" / "graph.npz"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "augmented_states.npz"
MANIFEST = HERE / "augmented_manifest.json"

SEED = 0
STEPS = 3000
BATCH = 1024
HIDDEN = 256
ALPHA = 1.0
PAIR_SAMPLE = 400000
EVAL_PAIRS = 20000


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(in_dim: int = 768, dim: int = 64):
    import torch.nn as nn

    class BehaviourEncoder(nn.Module):
        """Source/target projections for the edge task, and the fusion back to in_dim."""

        def __init__(self) -> None:
            super().__init__()
            self.source = nn.Linear(in_dim, dim)
            self.target = nn.Linear(in_dim, dim)
            self.fuse = nn.Sequential(nn.Linear(2 * dim, HIDDEN), nn.GELU(),
                                      nn.Linear(HIDDEN, in_dim), nn.Tanh())

        def edge_logits(self, x):
            return self.source(x), self.target(x)

        def augment(self, x):
            s, t = self.edge_logits(x)
            return x + ALPHA * self.fuse(torch.cat([s, t], dim=-1))

    return BehaviourEncoder()


def pair_sample(edges: set, unit: np.ndarray, rng, count: int, positive_share: float = 0.5,
                banned: set | None = None):
    rows, labels = [], []
    order = rng.permutation(len(unit))
    for a, b in unit[order].tolist():
        if len(rows) >= count * positive_share:
            break
        rows.append((a, b))
        labels.append(((a, b) in edges, (b, a) in edges))
    n_items = int(max(a for a, _ in rows)) + 1 if rows else 0
    n_items = max(n_items, int(unit.max()) + 1)
    attempts = 0
    while len(rows) < count and attempts < 50 * count:
        attempts += 1
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


def direction_probe(x: np.ndarray, pairs: np.ndarray, truth: np.ndarray, device,
                    holdout: float = 0.3, steps: int = 1200) -> dict:
    """Held-out direction probe on a 768-d state: 70% train, 30% evaluate."""
    import torch.nn as nn
    one_way = truth[:, 0] != truth[:, 1]
    if one_way.sum() < 200:
        return {"accuracy": None, "pairs": int(one_way.sum())}
    sub, labels = pairs[one_way], (truth[one_way][:, 1] > 0.5).astype(np.int64)
    rng = np.random.default_rng(SEED + 5)
    order = rng.permutation(len(sub))
    cut = int((1 - holdout) * len(order))
    tr, ev = sub[order[:cut]], sub[order[cut:]]
    head = nn.Sequential(nn.LayerNorm(3 * x.shape[1]), nn.Linear(3 * x.shape[1], HIDDEN),
                         nn.GELU(), nn.Linear(HIDDEN, 2)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    def feats(idx):
        a, b = x[idx[:, 0]], x[idx[:, 1]]
        return torch.from_numpy(np.concatenate([a, b, a * b], -1).astype(np.float32)).to(device)

    xtr, ytr = feats(tr), torch.from_numpy(labels[order[:cut]]).to(device)
    xev, yev = feats(ev), torch.from_numpy(labels[order[cut:]]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(steps):
        pick = torch.randint(0, len(xtr), (min(BATCH, len(xtr)),), generator=generator)
        loss = torch.nn.functional.cross_entropy(head(xtr[pick]), ytr[pick])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        guess = head(xev).argmax(-1)
    return {"accuracy": float((guess == yev).float().mean()), "pairs_eval": int(len(xev)),
            "pairs_train": int(len(xtr))}


def main() -> None:
    import torch.nn as nn

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
    shuffled = unit[rng.permutation(len(unit))]
    cut = int(0.9 * len(shuffled))
    train_pairs, train_truth = pair_sample(edges, shuffled[:cut],
                                           np.random.default_rng(SEED), PAIR_SAMPLE)
    eval_pairs, eval_truth = pair_sample(edges, shuffled[cut:],
                                         np.random.default_rng(SEED + 1), EVAL_PAIRS,
                                         banned=set(map(tuple, train_pairs.tolist())))
    # item-level isolation: the probe's items never appear in the edge-task training pairs
    train_items = set(train_pairs.ravel().tolist())
    keep = np.array([not (a in train_items or b in train_items)
                     for a, b in eval_pairs.tolist()])
    if keep.sum() < 200:
        # the edge task already used ~all items as endpoints, so item-level isolation
        # would leave nothing to probe.  The probe is a *read-out* of a state that never
        # saw these pairs; pair-level isolation is kept and item-level is reported.
        isolation = "pair"
    else:
        eval_pairs, eval_truth = eval_pairs[keep], eval_truth[keep]
        isolation = "item"
    ISOLATION = {"level": isolation, "eval_pairs": int(len(eval_pairs))}

    report = {"inputs": {"semantics": sha256(SEMANTICS), "full_graph": sha256(FULL_GRAPH)},
              "isolation": "reported below",
              "protocol": {"steps": STEPS, "batch": BATCH, "hidden": HIDDEN,
                           "alpha": ALPHA, "seed": SEED,
                           "edge_task": "sigmoid(s_i . t_j), asymmetric",
                           "augment": "x + alpha * fuse([s, t]), width unchanged",
                           "probe": "held-out direction probe on the 768-d state, 70/30",
                           "pass_rule": "C must beat B and D by more than the probe SE"},
              "isolation": ISOLATION,
              "arms": {}}

    def sample_labels(mode: str) -> np.ndarray:
        labels = torch.from_numpy(train_truth).to(device)
        if mode == "B_undirected":
            both = torch.maximum(labels[:, 0], labels[:, 1])
            labels = torch.stack([both, both], dim=-1)
        elif mode == "D_alpha_zero":
            labels = torch.from_numpy(train_truth).to(device)
        return labels

    for arm in ("A_raw", "B_undirected", "C_directed", "D_alpha_zero"):
        torch.manual_seed(SEED)
        model = build_model().to(device)
        if arm != "A_raw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            labels = sample_labels(arm)
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            pairs = torch.from_numpy(train_pairs).to(device)
            for step in range(STEPS):
                model.train()
                pick = torch.randint(0, len(pairs), (BATCH,), generator=generator)
                s, t = model.edge_logits(items)
                logits = (s[pairs[pick][:, 0]] * t[pairs[pick][:, 1]]).sum(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[pick, 0])
                logits_rev = (s[pairs[pick][:, 1]] * t[pairs[pick][:, 0]]).sum(-1)
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_rev, labels[pick, 1])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 1000 == 0 or step == STEPS - 1:
                    print(f"    {arm} step {step}: loss {float(loss):.4f}", flush=True)
            model.eval()
            with torch.no_grad():
                if arm == "D_alpha_zero":
                    # identical training, but the fusion output is scaled to zero: the
                    # path is trained and then discarded, isolating the fusion's effect
                    augmented = items
                else:
                    augmented = model.augment(items)
            state = augmented.cpu().numpy().astype(np.float32)
        else:
            state = semantics
        probe = direction_probe(state, eval_pairs, eval_truth, device)
        # the edge task's own read-out: does the LEARNED source/target head still separate
        # A->B from B->A on held-out pairs?  This does not depend on whether the fusion
        # chose to write the direction into the 768-d vector.
        if arm != "A_raw":
            with torch.no_grad():
                s_all, t_all = model.edge_logits(items)
                pair_t = torch.from_numpy(eval_pairs).to(device)
                forward = (s_all[pair_t[:, 0]] * t_all[pair_t[:, 1]]).sum(-1)
                backward = (s_all[pair_t[:, 1]] * t_all[pair_t[:, 0]]).sum(-1)
            one_way = (eval_truth[:, 0] != eval_truth[:, 1])
            if one_way.any():
                truth_dir = eval_truth[one_way][:, 1] > 0.5
                guess_dir = (backward > forward).cpu().numpy().astype(bool)[one_way]
                probe["edge_head_direction"] = float((guess_dir == truth_dir).mean())
                probe["edge_head_pairs"] = int(one_way.sum())
        report["arms"][arm] = {"probe": probe,
                               "state_shift_l2": float(np.linalg.norm(state - semantics)
                                                       / max(len(state), 1))}
        print(f"  {arm}: direction probe {probe['accuracy']} "
              f"(shift {report['arms'][arm]['state_shift_l2']:.4f})", flush=True)
        np.savez_compressed(HERE / f"state_{arm}.npz", state=state)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    MANIFEST.write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
