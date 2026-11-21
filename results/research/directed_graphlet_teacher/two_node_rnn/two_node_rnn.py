#!/usr/bin/env python3
"""Stage 1 of the curriculum: a two-node GraphRNN, one edge at a time.

A pair of items is read as two tokens; the decoder then emits 2 *consult*  ordered edges in
sequence, and each decision reads the ones already taken:

    p(A->B), then p(B->A) given p(A->B)

Four outcomes are therefore representable -- (0,0), (1,0), (0,1), (1,1) -- instead of the
four-class softmax used before, and the second decision is explicitly conditional on the
first.  That is the smallest case of the curriculum the plan asks for, and the only one
trained in this round.

Inputs compared: 768-d semantics, z, q, a whitened PCA of z at width 32 (the "same width,
different metric" control), random vectors and all-zero vectors.  Every number is read
against the random and zero controls, because the class prior alone scores 0.53 accuracy.

Labels: complete behaviour graph, strict item isolation, labels asserted against the
adjacency before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE.parent / "pair_relations"))

import pair_relations as pr  # noqa: E402

SEED = 0
STEPS = 6000
BATCH = 512
HIDDEN = 128
HEADS = 4
LAYERS = 2
TRAIN_PAIRS = 300000
HELD_PAIRS = 30000


def build_model(in_dim: int):
    import torch
    import torch.nn as nn

    class TwoNodeRNN(nn.Module):
        """Two tokens, two sequential edge decisions, the second reading the first."""

        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(in_dim, HIDDEN)
            self.layers = nn.ModuleList(
                nn.ModuleDict({
                    "query": nn.Linear(HIDDEN, HIDDEN),
                    "key": nn.Linear(HIDDEN, HIDDEN),
                    "value": nn.Linear(HIDDEN, HIDDEN),
                    "out": nn.Linear(HIDDEN, HIDDEN),
                    "ff": nn.Sequential(nn.Linear(HIDDEN, 2 * HIDDEN), nn.GELU(),
                                        nn.Linear(2 * HIDDEN, HIDDEN))}) for _ in range(LAYERS))
            # step 0 sees no committed edge, step 1 sees one
            self.gru = nn.GRU(3 * HIDDEN + 1, HIDDEN, batch_first=True)
            self.step = nn.Linear(4 * HIDDEN, 1)

        def encode(self, tokens):
            import torch
            h = self.input(tokens)
            for layer in self.layers:
                q = layer["query"](h).view(len(h), -1, HEADS, HIDDEN // HEADS)
                k = layer["key"](h).view(len(h), -1, HEADS, HIDDEN // HEADS)
                v = layer["value"](h).view(len(h), -1, HEADS, HIDDEN // HEADS)
                scores = torch.einsum("nqhd,nkhd->nhqk", q, k) / (HIDDEN // HEADS) ** 0.5
                weights = torch.softmax(scores, dim=-1)
                attended = torch.einsum("nhqk,nkhd->nqhd", weights, v).reshape(
                    len(h), -1, HIDDEN)
                h = h + layer["out"](attended)
                h = h + layer["ff"](h)
            return h

        def forward(self, tokens, target=None):
            """Return one logit for A->B and one for B->A, in that order."""
            import torch
            h = self.encode(tokens)
            batch = len(h)
            committed = torch.zeros((batch, 1), dtype=torch.float32, device=h.device)
            state = None
            logits = []
            for step in range(2):
                i, j = (0, 1) if step == 0 else (1, 0)
                pair_vector = torch.cat([h[:, i], h[:, j], h[:, i] * h[:, j]], dim=-1)
                step_input = torch.cat([pair_vector, committed], dim=-1).unsqueeze(1)
                output, state = self.gru(step_input, state)
                score = self.step(torch.cat([output.squeeze(1), pair_vector], dim=-1))
                logits.append(score.squeeze(-1))
                if target is not None:
                    chosen = target[:, step]
                else:
                    chosen = (torch.sigmoid(score.squeeze(-1)) > 0.5).float()
                committed = chosen.unsqueeze(-1)
            return torch.stack(logits, dim=-1)

    return TwoNodeRNN()


def labels_of(pairs: np.ndarray, edges: set) -> np.ndarray:
    return np.stack([[(a, b) in edges for a, b in pairs.tolist()],
                     [(b, a) in edges for a, b in pairs.tolist()]], axis=1).astype(np.float32)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "two_node_rnn.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pr.EDGES, pr.BEHAVIOUR, positive = pr.load_graph()
    pr.DEGREE_EDGES = np.quantile(pr.BEHAVIOUR, np.linspace(0, 1, pr.DEGREE_CALIPERS + 1))
    pr.UNIT_EDGES = np.array(sorted({(min(a, b), max(a, b)) for a, b in pr.EDGES}),
                             dtype=np.int64)
    rng = np.random.default_rng(SEED)
    items = np.arange(len(pr.BEHAVIOUR))
    rng.shuffle(items)
    cut = int(0.8 * len(items))
    train_items = np.sort(items[:cut])
    held_items = np.sort(items[cut:])

    def pairs_within(pool: np.ndarray, target: int, seed: int) -> tuple:
        """Sample ordered pairs entirely inside one item pool, both orientations."""
        local = np.zeros(len(pr.BEHAVIOUR), dtype=bool)
        local[pool] = True
        allowed = [item for item in pool.tolist()]
        edges = {(a, b) for a, b in pr.EDGES if local[a] and local[b]}
        unit = np.array([p for p in pr.UNIT_EDGES.tolist() if local[p[0]] and local[p[1]]],
                        dtype=np.int64)
        rng_local = np.random.default_rng(seed)
        rows = []
        for a, b in unit[rng_local.permutation(len(unit))][: target // 2].tolist():
            rows.append((a, b))
            rows.append((b, a))
        # degree-matched negatives inside the same pool
        buckets = pr.degree_calipers_within(local)
        while len(rows) < target:
            a = int(rng_local.choice(allowed))
            bucket = pr.bucket_of(a)
            candidates = [item for item in buckets[bucket] if item != a]
            if not candidates:
                continue
            b = int(rng_local.choice(candidates))
            if (a, b) in edges or (b, a) in edges:
                continue
            rows.append((a, b))
        rows = rows[:target]
        pairs = np.array(rows, dtype=np.int64)
        bad = int(sum(not (local[a] and local[b]) for a, b in rows))
        if bad:
            raise SystemExit(f"{bad} pairs leave the pool")
        return pairs, edges

    train_pairs, train_edges = pairs_within(train_items, TRAIN_PAIRS, SEED)
    held_pairs, held_edges = pairs_within(held_items, HELD_PAIRS, SEED + 1)
    train_y = labels_of(train_pairs, train_edges)
    held_y = labels_of(held_pairs, held_edges)
    train_edges_set = set(map(tuple, train_pairs.tolist())) | {
        (b, a) for a, b in train_pairs.tolist()}
    if any((a, b) in train_edges_set for a, b in held_pairs.tolist()):
        raise SystemExit("held-out pair duplicates a training pair direction")

    payload = np.load(pr.STATES)
    tables = {"semantics": payload["semantics"], "z": payload["z"], "q": payload["q"]}
    from sklearn.decomposition import PCA
    tables["z_pca32"] = PCA(n_components=32, whiten=True,
                            random_state=SEED).fit_transform(payload["z"]).astype(np.float32)
    tables["random"] = rng.normal(size=payload["z"].shape).astype(np.float32)
    tables["zero"] = np.zeros_like(payload["z"])

    report = {"inputs": {"states": pr.sha256(pr.STATES), "full_graph": pr.sha256(pr.FULL_GRAPH)},
              "labels": {"train_pairs": int(len(train_pairs)),
                         "held_pairs": int(len(held_pairs)),
                         "train_edge_counts": {
                             "both": int(((train_y[:, 0] == 1) & (train_y[:, 1] == 1)).sum()),
                             "forward": int(((train_y[:, 0] == 1) & (train_y[:, 1] == 0)).sum()),
                             "backward": int(((train_y[:, 0] == 0) & (train_y[:, 1] == 1)).sum()),
                             "none": int(((train_y[:, 0] == 0) & (train_y[:, 1] == 0)).sum())},
                         "held_edge_counts": {
                             "both": int(((held_y[:, 0] == 1) & (held_y[:, 1] == 1)).sum()),
                             "forward": int(((held_y[:, 0] == 1) & (held_y[:, 1] == 0)).sum()),
                             "backward": int(((held_y[:, 0] == 0) & (held_y[:, 1] == 1)).sum()),
                             "none": int(((held_y[:, 0] == 0) & (held_y[:, 1] == 0)).sum())},
                         "isolation": "pairs sampled inside one item pool per split"},
              "protocol": {"decoder": "two sequential edges; step 1 reads step 0",
                           "steps": STEPS, "batch": BATCH, "hidden": HIDDEN, "seed": SEED,
                           "metrics": "per-edge accuracy, direction accuracy on one-way "
                                      "pairs, and 4-outcome exact match"},
              "runs": {}}

    def structure_accuracy(guess: np.ndarray, truth: np.ndarray) -> dict:
        forward = (truth[:, 0] == 1) & (truth[:, 1] == 0)
        backward = (truth[:, 0] == 0) & (truth[:, 1] == 1)
        one_way = forward | backward
        return {"edge_0_accuracy": float((guess[:, 0] == truth[:, 0]).mean()),
                "edge_1_accuracy": float((guess[:, 1] == truth[:, 1]).mean()),
                "exact_outcome": float((guess == truth).all(axis=1).mean()),
                "one_way_direction_accuracy": float((guess[one_way] == truth[one_way])
                                                    .all(axis=1).mean())
                if one_way.any() else None,
                "one_way_pairs": int(one_way.sum())}

    for name, table in tables.items():
        torch.manual_seed(SEED)
        model = build_model(table.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.from_numpy(table[train_pairs.reshape(-1)].reshape(
            len(train_pairs), 2, table.shape[1]).astype(np.float32)).to(device)
        y = torch.from_numpy(train_y).to(device)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        for _ in range(STEPS):
            model.train()
            pick = torch.randint(0, len(x), (BATCH,), generator=generator)
            logits = model(x[pick], y[pick])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[pick])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            held_x = torch.from_numpy(table[held_pairs.reshape(-1)].reshape(
                len(held_pairs), 2, table.shape[1]).astype(np.float32)).to(device)
            guess = (torch.sigmoid(model(held_x)) > 0.5).cpu().numpy().astype(np.float32)
        report["runs"][name] = structure_accuracy(guess, held_y)
        row = report["runs"][name]
        print(f"  {name}: edge0 {row['edge_0_accuracy']:.3f} edge1 "
              f"{row['edge_1_accuracy']:.3f} exact {row['exact_outcome']:.3f} "
              f"one-way direction {row['one_way_direction_accuracy']}", flush=True)
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps(report, indent=1) + "\n")
        tmp.replace(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
