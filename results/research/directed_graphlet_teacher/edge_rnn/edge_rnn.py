#!/usr/bin/env python3
"""Autoregressive edge prediction: later decisions see the edges already chosen.

The transformer head scores every ordered pair independently.  A GraphRNN-style decoder
instead emits the six ordered pairs of a 3-node candidate (or twelve for 4 nodes) one at
a time, and each step reads the edges it has already committed to.  That is the only
difference worth testing: whether modelling the dependence between edges recovers whole
graphlets, which independent scoring did not (adjacency-exact stayed near zero).

Order: pairs are visited in a fixed canonical order, and at each step the model sees the
member tokens plus the committed adjacency of the previous steps.  During training the
*true* prefix is fed (teacher forcing); at inference the model's own arg-max prefix is
fed back.  Everything else -- labels, split, inputs, step budget, thresholds, controls --
is identical to `edge_reconstruction.py`, so the two are directly comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "edge_reconstruction"))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from edge_reconstruction import (FULL_GRAPH, HELD_ROWS, INSTANCES, STATES, STATES_META,
                                 build_rows, evaluate, sha256)  # noqa: E402

SEED = 0
STEPS = 6000
BATCH = 512
HIDDEN = 128
HEADS = 4
LAYERS = 2
TRAIN_ROWS = 200000
PAIR_ORDER = [(i, j) for i in range(4) for j in range(4) if i != j]   # canonical, width-aware


def build_model(in_dim: int, width: int):
    import torch
    import torch.nn as nn

    class EdgeRNN(nn.Module):
        """Encode the members, then score one pair at a time given the committed prefix."""

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
            self.gru = nn.GRU(3 * HIDDEN + width * width, HIDDEN, batch_first=True)
            # GRU state (HIDDEN) concatenated with the pair vector (3 * HIDDEN)
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

        def forward(self, tokens, width: int, target=None):
            """Return per-step logits.  `target` feeds the true prefix; None feeds back."""
            import torch
            h = self.encode(tokens)
            pairs = [(i, j) for i in range(width) for j in range(width) if i != j]
            batch = len(h)
            committed = torch.zeros((batch, width, width), dtype=torch.float32,
                                    device=h.device)
            state = None
            logits = []
            for position, (i, j) in enumerate(pairs):
                pair_vector = torch.cat([h[:, i], h[:, j], h[:, i] * h[:, j]], dim=-1)
                prefix = committed.reshape(batch, -1)
                step_input = torch.cat([pair_vector, prefix], dim=-1).unsqueeze(1)
                output, state = self.gru(step_input, state)
                score = self.step(torch.cat([output.squeeze(1), pair_vector], dim=-1))
                logits.append(score.squeeze(-1))
                if target is not None:
                    chosen = target[:, i, j]
                else:
                    chosen = (torch.sigmoid(score.squeeze(-1)) > 0.5).float()
                committed = committed.clone()
                committed[:, i, j] = chosen
            return torch.stack(logits, dim=-1), pairs

    return EdgeRNN()


def to_matrix(step_logits: np.ndarray, pairs, width: int) -> np.ndarray:
    batch = step_logits.shape[0]
    out = np.zeros((batch, width, width), dtype=np.float64)
    for index, (i, j) in enumerate(pairs):
        out[:, i, j] = step_logits[:, index]
    return out


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "edge_rnn.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {key: np.load(STATES)[key] for key in ("semantics", "z", "q")}
    report = {"inputs": {"instances": sha256(INSTANCES), "states": sha256(STATES),
                         "full_graph": sha256(FULL_GRAPH),
                         "checkpoint": json.loads(STATES_META.read_text())},
              "protocol": {"model": "GraphRNN-style autoregressive edge decoder",
                           "encoder": "same transformer as edge_reconstruction.py",
                           "feed": "true prefix while training, own arg-max prefix at test",
                           "optimisation": f"{STEPS} shuffled mini-batches of {BATCH}",
                           "pairs_per_instance": "6 at width 3, 12 at width 4",
                           "seed": SEED, "train_rows": TRAIN_ROWS, "held_rows": HELD_ROWS},
              "runs": {}, "controls": {}}

    for width in (3, 4):
        rows = build_rows(width)
        train_rows, held_rows = rows["train_rows"], rows["held_rows"]
        y_truth = rows["adjacency"][train_rows]
        report["runs"][str(width)] = {}
        for name in ("semantics", "z", "q"):
            torch.manual_seed(SEED)
            model = build_model(states[name].shape[1], width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            tokens = torch.from_numpy(
                states[name][rows["nodes"][train_rows]].astype(np.float32)).to(device)
            target = torch.from_numpy(y_truth.astype(np.float32)).to(device)
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            for _ in range(STEPS):
                model.train()
                pick = torch.randint(0, len(tokens), (BATCH,), generator=generator)
                logits, pairs = model(tokens[pick], width, target[pick])
                truth = torch.stack([target[pick][:, i, j] for i, j in pairs], dim=-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                held = torch.from_numpy(
                    states[name][rows["nodes"][held_rows]].astype(np.float32)).to(device)
                held_logits, pairs = model(held, width)
                train_logits, _ = model(tokens, width)
            probabilities = to_matrix(torch.sigmoid(held_logits).cpu().numpy(), pairs, width)
            train_probabilities = to_matrix(torch.sigmoid(train_logits).cpu().numpy(),
                                            pairs, width)
            train_scores = {t: evaluate(train_probabilities, y_truth,
                                        rows["family"][train_rows],
                                        rows["orbit"][train_rows], t)
                            for t in (0.3, 0.5, 0.7)}
            best = max(train_scores, key=lambda value: train_scores[value]["edge_f1"])
            report["runs"][str(width)][name] = {
                "threshold_chosen_on_train": best,
                "train_edge_f1": train_scores[best]["edge_f1"],
                "heldout": evaluate(probabilities, rows["adjacency"][held_rows],
                                    rows["family"][held_rows], rows["orbit"][held_rows], best)}
            row = report["runs"][str(width)][name]["heldout"]
            print(f"  w{width} {name} thr {best}: edgeF1 {row['edge_f1']:.3f} "
                  f"P {row['edge_precision']:.3f} R {row['edge_recall']:.3f} "
                  f"exact {row['adjacency_exact']:.4f} classifiable "
                  f"{row['classifiable_share']:.3f} family {row['family_accuracy_overall']:.3f} "
                  f"orbit {row['orbit_accuracy_overall']:.3f}", flush=True)
            tmp = Path(args.out + ".tmp")
            tmp.write_text(json.dumps(report, indent=1) + "\n")
            tmp.replace(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
