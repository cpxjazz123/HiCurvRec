#!/usr/bin/env python3
"""Capability ladder for the autoregressive edge decoder: A / B / C / D.

Before asking whether a 32-d quantised vector keeps structure, the decoder itself has to
be shown able to read structure:

  A  the true adjacency is given at every step     -> can the encoder+decoder use answers
  B  all but ONE ordered pair is given             -> can it infer the held-out pair
  C  no adjacency is given                         -> the full reconstruction the
                                                     earlier rounds reported
  D  C with random member embeddings               -> is the C advantage more than the
                                                     structure statistics alone

Leakage rule for B: the edge being predicted at step k is removed from the prefix fed at
step k, and the same removal is used at evaluation.  A mask channel tells the model which
pairs are given, so "removed" is never confused with "present and false".

Width 3, complete behaviour graph labels, the same 200k rows / 6000 steps / global orbit
ids as `edge_rnn.py`.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "edge_reconstruction"))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from edge_reconstruction import (FULL_GRAPH, HELD_ROWS, INSTANCES, STATES, STATES_META,
                                 build_rows, evaluate, sha256)  # noqa: E402

WIDTH = 3
SEED = 0
STEPS = 6000
BATCH = 512
HIDDEN = 128
HEADS = 4
LAYERS = 2
MODES = ("A_given_all", "B_leave_one_out", "C_embedding_only", "D_random_embedding")


def pairs_of(width: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(width) for j in range(width) if i != j]


def build_model(in_dim: int, width: int):
    import torch
    import torch.nn as nn

    class CapabilityRNN(nn.Module):
        """Set A adds a learned offset per directed pair to its own token; A and B use a
        fresh encoder with the shielded pair removed, so no recurrent state can have seen
        the answer for that pair."""

        """Encoder + GRU decoder; the prefix carries both given edges and a known mask."""

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
            # prefix = committed edges, then the same width*width "known" mask
            self.gru = nn.GRU(3 * HIDDEN + 2 * width * width, HIDDEN, batch_first=True)
            self.step = nn.Linear(4 * HIDDEN, 1)
            # a given adjacency enters through the encoder as a per-pair bias, so the
            # decoder never has to carry the answer inside its recurrent state
            self.edge_bias = nn.Parameter(torch.zeros(width, width, HIDDEN))

        def encode(self, tokens, given=None):
            import torch
            h = self.input(tokens)
            if given is not None:
                # (n, width, width) -> (n, width, hidden): the edges a token's owner is on
                # edge_bias is (width, width, hidden): the middle index is the other member
                bias = torch.einsum("nij,ijd->nid", given.float(), self.edge_bias)
                h = h + bias
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

        def forward(self, tokens, width: int, mode: str, target=None,
                    given: "torch.Tensor | None" = None):
            """One logit per ordered pair.

            A / B: the edge being asked about is removed from the input **before encoding**
            and every question is a fresh encoder pass, so neither the encoder nor the
            GRU state has ever seen that pair's truth.  A gives every other pair, B gives
            none; that is the only difference between them.
            C / D: no adjacency at all, the model's own arg-max choices are fed back.
            """
            import torch
            h = self.encode(tokens, given)
            batch = len(h)
            pairs = pairs_of(width)
            committed = torch.zeros((batch, width, width), dtype=torch.float32,
                                    device=h.device)
            known = torch.zeros((batch, width, width), dtype=torch.float32, device=h.device)
            state = None
            logits = []
            for (i, j) in pairs:
                pair_vector = torch.cat([h[:, i], h[:, j], h[:, i] * h[:, j]], dim=-1)
                step_input = torch.cat([pair_vector, committed.reshape(batch, -1),
                                        known.reshape(batch, -1)], dim=-1).unsqueeze(1)
                output, state = self.gru(step_input, state)
                score = self.step(torch.cat([output.squeeze(1), pair_vector], dim=-1))
                logits.append(score.squeeze(-1))
                chosen = (torch.sigmoid(score.squeeze(-1)) > 0.5).float()
                committed = committed.clone()
                committed[:, i, j] = chosen
                known = known.clone()
                known[:, i, j] = 1.0
            return torch.stack(logits, dim=-1), pairs

    return CapabilityRNN()


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "capability.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = build_rows(WIDTH)
    payload = np.load(STATES)
    tables = {"semantics": payload["semantics"], "z": payload["z"], "q": payload["q"]}
    rng = np.random.default_rng(SEED)
    tables["random"] = rng.normal(size=tables["z"].shape).astype(np.float32)
    # all-zero vectors: no identity to memorise, so any remaining structure is the prior
    tables["zero"] = np.zeros_like(tables["z"])
    report = {"inputs": {"instances": sha256(INSTANCES), "states": sha256(STATES),
                         "full_graph": sha256(FULL_GRAPH)},
              "labels": {"width": WIDTH, "isolation": rows["isolation"],
                         "dropped_for_item_overlap": rows["dropped_for_item_overlap"],
                         "train_rows": int(len(rows["train_rows"])),
                         "held_rows": int(len(rows["held_rows"]))},
              "protocol": {"steps": STEPS, "batch": BATCH, "seed": SEED,
                           "encoder": "2-layer, 4-head transformer over member tokens",
                           "decoder": "GRU over the given/committed prefix plus a known mask",
                           "modes": {"A_given_all": "every pair except the asked one is given",
                                     "B_leave_one_out": "same as A; the asked pair is the "
                                                        "only thing missing",
                                     "C_embedding_only": "no adjacency, own prefix fed back",
                                     "D_random_embedding": "C with random member vectors"},
                           "leakage_rule": "the asked pair is removed from both the prefix "
                                           "and the known mask at every step"},
              "runs": {}}

    train_rows, held_rows = rows["train_rows"], rows["held_rows"]
    y_truth = rows["adjacency"][train_rows]
    y_held = rows["adjacency"][held_rows]
    # B is evaluated on one cell at a time: the shielded pair is removed from the
    # encoder's input, every other pair is given, the state starts fresh
    plan = [("A_given_all", "z"), ("B_leave_one_out", "z"),
            ("C_embedding_only", "semantics"), ("C_embedding_only", "z"),
            ("C_embedding_only", "q"), ("D_random_embedding", "random"),
            ("E_zero_embedding", "zero")]
    def shielded(matrix, row, col):
        """Copy of an adjacency batch with one cell cleared per instance."""
        out = matrix.copy()
        out[:, row, col] = False
        return out

    def to_matrix(step_logits, pairs):
        out = np.zeros((len(step_logits), WIDTH, WIDTH), dtype=np.float64)
        values = torch.sigmoid(step_logits).cpu().numpy()
        for index, (i, j) in enumerate(pairs):
            out[:, i, j] = values[:, index]
        return out

    def run(mode: str, name: str) -> dict:
        torch.manual_seed(SEED)
        model = build_model(tables[name].shape[1], WIDTH).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tokens = torch.from_numpy(
            tables[name][rows["nodes"][train_rows]].astype(np.float32)).to(device)
        target = torch.from_numpy(y_truth.astype(np.float32)).to(device)
        generator = torch.Generator(device="cpu").manual_seed(SEED)

        for _ in range(STEPS):
            model.train()
            pick = torch.randint(0, len(tokens), (BATCH,), generator=generator)
            sub_target = y_truth[pick.tolist()]
            if mode in ("A_given_all", "B_leave_one_out"):
                # supervise every cell with its own shielded input: no cell ever sees
                # another cell's answer for itself
                losses = []
                for i, j in pairs_of(WIDTH):
                    shaped = None
                    if mode == "A_given_all":
                        shaped = torch.from_numpy(
                            shielded(sub_target.astype(np.float32), i, j)).to(device)
                    else:
                        shaped = torch.zeros((BATCH, WIDTH, WIDTH), dtype=torch.float32,
                                             device=device)
                    logits_cell, pairs_cell = model(tokens[pick], WIDTH, mode, given=shaped)
                    position = pairs_cell.index((i, j))
                    losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_cell[:, position],
                        torch.from_numpy(sub_target[:, i, j].astype(np.float32)).to(device)))
                loss = torch.stack(losses).mean()
            else:
                logits, pairs = model(tokens[pick], WIDTH, mode)
                truth = torch.stack([target[pick][:, i, j] for i, j in pairs], dim=-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            held_tokens = torch.from_numpy(
                tables[name][rows["nodes"][held_rows]].astype(np.float32)).to(device)
            train_tokens = tokens
            if mode in ("A_given_all", "B_leave_one_out"):
                def cell_matrix(base_tokens, truth_matrix):
                    out = np.zeros((len(base_tokens), WIDTH, WIDTH), dtype=np.float64)
                    for i, j in pairs_of(WIDTH):
                        shaped_np = (shielded(truth_matrix.astype(np.float32), i, j)
                                     if mode == "A_given_all"
                                     else np.zeros_like(truth_matrix, dtype=np.float32))
                        shaped = torch.from_numpy(shaped_np).to(device)
                        logits_cell, pairs_cell = model(base_tokens, WIDTH, mode,
                                                        given=shaped)
                        position = pairs_cell.index((i, j))
                        out[:, i, j] = torch.sigmoid(
                            logits_cell[:, position]).cpu().numpy()
                    return out
                held_prob = cell_matrix(held_tokens, y_held)
                train_prob = cell_matrix(train_tokens, y_truth)
            else:
                held_logits, pairs = model(held_tokens, WIDTH, mode)
                train_logits, _ = model(train_tokens, WIDTH, mode)
                held_prob = to_matrix(held_logits, pairs)
                train_prob = to_matrix(train_logits, pairs)
        scores = {t: evaluate(train_prob, y_truth, rows["family"][train_rows],
                              rows["orbit"][train_rows], t) for t in (0.3, 0.5, 0.7)}
        best = max(scores, key=lambda value: scores[value]["edge_f1"])
        held = evaluate(held_prob, y_held, rows["family"][held_rows],
                        rows["orbit"][held_rows], best)
        print(f"  {mode} {name}: thr {best} edgeF1 {held['edge_f1']:.3f} exact "
              f"{held['adjacency_exact']:.4f} classifiable "
              f"{held['classifiable_share']:.3f} family "
              f"{held['family_accuracy_overall']:.3f} orbit "
              f"{held['orbit_accuracy_overall']:.3f}", flush=True)
        return {"threshold_chosen_on_train": best, "train_edge_f1": scores[best]["edge_f1"],
                "heldout": held}

    for mode, name in plan:
        report["runs"][f"{mode}:{name}"] = run(mode, name)
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps(report, indent=1) + "\n")
        tmp.replace(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
