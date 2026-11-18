"""Make the codebook follow the encoder's distribution instead of holding still.

Round 5 established the asymmetry: the encoder must move to learn, but a codeword that
loses its items receives exactly zero gradient (the codebook term detaches the residual),
so it can never come back. The drift experiments showed that restraining the encoder
either kills learning (an anchor) or merely postpones the collapse (a speed cap).

This module takes the other side of the asymmetry: every optimizer step, measure how far
the *assigned* codewords moved, and apply that displacement to the codewords that the
step could not touch.

    F1  dead-shift   : only the codewords with no item this step receive the mean move
    F2  rigid-shift  : every codeword receives it on top of its own gradient

Both are deterministic functions of the step itself -- there is no decay, no window and
nothing to tune -- and neither changes the assignment rule or the loss.
"""
from __future__ import annotations

import torch


def assigned_mean_move(before: list[torch.Tensor], after: list[torch.Tensor],
                       tokens: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Per level, the mean displacement of the codewords this batch actually selected."""
    moves = []
    with torch.no_grad():
        for level in range(levels):
            counts = torch.bincount(tokens[:, level], minlength=before[level].shape[0])
            alive = counts > 0
            if not bool(alive.any()):
                moves.append(torch.zeros_like(before[level][0]))
                continue
            step = (after[level] - before[level])[alive]
            moves.append(step.mean(dim=0))
    return moves


@torch.no_grad()
def apply_shift(layers, tokens: torch.Tensor, moves: list[torch.Tensor],
                include_assigned: bool) -> int:
    """Add the measured move to the codewords the step did not (or did) reach.

    Returns how many codewords were touched, which the telemetry records.
    """
    touched = 0
    for level, layer in enumerate(layers):
        codes = layer.get_code_embs()
        counts = torch.bincount(tokens[:, level], minlength=codes.shape[0])
        target = torch.ones_like(counts, dtype=torch.bool) if include_assigned else (counts == 0)
        if not bool(target.any()):
            continue
        codes[target] = codes[target] + moves[level]
        touched += int(target.sum())
    return touched


def snapshot(layers) -> list[torch.Tensor]:
    return [layer.get_code_embs().detach().clone() for layer in layers]
