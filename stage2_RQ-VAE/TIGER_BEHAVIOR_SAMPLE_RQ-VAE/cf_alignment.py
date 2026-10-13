"""E2 collaborative alignment: the quantised representation against a frozen CF table.

The LETTER arm this replaces aligns the quantised sum to the encoder's own output,
``align_loss(residuals, quantized_sum) = ||quantized_sum - residuals[0]||^2``. That
target carries no information the encoder does not already hold, so the term can
only pull the codebooks toward the encoder's current geometry. LETTER's own
``CF_loss`` has the same defect -- its positive label is ``arange(batch)``, so each
item is again scored against itself.

The original paper's recipe is the version that has an outside signal in it: score
the quantised representation against a per-item collaborative-filtering table. That
table is what ``scripts/build_cf_embedding.py`` produces, from train interactions
only, frozen for the whole run.

This module supplies the three pieces the plan asks for: a trainable projection so
the 32-dim quantised representation can be compared with the 64-dim CF table, an
InfoNCE objective whose positive is the item's OWN CF row and whose negatives are
the other items in the batch, and -- because the quantised representation is built
with the same straight-through codebook lookup the rest of the family uses -- a
gradient that reaches L1, L2 and L3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
CF_EMB_PATH = (
    ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/cf_item_emb.npy"
)
CF_DIM = 64
CF_PROJECTION_DIM = 64
CF_WEIGHT = 0.1
CF_TEMPERATURE = 0.07

# The three configurations the plan names, as additive components over the base
# behaviour supervision every arm in this family already carries.
ARM_COMPONENTS = {
    "CF_ALIGN": {"align": True, "diversity": False, "ranking": False},
    "CF_ALIGN_DIV": {"align": True, "diversity": True, "ranking": False},
    "CF_ALIGN_DIV_RANK": {"align": True, "diversity": True, "ranking": True},
}


def load_cf_table(path: Path = CF_EMB_PATH) -> torch.Tensor:
    """Return the frozen [items, CF_DIM] table, indexed by raw item id."""
    table = np.load(path)
    if table.ndim != 2 or table.shape[1] != CF_DIM:
        raise ValueError(f"CF table must be [items, {CF_DIM}], got {table.shape}")
    return torch.from_numpy(np.ascontiguousarray(table, dtype=np.float32))


class CFProjection(nn.Module):
    """Trainable map from the quantised representation into the CF space."""

    def __init__(self, in_dim: int, out_dim: int = CF_PROJECTION_DIM) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.projection(quantized)


def cf_alignment_loss(
    quantized: torch.Tensor,
    cf_targets: torch.Tensor,
    projection: nn.Module,
    *,
    weight: float = CF_WEIGHT,
    temperature: float = CF_TEMPERATURE,
) -> torch.Tensor:
    """InfoNCE against the frozen CF rows, one positive per row.

    ``quantized`` and ``cf_targets`` are row-aligned: row i is item i and its
    positive is row i of ``cf_targets``. Everything else in the batch is a
    negative, which is the form LETTER's paper uses.
    """
    if quantized.shape[0] != cf_targets.shape[0]:
        raise ValueError("quantized rows and CF rows must be row-aligned")
    projected = F.normalize(projection(quantized), dim=-1)
    targets = F.normalize(cf_targets, dim=-1)
    logits = projected @ targets.t() / float(temperature)
    labels = torch.arange(projected.shape[0], device=projected.device)
    return float(weight) * F.cross_entropy(logits, labels)


def aligned_rows(cf_targets: torch.Tensor) -> torch.Tensor:
    """Boolean mask of rows whose CF vector is not the zero row.

    The 31 items with no train interaction get a zero row out of the builder; they
    have no positive to be pulled toward, so they are dropped rather than matched
    against a meaningless target.
    """
    return cf_targets.abs().sum(dim=-1) > 0.0
