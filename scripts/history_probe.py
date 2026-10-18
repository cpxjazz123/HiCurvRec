"""History-conditioned SID predictability probe (F1) for issue #48.

A small autoregressive probe trained from scratch, per SID arm, on train.parquet
only. It answers one question: does the user's item history predict the next item's
first SID token better than a train-frequency prior that sees the same information?

    probe   q_l(c_l | H, c_{<l})   H = last-20 history items mapped to their SIDs
    prior   p_l(c_l |     c_{<l})   train-frequency counts, fixed smoothing

Both sides see the same preceding TRUE target tokens, so any gain is information in
the history rather than the intrinsic low entropy of a shorter prefix. At level 1
there are no preceding tokens on either side, which is exactly why level 1 is the
primary feature.

The probe never receives the target item, the user id, any future step, or any
validation label. The encoder sees history SIDs only; the decoder head at level l is
handed the target's true earlier levels, which is ordinary teacher forcing and is
labelled as such in every emitted record.

CONFIG below is frozen before any Stage 3 outcome is inspected and is identical for
every arm and every seed. No arm-specific tuning, no pretrained Stage 3 weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODEBOOK_CAPACITY = 256
LEVELS = (1, 2, 3)

PRIMARY_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class ProbeConfig:
    """Frozen probe architecture and budget. Identical for every arm and seed."""

    seq_len: int = 20          # history truncation; 1 is the prespecified ablation
    token_embed_dim: int = 128
    gru_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    epochs: int = 6
    dev_user_fraction: float = 0.10
    grad_clip: float = 5.0
    label_smoothing: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def token_ids(sid: np.ndarray, item_ids: np.ndarray, level: int) -> np.ndarray:
    """Item id -> level-l token id, -1 for items absent from the catalogue."""
    out = np.full(sid.shape[0], -1, dtype=np.int64)
    valid = item_ids[(item_ids >= 0) & (item_ids < sid.shape[0])]
    out[valid] = sid[valid, level - 1]
    return out


def build_sequences(
    histories: list[np.ndarray],
    targets: np.ndarray,
    level_tokens: list[np.ndarray],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3, seq_len) history tokens per level and (N,) target token at level 1.

    The three levels get separate streams because the probe embeds each level with
    its own vocabulary and sums the three embeddings. Padding uses token id 0,
    which is a real codeword; the GRU sees a fixed-length window either way.
    """
    n = len(targets)
    seq = np.zeros((n, len(LEVELS), seq_len), dtype=np.int64)
    for i in range(n):
        h = histories[i]
        if len(h):
            tail = h[-seq_len:]
            for l in range(len(LEVELS)):
                seq[i, l, : len(tail)] = level_tokens[l][tail]
    return seq, level_tokens[0][targets]


class HistoryProbe(nn.Module):
    """History GRU plus one linear head per level over [hidden ; true earlier tokens]."""

    def __init__(self, cfg: ProbeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(CODEBOOK_CAPACITY, cfg.token_embed_dim)
        self.gru = nn.GRU(
            input_size=cfg.token_embed_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = nn.ModuleList(
            [
                nn.Linear(cfg.gru_hidden + cfg.token_embed_dim * (level - 1), CODEBOOK_CAPACITY)
                for level in LEVELS
            ]
        )
        self.input_project = nn.Linear(cfg.token_embed_dim, cfg.gru_hidden)

    def forward(
        self, seq: torch.Tensor, prior_tokens: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """``seq``: (B, 3, L) history token ids per level. ``prior_tokens``: the
        target's true earlier-level token ids, teacher forced; they are embedded
        here so the caller only ever passes ids."""
        embeds = [self.input_project(self.embed(seq[:, l, :])) for l in range(len(LEVELS))]
        summed = embeds[0] + embeds[1] + embeds[2]
        out, _ = self.gru(self.dropout(summed))
        hidden = out[:, -1, :]
        logits = []
        for i, level in enumerate(LEVELS):
            if level == 1:
                logits.append(self.heads[0](hidden))
            else:
                ctx = torch.cat(
                    [self.embed(prior_tokens[l]) for l in range(level - 1)], dim=-1
                )
                logits.append(self.heads[level - 1](torch.cat([hidden, ctx], dim=-1)))
        return logits


def train_frequency_prior(
    tokens_by_level: list[np.ndarray],
    targets: np.ndarray,
    smooth: float = 1.0,
) -> list[np.ndarray]:
    """p_l(c_l | c_{<l}) from train counts only, additive smoothing.

    ``targets`` are ITEM ids, and every train target is guaranteed to be present in
    the arm's catalogue, so the tables are exactly 256 wide. Level l needs only a
    table indexed by the l-1 earlier tokens.
    """
    priors: list[np.ndarray] = []
    for level in LEVELS:
        prefix_shape = (CODEBOOK_CAPACITY,) * (level - 1)
        counts = np.full(prefix_shape + (CODEBOOK_CAPACITY,), smooth, dtype=np.float64)
        if level == 1:
            np.add.at(counts, tokens_by_level[0][targets], 1.0)
        else:
            prefix = np.ravel_multi_index(
                tuple(tokens_by_level[l][targets] for l in range(level - 1)),
                prefix_shape,
            )
            np.add.at(
                counts.reshape(-1, CODEBOOK_CAPACITY),
                (prefix, tokens_by_level[level - 1][targets]),
                1.0,
            )
        priors.append(counts / counts.sum(axis=-1, keepdims=True))
    return priors


def evaluate_probe(
    model: HistoryProbe,
    seq: torch.Tensor,
    tokens_by_level: list[torch.Tensor],
    targets: torch.Tensor,
    idx: torch.Tensor,
    priors: list[torch.Tensor],
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Per-level NLL and top-B accuracy for the probe and the matched prior.

    The prior sees exactly the same conditioning as the probe: the target's true
    earlier levels, and nothing else. Evaluation is batched because a whole dev
    split through the GRU at once is tens of gigabytes.
    """
    model.eval()
    sums: dict[str, float] = {}
    total = int(idx.numel())
    with torch.no_grad():
        for start in range(0, total, batch_size):
            rows = idx[start : start + batch_size]
            logits_all = model(
                seq[rows], [tokens_by_level[l][rows] for l in range(len(LEVELS))]
            )
            for level in LEVELS:
                logits = logits_all[level - 1]
                tgt = tokens_by_level[level - 1][rows]

                # Index every conditioning level in ONE advanced-indexing call.
                # Chaining `prior[prior[c1]][c2]` would materialise a (B, 256, 256)
                # intermediate, which is gigabytes at this batch size.
                prior_rows = priors[level - 1]
                if level > 1:
                    cond = tuple(
                        tokens_by_level[l][rows].long() for l in range(level - 1)
                    )
                    prior_rows = prior_rows[cond]
                if level == 1:
                    prior_rows = prior_rows[tgt]

                vals = {
                    f"probe_ce_l{level}": F.cross_entropy(logits, tgt).item() * len(rows),
                    f"prior_ce_l{level}": -torch.log(prior_rows).sum().item(),
                }
                for b in (1, 5, 20):
                    vals[f"probe_top{b}_acc_l{level}"] = float(
                        (logits.topk(b, dim=-1).indices == tgt[:, None])
                        .any(dim=1)
                        .float()
                        .sum()
                        .item()
                    )
                    vals[f"prior_top{b}_acc_l{level}"] = float(
                        (prior_rows.topk(b, dim=-1).indices == tgt[:, None])
                        .any(dim=1)
                        .float()
                        .sum()
                        .item()
                    )
                for k, v in vals.items():
                    sums[k] = sums.get(k, 0.0) + v

    out = {k: v / total for k, v in sums.items()}
    for level in LEVELS:
        prior_nll = out[f"prior_ce_l{level}"]
        gain = prior_nll - out[f"probe_ce_l{level}"]
        out[f"gain_l{level}"] = gain
        out[f"rel_gain_l{level}"] = gain / prior_nll if prior_nll else float("nan")
    out["_dev_n"] = total
    return out
