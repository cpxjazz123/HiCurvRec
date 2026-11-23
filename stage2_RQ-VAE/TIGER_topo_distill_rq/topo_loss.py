"""Encoder-only topology distillation for issue #244 §5.

    L_topo = mean over the frozen pair bank of Huber( (1 + cos(z_i, z_j)) / 2 - S_G(i,j) )

``z`` is the encoder's pre-quantisation 32-dim output, so the term reaches the encoder and
nothing else.  ``S_G`` is the frozen teacher target: S_G = (S_geo + C_ctx)/2 with
S_geo = exp(-D_geo/tau) and C_ctx the Bhattacharyya overlap of the visible graph's directed
transition distributions; pairs whose context evidence is missing keep S_geo alone.

Fixed before training (CONTRACT §9, issue #244):

* ``DELTA = 0.1`` -- on a [0,1] target the quadratic region covers a tenth of the range, so
  the weak-evidence group cannot dominate;
* ``EPS = 1e-8`` -- cosine denominator floor; a zero-norm pair contributes 0 instead of NaN;
* pair batch 2048, sampler seed 243401, drawn from its own generator so the training RNG
  stream is untouched;
* weight 0 while ``epoch <= RAMP_START``, then linear to ``LAMBDA_MAX`` at ``EPOCHS``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RAMP_START = 1500
DELTA = 0.1
EPS = 1e-8
PAIR_BATCH = 2048
SAMPLER_SEED = 243401
#: filled by calibrate_topo.py before the formal run (CONTRACT §9)
LAMBDA_MAX = 0.0


class TopologyDistill:
    def __init__(self, embeddings: torch.Tensor, repo_root: Path, device, epochs: int,
                 bank_path: Path | None = None):
        bank = Path(bank_path) if bank_path is not None else (
            Path(repo_root) / "results" / "research" / "issue243" / "pairs" / "bank.npz")
        if not bank.exists():
            raise FileNotFoundError(f"missing frozen pair bank: {bank}")
        payload = np.load(bank)
        self.pairs = torch.tensor(payload["pairs"], dtype=torch.long)
        self.scores = torch.tensor(payload["scores"], dtype=torch.float32)
        self.embeddings = embeddings
        self.device = device
        self.epochs = int(epochs)
        self.generator = torch.Generator().manual_seed(SAMPLER_SEED)
        self.stats = {"zero_norm_pairs": 0, "steps": 0, "pairs_seen": 0, "pairs": int(len(self.pairs))}

    def weight(self, epoch: int) -> float:
        span = max(self.epochs - RAMP_START, 1)
        alpha = min(max((epoch - RAMP_START) / span, 0.0), 1.0)
        return LAMBDA_MAX * alpha

    def loss(self, encoder) -> torch.Tensor:
        index = torch.randint(0, len(self.pairs), (PAIR_BATCH,), generator=self.generator)
        pairs = self.pairs[index]
        scores = self.scores[index].to(self.device)
        left = self.embeddings[pairs[:, 0]].to(self.device, non_blocking=True)
        right = self.embeddings[pairs[:, 1]].to(self.device, non_blocking=True)
        z_left = encoder(left)
        z_right = encoder(right)
        dot = (z_left * z_right).sum(dim=1)
        norms = z_left.norm(dim=1) * z_right.norm(dim=1)
        zero = norms < EPS
        cosine = torch.where(zero, torch.zeros_like(dot), dot / norms.clamp(min=EPS))
        prediction = 0.5 * (1.0 + cosine)
        per_pair = F.huber_loss(prediction, scores, reduction="none", delta=DELTA)
        self.stats["zero_norm_pairs"] += int(zero.sum())
        self.stats["steps"] += 1
        self.stats["pairs_seen"] += int(pairs.shape[0])
        return per_pair.mean()
