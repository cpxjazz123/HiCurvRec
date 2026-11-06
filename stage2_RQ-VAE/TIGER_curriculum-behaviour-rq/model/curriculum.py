"""The three-stage curriculum: geometry first, then the behaviour, then both.

The diagnosis that motivates this arm: when a mechanism biases the level-1
ASSIGNMENT with a behaviour signal, the encoder's reconstruction gradient stops
requiring the latent to carry geometry, and the encoder collapses to a
low-rank cone (measured: the level-1 input's effective rank fell from 15.3 to
3.5 within fifty epochs, and the level-2 codebook then had nothing to
partition).  The curriculum keeps the quantizer's rule exactly as the baseline
has it - the assignment still follows the geometry - and introduces the
behaviour where it cannot break the quantizer: as a LOSS, on the latents, with a
weight that starts at zero.

Stage 1 (the first STAGE1_FRACTION of the epochs): the baseline's training,
exactly.  No behaviour term, no diversity term, the assignment rule untouched.
The representation distribution is allowed to become stable and to keep the
level-1/2/3 discrimination it has.

Stage 2 (the next STAGE2_FRACTION): the behaviour term is ramped in linearly
from zero to BEHAVIOUR_MAX, so the model adjusts the item relations gradually
instead of being reorganised in one step.

Stage 3 (the rest): reconstruction, behaviour alignment and representation
diversity are optimised together.

Every stage boundary and every weight is a stated constant; the behaviour signal
is the training split's directed successor pairs and nothing else, and the
monitoring reads the training split's own pairs.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

#: The share of the epochs that belong to stage 1 (geometry only).
STAGE1_FRACTION = 0.30
#: The share that belongs to stage 2 (the behaviour ramps in); stage 3 is the rest.
STAGE2_FRACTION = 0.40
#: The behaviour term's GRADIENT SHARE of the reconstruction's, at the end of the
#: ramp: the term's gradient is scaled to this fraction of the reconstruction's
#: gradient, so the number means what it says.  The share has to be stated in
#: gradients and not in loss values: measured at the initial weights, the
#: behaviour term's gradient is 57643 times the reconstruction's and the
#: diversity term's is 123702 times, so a weight read on the loss values would
#: have delivered a share of more than one without saying so.
BEHAVIOUR_MAX = 0.02
#: The diversity term's gradient share: the counterweight of the behaviour's.
DIVERSITY_MAX = 0.02
#: The InfoNCE temperature.
TEMPERATURE = 0.1


def stage_of(epoch: int, total: int) -> int:
    """Which stage an epoch belongs to: 1, 2 or 3."""
    total = max(int(total), 1)
    first = int(total * STAGE1_FRACTION)
    second = int(total * (STAGE1_FRACTION + STAGE2_FRACTION))
    if epoch <= first:
        return 1
    if epoch <= second:
        return 2
    return 3


def behaviour_weight(epoch: int, total: int, maximum: float = BEHAVIOUR_MAX) -> float:
    """The behaviour term's weight: zero in stage 1, ramped in stage 2, fixed."""
    total = max(int(total), 1)
    first = int(total * STAGE1_FRACTION)
    second = int(total * (STAGE1_FRACTION + STAGE2_FRACTION))
    if epoch <= first:
        return 0.0
    if epoch <= second:
        progress = (epoch - first) / max(second - first, 1)
        return float(maximum) * progress
    return float(maximum)


def diversity_weight(epoch: int, total: int, maximum: float = DIVERSITY_MAX) -> float:
    """The diversity term's weight: it ramps in WITH the behaviour term.

    The first version of this curriculum let the diversity term wait for stage 3,
    and a three-epoch smoke with the behaviour term on showed why that cannot
    work: the behaviour term pulls the latents together, the encoder answers by
    collapsing them (the level-1 input's effective rank fell from 8.5 to 2.2 in
    three epochs and the identifiers' collision rose from 0.23 to 0.89), and the
    positive pairs' sharing ran into the negative pairs' (0.568 against 0.534 -
    the codes carried no behaviour at all any more).  The diversity term is the
    counterweight, so it has to arrive with the behaviour, not after it.
    """
    return behaviour_weight(epoch, total, maximum)


def recover_sequences(frame) -> list:
    """The training sequence of every user: the longest prefix plus its target."""
    latest: dict[int, tuple] = {}
    for row in frame.itertuples(index=False):
        user = int(row.user)
        sequence = tuple(int(value) for value in row.history) + (int(row.target),)
        if user not in latest or len(sequence) > len(latest[user]):
            latest[user] = sequence
    return [latest[user] for user in sorted(latest)]


def successor_rows(frame, item_ids):
    """The train-only directed successor structure, as a CSR.

    Returns (offsets, successors, weights) over the items of ``item_ids`` (the
    training items, in the order given), so the successors of an item are the
    slice ``offsets[i]:offsets[i + 1]``.
    """
    position = {int(value): index for index, value in enumerate(item_ids)}
    counts: dict[int, dict[int, int]] = defaultdict(dict)
    for sequence in recover_sequences(frame):
        for left, right in zip(sequence, sequence[1:]):
            if left == right or left not in position or right not in position:
                continue
            counts[position[left]][position[right]] = counts[position[left]].get(position[right], 0) + 1
    offsets = [0]
    successors: list[int] = []
    weights: list[float] = []
    for index in range(len(item_ids)):
        row = counts.get(index, {})
        for target in sorted(row):
            successors.append(target)
            weights.append(float(row[target]))
        offsets.append(len(successors))
    return (np.asarray(offsets, dtype=np.int64), np.asarray(successors, dtype=np.int64),
            np.asarray(weights, dtype=np.float32))


def batch_successors(offsets, successors, weights, indices):
    """The successors of the batch's items, flattened.

    Returns (item_position, successor_index, weight) for the edges whose SOURCE
    is in the batch.  The edges are gathered from the CSR with one pass over the
    batch's items, which is a Python loop over the batch - the alternative, a
    dense mask over every edge, is slower at this size.
    """
    items, targets, values = [], [], []
    for position, item in enumerate(indices.tolist()):
        start, stop = int(offsets[item]), int(offsets[item + 1])
        if stop <= start:
            continue
        for offset in range(start, stop):
            items.append(position)
            targets.append(int(successors[offset]))
            values.append(float(weights[offset]))
    if not items:
        return (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                torch.zeros(0))
    return (torch.as_tensor(items, dtype=torch.long),
            torch.as_tensor(targets, dtype=torch.long),
            torch.as_tensor(values, dtype=torch.float32))


def behaviour_kernel(anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """The behaviour term's per-item kernel: the cosine to the successor.

    This is the part of the term the framework rebuilds symbolically; the term
    itself reduces it over the batch.
    """
    return F.cosine_similarity(anchor, positive, dim=-1)


def gradient_share_factor(reconstruction_loss: torch.Tensor, auxiliary_loss: torch.Tensor,
                          latent: torch.Tensor) -> torch.Tensor:
    """The factor that gives an auxiliary term a stated share of the gradient.

    The factor is the ratio of the reconstruction's gradient norm to the
    auxiliary term's, so multiplying the auxiliary term by it times the wanted
    share makes the auxiliary gradient that share of the reconstruction's.  It is
    measured once per epoch: the latents move inside the epoch, but slowly
    enough that the share holds.
    """
    recon_grad = torch.autograd.grad(reconstruction_loss, latent, retain_graph=True)[0]
    aux_grad = torch.autograd.grad(auxiliary_loss, latent, retain_graph=True)[0]
    recon_norm = recon_grad.detach().norm()
    aux_norm = aux_grad.detach().norm()
    return recon_norm / torch.clamp(aux_norm, min=1e-12)


def edge_sources(offsets, count: int) -> torch.Tensor:
    """The source item of every edge, in the CSR's order."""
    import numpy as _np

    if count <= 0:
        return torch.zeros(0, dtype=torch.long)
    device = torch.as_tensor(offsets).device
    bounds = torch.as_tensor(offsets).detach().cpu().numpy()
    source = _np.searchsorted(bounds, _np.arange(int(count)), side="right") - 1
    return torch.as_tensor(source, dtype=torch.long, device=device)


def info_nce(anchor: torch.Tensor, positive: torch.Tensor,
             temperature: float = TEMPERATURE) -> torch.Tensor:
    """The behaviour alignment term: a successor is closer than the others.

    The positives are the successors' snapshot latents - detached, refreshed once
    per epoch - and the negatives are the batch's own items.
    """
    if anchor.shape[0] < 2:
        return anchor.sum() * 0.0
    unit_anchor = F.normalize(anchor, dim=-1)
    unit_positive = F.normalize(positive, dim=-1)
    logits = unit_anchor @ unit_positive.t() / float(temperature)
    target = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, target)


def diversity_penalty(z: torch.Tensor) -> torch.Tensor:
    """The diversity term: the batch's directions must not crowd together.

    The latents are normalised first, so the term asks for the DIRECTIONS to
    spread and leaves the scale - which the quantizer and the commitment own -
    alone.
    """
    if z.shape[0] < 2:
        return z.sum() * 0.0
    unit = F.normalize(z, dim=-1)
    distance = torch.cdist(unit, unit).pow(2)
    rows = torch.arange(z.shape[0], device=z.device)
    distance = distance[rows.unsqueeze(1), rows.unsqueeze(0)]
    return torch.log(torch.exp(-2.0 * distance).mean())


def effective_rank(matrix: torch.Tensor) -> float:
    """The participation ratio of the covariance's eigenvalues."""
    if matrix.shape[0] < 2:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = (centered.t() @ centered) / float(centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float(total * total / float((eigenvalues * eigenvalues).sum()))


def pair_sharing(tokens: torch.Tensor, edges_item: torch.Tensor,
                 edges_successor: torch.Tensor, edges_negative: torch.Tensor) -> dict:
    """How the behaviour pairs share their level-1 code.

    The positives are the training split's successor pairs and the negatives the
    pairs the training split builds against a random item; the report is the
    share of positives whose level-1 code matches and of negatives whose
    level-1 code matches.  It reads the TRAINING split's own pairs: nothing
    held out is touched.
    """
    if edges_item.numel() == 0:
        return {"positive_sharing": 0.0, "negative_sharing": 0.0}
    positive = tokens[edges_item, 0] == tokens[edges_successor, 0]
    negative = tokens[edges_item, 0] == tokens[edges_negative, 0]
    return {"positive_sharing": float(positive.to(torch.float64).mean()),
            "negative_sharing": float(negative.to(torch.float64).mean())}
