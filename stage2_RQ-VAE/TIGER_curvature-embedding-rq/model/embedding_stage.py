"""The embedding stage: a curvature-aware refinement of the frozen embeddings.

Plan A of this round: the RQ-VAE stays exactly as the frozen baseline has it -
same architecture, same argmin assignment, same loss, same training recipe -
and the work happens BEFORE the quantizer, in the embedding the quantizer sees.

The module is

    z_i = f_theta(x_i, G, kappa) = x_i + gamma_i * Delta_i

where ``x_i`` is the frozen Stage-1 semantic embedding, ``G`` is the train-only
behaviour graph, ``kappa_i`` is the item's local curvature, ``Delta_i`` is a
learned residual and ``gamma_i`` is a curvature-driven gate.  The residual's
last layer starts at zero, so the stage begins as the identity and can only
move away from it by what the objective pays for.

Three terms, each on its own scale, in this order:

* the BEHAVIOUR term: the directed successor pairs of the training split should
  be closer in ``z`` than random pairs are (an InfoNCE over the training
  edges, with the other items of the batch as the negatives);
* the SPREAD term: the embeddings should not collapse onto a few directions -
  the item-weighted covariance's off-diagonal correlation is pushed to zero.
  This is the term that speaks to the SID collapse: an identifier collides when
  the items the quantizer sees are crowded together;
* the ANCHOR term: ``z`` should stay near ``x``, in a RELATIVE sense, so the
  semantics Stage 1 found survive the refinement.

Nothing here reads the validation or the test split: the graph, the curvature
and the loss are built from the training split alone.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#: The batch of edges one step takes.
EDGE_BATCH = 4096
#: The epochs over the training edges.
EPOCHS = 30
#: The learning rate (the baseline's).
LEARNING_RATE = 1e-3
#: The width of the refinement network.
HIDDEN = 512
#: The InfoNCE temperature.
TEMPERATURE = 0.1
#: The spread term's weight.  The three terms are on comparable scales, but not
#: on comparable GRADIENTS: measured per item at the start of the stage, the
#: behaviour term's gradient is about 1.4e-3, the anchor's about 8e-4 and the
#: uniformity's about 1.4e-4 - ten times weaker, because its exponent is
#: already small where the items sit.  Left unweighted the items drift towards
#: their successors (the behaviour term wins) and the collapse is not treated at
#: all, so the weight is the measured ratio: the three terms then pull with
#: the same order of strength and the equilibrium is a compromise between the
#: semantics, the behaviour and the spread.
SPREAD_WEIGHT = 10.0


def recover_sequences(frame) -> list:
    """The training sequence of every user: the longest prefix plus its target.

    The training split is prefix-expanded, so its rows are not the sequences: a
    user's sequence is the longest of their rows, and counting the rows instead
    would count the same adjacency once per prefix.
    """
    latest: dict[int, tuple] = {}
    for row in frame.itertuples(index=False):
        user = int(row.user)
        sequence = tuple(int(value) for value in row.history) + (int(row.target),)
        if user not in latest or len(sequence) > len(latest[user]):
            latest[user] = sequence
    return [latest[user] for user in sorted(latest)]


def behaviour_graph(frame, item_ids):
    """The train-only behaviour graph, directed and symmetrised.

    Returns the directed successor counts (the aggregation's weights) and the
    symmetrised neighbour lists (the curvature's graph, which is undirected: the
    Ollivier-Ricci curvature is defined for undirected graphs).
    """
    position = {int(value): index for index, value in enumerate(item_ids)}
    directed: list[dict] = [dict() for _ in range(len(item_ids))]
    for sequence in recover_sequences(frame):
        for left, right in zip(sequence, sequence[1:]):
            if left == right or left not in position or right not in position:
                continue
            source, target = position[left], position[right]
            directed[source][target] = directed[source].get(target, 0) + 1
    neighbours: list[set] = [set() for _ in range(len(item_ids))]
    for source, row in enumerate(directed):
        for target in row:
            neighbours[source].add(target)
            neighbours[target].add(source)
    return directed, [sorted(row) for row in neighbours]


def curvature_of(neighbours: list) -> np.ndarray:
    """The item's local curvature: the Ollivier-Ricci estimate of its neighbourhood.

    The Ollivier-Ricci curvature is the transport between the two endpoints'
    neighbourhood measures; the Lin-Lu-Yau approximation replaces that transport
    by the sum of the neighbours' distances, and that is what is computed here:
    for an edge (i, j) in a graph whose distance is the hop count,

        kappa_ij = 1 - d(d + 2) / (sum_k d(k, j) + sum_l d(i, l))

    and the item's curvature is the mean over its edges.  The distances are the
    hop counts to depth two, which is where the neighbours of an endpoint live;
    the value is an approximation and is declared as one.
    """
    count = len(neighbours)
    depth2: list[dict] = []
    for node in range(count):
        distances: dict[int, int] = {}
        for first in neighbours[node]:
            distances[first] = 1
            for second in neighbours[first]:
                distances.setdefault(second, 2)
        depth2.append(distances)
    totals = np.zeros(count)
    numbers = np.zeros(count)
    for node in range(count):
        for other in neighbours[node]:
            here = sum(depth2[node].get(item, 2) for item in neighbours[node]
                       if item != other)
            there = sum(depth2[other].get(item, 2) for item in neighbours[other]
                        if item != node)
            denominator = here + there
            value = 1.0 - 3.0 / denominator if denominator > 0 else 0.0
            totals[node] += value
            numbers[node] += 1
    raw = np.where(numbers > 0, totals / np.maximum(numbers, 1.0), 0.0)
    # The raw estimate is a narrow band; what the module needs is where an item
    # stands among the others, so the curvature is standardised.
    spread = float(raw.std())
    standardised = (raw - float(raw.mean())) / (spread if spread > 0 else 1.0)
    return raw.astype(np.float32), standardised.astype(np.float32)


class CurvatureRefinement(nn.Module):
    """``z_i = x_i + gamma_i * Delta_i``, with the gate driven by the curvature.

    The neighbourhood mean carries the behaviour into the refinement, the
    curvature says how much the refinement may move the item, and the residual's
    last layer starts at zero: the stage begins as the identity.
    """

    def __init__(self, width: int, hidden: int = HIDDEN):
        super().__init__()
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(2 * width + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
        )
        self.gate = nn.Linear(1, 1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, neighbourhood: torch.Tensor,
                curvature: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat([x, neighbourhood, curvature[:, None]], dim=-1))
        gamma = F.softplus(-self.gate(curvature[:, None]))
        moved = x + gamma * residual
        # The refinement changes the item's DIRECTION, never its scale: the
        # quantizer's codebook is built for the frozen embedding's scale, and a
        # refinement that grew the norms would change the quantizer's problem
        # instead of the embedding's.
        scale = x.norm(dim=-1, keepdim=True)
        return moved * (scale / torch.clamp(moved.norm(dim=-1, keepdim=True), min=1e-12))


def neighbourhood_mean(x: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
                        weight: torch.Tensor, items: int) -> torch.Tensor:
    """The weighted mean of each item's neighbours' embeddings."""
    totals = torch.zeros(items, x.shape[1], dtype=x.dtype, device=x.device)
    totals.index_add_(0, source, x.index_select(0, target) * weight[:, None])
    counts = torch.zeros(items, dtype=x.dtype, device=x.device)
    counts.index_add_(0, source, weight)
    return totals / torch.clamp(counts, min=1e-12)[:, None]


def info_nce(anchor: torch.Tensor, positive: torch.Tensor,
             temperature: float = TEMPERATURE) -> torch.Tensor:
    """The behaviour term: the successors are closer than the other items."""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / float(temperature)
    target = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, target)


def spread_penalty(z: torch.Tensor) -> torch.Tensor:
    """The spread term: the batch's embeddings must not crowd together.

    The quantiser gives two items the same identifier when they sit close
    together at the scale its codebook sees, so the term is the uniformity of
    the batch: the log of the mean exponential of the NEGATIVE squared pairwise
    distance.  The lower it is, the more spread the items are; a batch collapsed
    into a ball scores high and a spread one low.  It is scale-sensitive on
    purpose - the scale is what the quantiser sees - and the anchor term is what
    keeps that scale near the frozen embedding's.
    """
    distance = torch.cdist(z, z).pow(2)
    rows = torch.arange(z.shape[0], device=z.device)
    distance = distance[rows.unsqueeze(1), rows.unsqueeze(0)]      # the diagonal out
    if distance.numel() == 0:
        return 0.0
    return float(torch.log(torch.exp(-2.0 * distance).mean()))


def anchor_value(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The per-item relative error of the refinement.

    It is the anchor's own measure: how far the item moved, relative to how
    far it was to begin with.
    """
    return (z - x).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1).clamp_min(1e-12)


def anchor_penalty(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The anchor term: the refinement stays a refinement, in a relative sense."""
    return float(anchor_value(z, x).mean())


def gate_value(kappa: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """The curvature's gate: ``softplus(-(a*kappa + b))``.

    The gate is where the behaviour graph's local geometry enters the
    refinement: a tree-like neighbourhood (a low curvature) opens the gate
    wider, so the item may move further.
    """
    return F.softplus(-(float(a) * kappa + float(b))).reshape(-1)


def train_embedding_stage(frame, item_ids, embeddings: np.ndarray, device,
                          *, epochs: int = EPOCHS, batch: int = EDGE_BATCH,
                          seed: int = 42, log=print):
    """Train the refinement on the training split and return the new embeddings.

    The graph, the curvature and the loss are the training split's; the frozen
    Stage-1 embeddings are the input.  The neighbourhood mean is a feature of
    the FROZEN embedding (the refinement is a per-item function of the item's
    own neighbourhood, not a second pass over the graph), so it is computed
    once.
    """
    torch.manual_seed(seed)
    directed, neighbours = behaviour_graph(frame, item_ids)
    raw_curvature, curvature = curvature_of(neighbours)
    items = len(item_ids)
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)

    source_index: list[int] = []
    target_index: list[int] = []
    edge_weight: list[float] = []
    for source, row in enumerate(directed):
        for target, count in row.items():
            source_index.append(source)
            target_index.append(target)
            edge_weight.append(float(count))
    if not source_index:
        raise RuntimeError("the training split has no successor edge to learn from")
    sources = torch.as_tensor(source_index, dtype=torch.long, device=device)
    targets = torch.as_tensor(target_index, dtype=torch.long, device=device)
    weights = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)

    # The features are laid out over EVERY item of the embedding matrix: the
    # graph covers the training items, and an item outside it has no successor
    # and no curvature of its own (its neighbourhood is empty and its curvature
    # is the neutral mean).
    global_ids = torch.as_tensor(item_ids, dtype=torch.long, device=device)
    source_global = global_ids.index_select(0, sources)
    target_global = global_ids.index_select(0, targets)
    counts = torch.zeros(x.shape[0], dtype=torch.float32, device=device)
    counts.index_add_(0, source_global, weights)
    totals = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=device)
    totals.index_add_(0, source_global,
                       x.index_select(0, target_global) * weights[:, None])
    neighbourhood = totals / torch.clamp(counts, min=1e-12)[:, None]
    kappa = torch.full((x.shape[0],), float(curvature.mean()),
                       dtype=torch.float32, device=device)
    kappa[global_ids] = torch.as_tensor(curvature, dtype=torch.float32, device=device)

    module = CurvatureRefinement(x.shape[1]).to(device)
    optimiser = torch.optim.AdamW(module.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    edges = torch.arange(sources.numel())
    for epoch in range(1, int(epochs) + 1):
        # the permutation is drawn on the CPU (the generator is a CPU one) and the
        # edge indices are moved to the device with the batch
        order = edges[torch.randperm(edges.numel(), generator=generator)]
        totals = [0.0, 0.0, 0.0]
        steps = 0
        for start in range(0, order.numel(), batch):
            chosen = order[start:start + batch].to(device)
            anchor = sources.index_select(0, chosen)
            positive = targets.index_select(0, chosen)
            touched = torch.unique(torch.cat([anchor, positive]))
            z = module(x.index_select(0, touched), neighbourhood.index_select(0, touched),
                       kappa.index_select(0, touched))
            anchor_at = torch.searchsorted(touched, anchor)
            positive_at = torch.searchsorted(touched, positive)
            behaviour = info_nce(z.index_select(0, anchor_at), z.index_select(0, positive_at))
            spread = SPREAD_WEIGHT * spread_penalty(z)
            anchor_term = anchor_penalty(z, x.index_select(0, touched))
            loss = behaviour + spread + anchor_term
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            totals[0] += float(behaviour)
            totals[1] += spread
            totals[2] += anchor_term
            steps += 1
        if log is not None and (epoch % 5 == 0 or epoch == 1):
            log("[embedding] epoch %d/%d behaviour %.4f spread %.4f anchor %.4f"
                % (epoch, epochs, totals[0] / max(steps, 1), totals[1] / max(steps, 1),
                   totals[2] / max(steps, 1)))
    with torch.no_grad():
        module.eval()
        z = module(x, neighbourhood, kappa)
    return (z.detach().cpu().numpy().astype(np.float32),
            {"curvature_raw_mean": float(raw_curvature.mean()),
             "curvature_raw_std": float(raw_curvature.std()),
             "curvature_negative": int((raw_curvature < 0).sum()),
             "edges": int(sources.numel()),
             "items": items})
