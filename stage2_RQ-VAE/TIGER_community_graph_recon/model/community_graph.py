"""The community-guided graph reconstruction (CGQ).

One loss term is added to the frozen baseline's objective: the quantised
representation of an item must predict the distribution of its behaviour
neighbours, where the neighbours are weighted by the community model's own
geometry. The quantiser, the three levels, the codebook sizes and the balanced
assignment are untouched; the mechanism enters through the loss.

The pieces:

``behaviour_edges``
    The training split's undirected co-occurrence graph. Two items are
    adjacent when they follow each other in some user's training sequence.
    Nothing else is read.

``affinity_weights``
    The community's weight on each observed edge: ``w = A * (1 + s)``, with
    ``s`` the exponential of the negative mixed-space squared distance. A real
    edge therefore keeps a weight in ``[1, 2]``, a non-edge keeps zero, and
    no edge is ever invented or deleted.

``neighbour_target``
    The behaviour neighbourhood, renormalised inside the candidate set.

``quantised_views``
    The quantised representation of an item in two forms with the same value
    and different gradient routes: the straight-through sum reaches the
    encoder, the un-detached sum of the codeword vectors reaches the
    codebook.

``community_graph_loss``
    The cross-entropy of the model's prediction over the candidate set
    against that target, averaged over the two views.

``GradientBudget``
    The cap on the term's share of the gradient, checked separately for the
    encoder and the codebook, with the stricter of the two scalings applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .poincare_contract import distance as contract_distance

#: The number of a candidate set's columns: real neighbours plus contrast
#: candidates. Fixed, so the batch's shape does not vary between steps.
MAX_CANDIDATES = 48

#: How many real neighbours an anchor keeps, and how many contrast candidates
#: of each kind. The real neighbours are the target; the contrast candidates
#: give the softmax a denominator and carry no target mass.
MAX_NEIGHBOURS = 16
MAX_SAME_COMMUNITY = 8
MAX_OTHER_COMMUNITY = 8

#: The temperature of the cosine softmax, fixed by the specification.
TEMPERATURE = 0.2


# --------------------------------------------------------------------------
# the behaviour graph
# --------------------------------------------------------------------------

def behaviour_edges(sequences: Iterable[Sequence[int]], n_items: int) -> torch.Tensor:
    """The training split's undirected co-occurrence graph.

    Two items are adjacent when they follow each other in some user's
    training sequence. Every edge appears once, with the smaller item first,
    and no self-edge is kept.

    The only source of edges is the sequences handed in, which the trainer
    builds from the train split alone; no validation or test interaction can
    reach this function.
    """
    pairs: set[tuple[int, int]] = set()
    for sequence in sequences:
        values = [int(item) for item in sequence]
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            if not (0 <= first < n_items and 0 <= second < n_items):
                continue
            pairs.add((first, second) if first < second else (second, first))
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    ordered = sorted(pairs)
    rows = torch.tensor([a for a, _ in ordered], dtype=torch.long)
    cols = torch.tensor([b for _, b in ordered], dtype=torch.long)
    return torch.stack([rows, cols])


def edge_weight_free(edges: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """The squared Euclidean distance of every edge's endpoints in the free branch."""
    left = free[edges[0]]
    right = free[edges[1]]
    return (left - right).pow(2).sum(dim=-1)


def edge_weight_ball(edges: torch.Tensor, ball: torch.Tensor, curvature: float) -> torch.Tensor:
    """The squared contract distance of every edge's endpoints in the ball branch."""
    left = ball[edges[0]].double()
    right = ball[edges[1]].double()
    distance = contract_distance(left, right, float(curvature)).float()
    return distance.pow(2)


def affinity_weights(free: torch.Tensor, ball: torch.Tensor, edges: torch.Tensor,
                     curvature: float = 1.0) -> torch.Tensor:
    """The community's weight on each edge: ``1 + s``, with ``s`` in ``(0, 1]``.

    The two branches are divided by the median squared distance over the
    training edges of that branch, so they enter on a comparable scale, and the
    affinity is the exponential of the negative half of their sum. A weight is
    therefore in ``[1, 2]`` on an observed edge.
    """
    if edges.shape[1] == 0:
        return torch.zeros((0,), dtype=torch.float32)
    # an edge whose endpoint is not an item carries no weight; the graph
    # builder never produces one, and a caller that does must not crash
    n_items = int(free.shape[0])
    # the edges may arrive on the CPU while the representation is on the GPU
    edges = edges.to(free.device)
    valid = ((edges[0] >= 0) & (edges[0] < n_items)
             & (edges[1] >= 0) & (edges[1] < n_items))
    if not bool(valid.all()):
        edges = edges.clamp(min=0, max=max(n_items - 1, 0))
    free_squared = edge_weight_free(edges, free).double()
    ball_squared = edge_weight_ball(edges, ball, curvature).double()
    free_scale = free_squared.median().clamp_min(1e-12)
    ball_scale = ball_squared.median().clamp_min(1e-12)
    exponent = -0.5 * (free_squared / free_scale + ball_squared / ball_scale)
    affinity = torch.exp(exponent).clamp(max=1.0)
    weights = (1.0 + affinity).float()
    return weights * valid.to(weights.dtype)


# --------------------------------------------------------------------------
# the candidate set
# --------------------------------------------------------------------------

@dataclass
class CandidateSet:
    """The candidate set of every anchor in a batch.

    ``candidate_rows`` holds, for each anchor, the indices of its candidates in
    the ENCODED candidate tensor (the candidates are not in the batch, so they
    are encoded in their own forward). ``candidate_weights`` is the behaviour
    weight of each position, zero for a contrast candidate, and
    ``candidate_mask`` says which positions are real at all.
    """

    anchors: torch.Tensor
    candidate_rows: torch.Tensor
    candidate_weights: torch.Tensor
    candidate_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def width(self) -> int:
        return int(self.candidate_rows.shape[1])

    def has_neighbour(self) -> torch.Tensor:
        """Which anchors have at least one real neighbour."""
        return (self.candidate_weights > 0).any(dim=1)

    def to(self, device: torch.device) -> "CandidateSet":
        return CandidateSet(
            anchors=self.anchors.to(device),
            candidate_rows=self.candidate_rows.to(device),
            candidate_weights=self.candidate_weights.to(device),
            candidate_mask=self.candidate_mask.to(device),
        )


def build_candidate_set(anchor_ids: torch.Tensor,
                        neighbour_ids: torch.Tensor,
                        neighbour_weights: torch.Tensor,
                        same_community_ids: torch.Tensor,
                        other_community_ids: torch.Tensor) -> CandidateSet:
    """Assemble the candidate set of a batch of anchors.

    The real neighbours come first: they carry the target, and every one of
    them takes part in it. Then the same-community and the other-community
    contrast candidates, which carry no target mass. Every anchor's set is
    padded to the same width, and ``candidate_mask`` says which positions hold
    a real candidate.

    ``candidate_rows`` are the positions in the encoded candidate tensor: the
    candidates are encoded once per step, in their own forward, and the rows
    point into that tensor.
    """
    batch = int(anchor_ids.shape[0])
    width = MAX_NEIGHBOURS + MAX_SAME_COMMUNITY + MAX_OTHER_COMMUNITY
    rows = torch.zeros((batch, width), dtype=torch.long)
    weights = torch.zeros((batch, width), dtype=torch.float32)
    mask = torch.zeros((batch, width), dtype=torch.bool)

    offset = 0
    for row in range(batch):
        position = 0
        for values, values_weights, limit in (
            (neighbour_ids[row], neighbour_weights[row], MAX_NEIGHBOURS),
            (same_community_ids[row], None, MAX_SAME_COMMUNITY),
            (other_community_ids[row], None, MAX_OTHER_COMMUNITY),
        ):
            taken = 0
            for index, _ in enumerate(values.tolist()):
                if taken >= limit:
                    break
                rows[row, position] = offset + position
                if values_weights is not None:
                    weights[row, position] = float(values_weights[index])
                mask[row, position] = True
                position += 1
                taken += 1
        offset += position
    return CandidateSet(anchors=anchor_ids, candidate_rows=rows,
                        candidate_weights=weights, candidate_mask=mask)


# --------------------------------------------------------------------------
# the target
# --------------------------------------------------------------------------

def neighbour_target(weights: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
    """The behaviour neighbourhood, renormalised inside the candidate set.

    A candidate with no observed behaviour relation carries no target mass:
    its weight is zero and the target is the normalised weight over the real
    neighbours only. ``mask`` is optional and only selects which positions are
    real; the weights already encode the relation.
    """
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    total = weights.sum(dim=-1, keepdim=True)
    return weights / total.clamp_min(1e-12)


# --------------------------------------------------------------------------
# the two views of the quantised representation
# --------------------------------------------------------------------------

def quantised_views(model, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The two views of the quantised representation, with the same value.

    The model owns the computation (``RQVAE.quantised_views``); this wrapper
    exists so the loss and the tests name it once. The first view's gradient
    reaches the encoder through the straight-through estimator, the second
    view's reaches the codebook through the undetached codeword vectors.
    """
    straight_through, codewords, _ = model.quantised_views(embeddings)
    return straight_through, codewords


# --------------------------------------------------------------------------
# the loss
# --------------------------------------------------------------------------

def community_graph_loss(anchor_ste: torch.Tensor, anchor_cb: torch.Tensor,
                         candidate_ste: torch.Tensor, candidate_cb: torch.Tensor,
                         candidates: CandidateSet,
                         temperature: float = TEMPERATURE) -> torch.Tensor:
    """The cross-entropy of the neighbour prediction, averaged over the two views.

    The model predicts, for every anchor, a distribution over its candidate set
    from the cosine similarity of the quantised representations; the target is
    the behaviour neighbourhood. The anchors and the candidates are encoded in
    separate forwards, so each arrives as its own pair of views.

    The two views are averaged, so the term reaches the encoder through the
    straight-through view and the codebook through the undetached codeword
    view. Both the anchors' and the candidates' representations carry gradient:
    the codes of a neighbour move toward the anchor that names it, which is the
    graph reconstruction itself.

    The tensors are positional arguments and not a bag of views held by the
    caller, so that a gradient probe can ask each of them whether it is wired
    into the returned scalar.
    """
    if candidates.batch_size == 0:
        return torch.zeros((), dtype=torch.float32, device=anchor_ste.device)

    anchor_index = torch.arange(candidates.batch_size, device=anchor_ste.device)
    target = neighbour_target(candidates.candidate_weights, candidates.candidate_mask)
    mask = candidates.candidate_mask.to(anchor_ste.dtype)
    contributing = candidates.has_neighbour().to(anchor_ste.dtype)

    def one_view(anchor_view: torch.Tensor, candidate_view: torch.Tensor) -> torch.Tensor:
        anchor_rows = anchor_view[anchor_index]
        candidate_rows = candidate_view[candidates.candidate_rows]
        similarity = F.cosine_similarity(anchor_rows.unsqueeze(1), candidate_rows, dim=-1)
        logits = similarity / max(temperature, 1e-6)
        logits = logits.masked_fill(mask <= 0, float("-inf"))
        log_probability = torch.log_softmax(logits, dim=-1)
        # A padded position has no target mass AND a log-probability of -inf,
        # and 0 * -inf is NaN. Zeroing the masked positions keeps the product
        # finite without changing it: the target is already zero there.
        log_probability = log_probability.masked_fill(mask <= 0, 0.0)
        per_anchor = -(target * log_probability).sum(dim=-1)
        if float(contributing.sum()) == 0.0:
            return torch.zeros((), dtype=per_anchor.dtype, device=per_anchor.device)
        return (per_anchor * contributing).sum() / contributing.sum()

    return 0.5 * one_view(anchor_ste, candidate_ste) + 0.5 * one_view(anchor_cb, candidate_cb)


# --------------------------------------------------------------------------
# the gradient budget
# --------------------------------------------------------------------------

@dataclass
class GradientBudget:
    """The cap on the new term's share of the gradient.

    The term's gradient norm is measured against the original objective's, for
    the encoder and the codebook separately, and the stricter of the two
    scalings is applied. The measurement is a read-only statistic: it never
    enters the backward pass.
    """

    share: float = 0.10
    scale: float = 1.0
    encoder_scale: float = 1.0
    codebook_scale: float = 1.0

    def observe(self, base_encoder: float, new_encoder: float,
                base_codebook: float, new_codebook: float) -> float:
        """Update the scale from one step's four gradient norms."""
        self.encoder_scale = self._ratio(base_encoder, new_encoder)
        self.codebook_scale = self._ratio(base_codebook, new_codebook)
        self.scale = min(self.encoder_scale, self.codebook_scale)
        return self.scale

    def _ratio(self, base: float, new: float) -> float:
        if new <= 0.0:
            return 1.0
        if base <= 0.0:
            return 1.0
        return min(1.0, self.share * base / new)


def gradient_norm(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter],
                  retain_graph: bool = True) -> float:
    """The L2 norm of the gradient of ``loss`` over ``parameters``."""
    selected = [p for p in parameters if p.requires_grad]
    if not selected:
        return 0.0
    grads = torch.autograd.grad(loss, selected, retain_graph=retain_graph,
                                allow_unused=True)
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().double().pow(2).sum()
    return float(total.sqrt())


# --------------------------------------------------------------------------
# the surfaces the reference contract declares
# --------------------------------------------------------------------------

def affinity_of_edge(free_left: torch.Tensor, free_right: torch.Tensor,
                     ball_left: torch.Tensor, ball_right: torch.Tensor,
                     curvature, free_scale, ball_scale) -> torch.Tensor:
    """The community affinity of one edge, row by row.

    ``exp(-1/2 (d_E^2 / m_E + d_H^2 / m_H))`` with ``d_E`` the Euclidean
    distance in the free branch and ``d_H`` the contract's Poincare distance in
    the ball branch at the teacher's curvature. The two branches are divided by
    the median squared distance of their own edges, which is what makes them
    comparable.
    """
    free_squared = (free_left - free_right).pow(2).sum(dim=-1)
    ball_distance = contract_distance(ball_left, ball_right, curvature)
    ball_squared = ball_distance.pow(2)
    exponent = -0.5 * (free_squared / free_scale + ball_squared / ball_scale)
    return torch.exp(exponent)


def cgq_pair_loss(anchor: torch.Tensor, candidate_a: torch.Tensor,
                  candidate_b: torch.Tensor, offset_a: torch.Tensor,
                  offset_b: torch.Tensor, temperature) -> torch.Tensor:
    """One anchor's contribution, row by row, against two candidates.

    The target is the behaviour weight renormalised across the candidate set,
    the prediction is the cosine softmax over the same set, and the term is the
    cross-entropy between them. Two candidates is the smallest set that still
    contains the softmax and the renormalisation, so all of the continuous
    surface is present.
    """
    logit_a = F.cosine_similarity(anchor, candidate_a, dim=-1) / temperature
    logit_b = F.cosine_similarity(anchor, candidate_b, dim=-1) / temperature
    # the declared edge weight is 1 + s with s in (0, 1]; the argument is the
    # affinity s itself, so the weight is strictly positive by construction
    weight_a = 1.0 + offset_a
    weight_b = 1.0 + offset_b
    total = weight_a + weight_b
    target_a = weight_a.squeeze(-1) / total.squeeze(-1)
    target_b = weight_b.squeeze(-1) / total.squeeze(-1)
    log_probability = torch.log_softmax(torch.stack([logit_a, logit_b], dim=-1), dim=-1)
    return -(target_a * log_probability[..., 0] + target_b * log_probability[..., 1])
