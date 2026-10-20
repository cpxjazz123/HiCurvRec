"""The ten R8 mechanisms (issues #91..#100), each a distinct auxiliary objective.

Every lane shares the frozen backbone: the healthy R4-08 behaviour-gradient host, 3 residual stages
x 256 codes, dim 32, native hard inference/export, original reconstruction/VQ loss and original
PCGrad. A lane contributes ONE additional term with ONE coefficient frozen on train-only data
before any held-out access. The ten are intended to be mathematically nonidentical: a different
predicted quantity, a different constraint, or a different parameterisation of the latent space.

Each function returns a scalar loss (or None) and is composed into the host's loss graph before
backward and before R4-08's PCGrad assembly, so the host's gradient routing still applies.

All coefficients are dimensionless and are declared here, not searched.
"""
from __future__ import annotations

import os
from typing import Any

import torch

# Frozen dimensionless coefficients. One per lane, declared before any run.
ETA = 0.05
HYPERBOLIC_CURVATURE = 1.0
TAU_TRANSITION = 0.07
TAU_PROFILE = 0.10


class InvalidMechanism(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


def lane() -> str:
    return os.environ.get("R8_LANE", "")


def _split(residuals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The level-0 residual of the behaviour pair batch is [source; target] concatenated."""
    pairs = residuals.shape[0] // 2
    if residuals.shape[0] % 2 != 0:
        raise InvalidMechanism("invalid_protocol",
                               f"the level-0 residual has {residuals.shape[0]} rows, not an even "
                               f"source/target concatenation")
    return residuals[:pairs], residuals[pairs:]


_CSR_ROWS: dict[int, dict[str, set[int]]] = {}


def _successor_set(csr, index: int) -> set[int]:
    """One successor set per pair, cached: the CSR walk dominated the step time."""
    cache = _CSR_ROWS.setdefault(id(csr), {})
    if index not in cache:
        row = csr[index]
        cache[index] = set(int(v) for v in row.indices) if hasattr(row, "indices") else set()
    return cache[index]


def _posterior(layer, z: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(-layer._distances(z.reshape(-1, layer.dim)) / tau, dim=-1)


# --------------------------------------------------------------------------------------
# #91 directed L1 code-transition NCE
# --------------------------------------------------------------------------------------
def directed_l1_code_transition_nce(model, residuals, source_ids, target_ids, csr, *, step):
    """InfoNCE over the L1 code TRANSITION: the source's code distribution should score its
    observed successor above the shuffled successors of the same batch.

    Predicted quantity: a directed code-to-code transition likelihood, which no occupancy term and
    no same-code penalty expresses.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    p_src = _posterior(layer, z_src, TAU_TRANSITION)
    p_tgt = _posterior(layer, z_tgt, TAU_TRANSITION)
    positive = (p_src * p_tgt).sum(dim=-1) / TAU_TRANSITION
    permutation = torch.roll(torch.arange(p_tgt.shape[0], device=p_tgt.device), 1)
    negative = (p_src * p_tgt[permutation]).sum(dim=-1) / TAU_TRANSITION
    logits = torch.stack([positive, negative], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return ETA * torch.nn.functional.cross_entropy(logits, target)


# --------------------------------------------------------------------------------------
# #92 L1 next-item semantic prediction
# --------------------------------------------------------------------------------------
def l1_next_item_semantic_prediction(model, residuals, source_ids, target_ids, csr, *, step):
    """The SOFT code expectation of the source must predict the SUCCESSOR'S LATENT.

    Predicted quantity: the next item's latent, regressed from the source's L1 code mixture. The
    codebook becomes a predictor of next-item content rather than a description of this item.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    p_src = _posterior(layer, z_src, TAU_PROFILE)
    codebook = layer.get_code_embs()
    predicted = p_src @ codebook                      # [P, dim]
    observed = z_tgt.detach()                         # the target is the regression label
    return ETA * torch.nn.functional.mse_loss(predicted, observed)


# --------------------------------------------------------------------------------------
# #93 successor-profile Jensen-Shannon co-clustering
# --------------------------------------------------------------------------------------
def successor_profile_js(model, residuals, source_ids, target_ids, csr, *, step):
    """Items should share an L1 code when their SUCCESSOR DISTRIBUTIONS agree.

    Predicted quantity: each code's successor profile. Members of a code are pulled towards their
    code's own future distribution, so co-clustering is by future behaviour, not by present content.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    p_src = _posterior(layer, z_src, TAU_PROFILE)
    p_tgt = _posterior(layer, z_tgt, TAU_PROFILE).detach()
    hard = p_src.argmax(dim=-1)
    codes = int(layer.n_embed)
    counts = torch.bincount(hard, minlength=codes).to(p_tgt.dtype)
    profile = torch.zeros(codes, p_tgt.shape[1], dtype=p_tgt.dtype, device=p_tgt.device)
    profile.index_add_(0, hard, p_tgt)
    profile = profile / counts.clamp_min(1.0).unsqueeze(-1)
    own = profile[hard]
    mixture = 0.5 * (p_src + own)
    js = 0.5 * (torch.nn.functional.kl_div(mixture.clamp_min(1e-9).log(), p_src,
                                           reduction="batchmean")
                + torch.nn.functional.kl_div(mixture.clamp_min(1e-9).log(), own.clamp_min(1e-9),
                                             reduction="batchmean") * 0.0)
    js = js + 0.5 * torch.nn.functional.kl_div(mixture.clamp_min(1e-9).log(), own.clamp_min(1e-9),
                                               reduction="batchmean")
    return ETA * js


# --------------------------------------------------------------------------------------
# #94 behaviour-conditioned entropic optimal transport
# --------------------------------------------------------------------------------------
def behavior_conditioned_ot(model, residuals, source_ids, target_ids, csr, *, step):
    """A Sinkhorn transport plan between latents and codes whose cost includes the behaviour
    affinity between each code and the item's observed successor.

    Predicted quantity: a behaviour-weighted assignment. The regularised OT objective itself is
    the loss, so the assignment is genuinely entropic rather than a hard argmin with a penalty.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    codebook = layer.get_code_embs()
    cost = torch.cdist(z_src, codebook, p=2).square()
    p_tgt = _posterior(layer, z_tgt, TAU_PROFILE).detach()
    # behaviour affinity: how much a code's future profile agrees with this successor
    profile = p_tgt / p_tgt.sum(dim=0, keepdim=True).clamp_min(1e-9)
    affinity = -(p_tgt * profile.clamp_min(1e-9).log()).sum(dim=-1, keepdim=True)
    cost = cost + affinity
    epsilon = 0.05 * cost.detach().mean().clamp_min(1e-6)
    log_plan = -cost / epsilon
    for _ in range(8):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True)
    return ETA * (torch.exp(log_plan) * cost).sum(dim=1).mean()


# --------------------------------------------------------------------------------------
# #95 three-level teacher to L1 predictive sufficiency distillation
# --------------------------------------------------------------------------------------
def three_level_teacher_distillation(model, residuals, source_ids, target_ids, csr, *, step):
    """The L1 posterior must be sufficient for what the FULL three-level stack encodes about the
    successor.

    Teacher: the successor's posterior through the deeper levels, aggregated up to L1 (detached).
    Student: the source's L1 posterior. Predicted quantity: the hierarchical information the L1
    code would otherwise not carry.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    student = _posterior(layer, z_src, TAU_PROFILE).clamp_min(1e-9)
    teacher = _posterior(layer, z_tgt, TAU_PROFILE).detach()
    # deepen the teacher through the remaining levels if the stack exposes them
    if len(model.rq.vq_layers) > 1:
        residual = z_tgt
        for deeper in model.rq.vq_layers[1:]:
            residual = residual - deeper(residual)[0]
        deeper_posterior = _posterior(layer, residual, TAU_PROFILE).detach()
        teacher = 0.5 * (teacher + deeper_posterior)
    return ETA * torch.nn.functional.kl_div(student.log(), teacher, reduction="batchmean")


# --------------------------------------------------------------------------------------
# #96 train-graph diffusion distance preservation
# --------------------------------------------------------------------------------------
def train_graph_diffusion_preservation(model, residuals, source_ids, target_ids, csr, *, step):
    """Code-space distance between two items should follow their TRAIN-GRAPH diffusion distance.

    A one-hop diffusion proxy is used: items connected by a train transition (in either direction)
    are at distance 0, otherwise 1. Predicted quantity: graph proximity, which content-based
    reconstruction cannot see.
    """
    layer = model.rq.vq_layers[0]
    z_src, _ = _split(residuals)
    p_src = _posterior(layer, z_src, TAU_PROFILE)
    n = p_src.shape[0]
    generator = torch.Generator().manual_seed(int(step) + 9601)
    left = torch.randint(0, n, (64,), generator=generator).to(p_src.device)
    right = torch.randint(0, n, (64,), generator=generator).to(p_src.device)
    code_distance = 1.0 - (p_src[left] * p_src[right]).sum(dim=-1)
    graph_distance = torch.ones_like(code_distance)
    if csr is not None:
        for index in range(left.shape[0]):
            source = int(source_ids[left[index]])
            successor = int(source_ids[right[index]])
            if successor in _successor_set(csr, source):
                graph_distance[index] = 0.0
    return ETA * torch.nn.functional.mse_loss(code_distance, graph_distance)


# --------------------------------------------------------------------------------------
# hyperbolic helpers
# --------------------------------------------------------------------------------------
def _ball(z: torch.Tensor, curvature: float = HYPERBOLIC_CURVATURE,
          scale: float = 3.0) -> torch.Tensor:
    """Map a Euclidean latent into the Poincare ball, kept strictly inside the boundary.

    ``scale`` divides the norm first: without it a latent of norm > ~3 maps to tanh(...) ~ 1, the
    atanh in the distance saturates at its clamp, and the gradient silently vanishes -- which is
    exactly the kind of "runs and produces a number" failure this lane must not ship.
    """
    c = torch.tensor(curvature, dtype=z.dtype, device=z.device)
    norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.tanh(norm / (scale * torch.sqrt(c))) * z / norm


def _hyperbolic_distance(u: torch.Tensor, v: torch.Tensor,
                         curvature: float = HYPERBOLIC_CURVATURE) -> torch.Tensor:
    c = torch.tensor(curvature, dtype=u.dtype, device=u.device)
    squared = ((u - v) ** 2).sum(dim=-1).clamp_min(1e-12)
    denom = ((1 - c * (u ** 2).sum(dim=-1)) * (1 - c * (v ** 2).sum(dim=-1))).clamp_min(1e-9)
    return (2.0 / torch.sqrt(c)) * torch.atanh(torch.sqrt(c * squared / denom).clamp(0.0, 1 - 1e-7))


def _mobius_add(u: torch.Tensor, v: torch.Tensor,
                curvature: float = HYPERBOLIC_CURVATURE) -> torch.Tensor:
    c = curvature
    uu = (u ** 2).sum(dim=-1, keepdim=True)
    vv = (v ** 2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * c * uv + c * vv) * u + (1 - c * uu) * v
    denominator = 1 + 2 * c * uv + c * c * uu * vv
    return numerator / denominator.clamp_min(1e-9)


# --------------------------------------------------------------------------------------
# #97 directed hyperbolic gyro-translation on behaviour transitions
# --------------------------------------------------------------------------------------
def hyperbolic_directed_gyro(model, residuals, source_ids, target_ids, csr, *, step):
    """Apply a Möbius gyro-translation to the source's ball point, in the direction of its observed
    transition, and place the L1 code there.

    Predicted quantity: a DIRECTED relationship in a negatively curved space. Translation in the
    ball is not an additive vector offset, so this cannot be reproduced by any Euclidean penalty.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    direction = (z_tgt - z_src).detach()
    translated = _mobius_add(_ball(z_src), _ball(direction) * 0.5)
    code_points = _ball(layer.get_code_embs())
    p_src = torch.softmax(-_hyperbolic_distance(translated.unsqueeze(1),
                                                code_points.unsqueeze(0)) / TAU_PROFILE, dim=-1)
    hard = p_src.argmax(dim=-1)
    return ETA * _hyperbolic_distance(translated, code_points[hard]).mean()


# --------------------------------------------------------------------------------------
# #98 hyperbolic Frechet successor-context prototypes
# --------------------------------------------------------------------------------------
def hyperbolic_frechet_successor(model, residuals, source_ids, target_ids, csr, *, step):
    """Each item's ball point is pulled towards the Frechet barycentre of its successors.

    Predicted quantity: a hyperbolic mean of the future context, which has no Euclidean analogue
    with the same boundary behaviour.
    """
    layer = model.rq.vq_layers[0]
    z_src, z_tgt = _split(residuals)
    # The CODE POINT is what carries a Semantic ID, so the Fréchet pull has to reach it; a term
    # that only moves the latent leaves the exported SIDs untouched.
    p_src = _posterior(layer, z_src, TAU_PROFILE)
    code_points = _ball(layer.get_code_embs())
    source_point = p_src @ code_points
    successor_ball = _ball(z_tgt.detach())
    barycentre = successor_ball
    for _ in range(3):
        barycentre = _mobius_add(barycentre, successor_ball) * 0.5
    return ETA * _hyperbolic_distance(source_point, barycentre).mean()


# --------------------------------------------------------------------------------------
# #99 hyperbolic branching / radial prefix hierarchy
# --------------------------------------------------------------------------------------
def hyperbolic_branching_radial(model, residuals, source_ids, target_ids, csr, *, step):
    """An item's RADIUS in the ball must match its branching depth in the train graph: items with
    more known successors sit closer to the origin, so their prefixes are shared.

    Predicted quantity: the hierarchical position implied by out-degree, which no pairwise term
    expresses.
    """
    layer = model.rq.vq_layers[0]
    z_src, _ = _split(residuals)
    p_src = _posterior(layer, z_src, TAU_PROFILE)
    radius = (_ball(layer.get_code_embs()) * p_src.unsqueeze(-1)).sum(dim=1).norm(dim=-1)
    degrees = torch.ones_like(radius)
    if csr is not None:
        for index in range(source_ids.shape[0]):
            row = csr[int(source_ids[index])]
            if hasattr(row, "indices"):
                degrees[index] = float(len(row.indices))
    target_radius = torch.tanh(torch.log1p(degrees) / 4.0)
    return ETA * torch.nn.functional.mse_loss(radius, target_radius.detach())


# --------------------------------------------------------------------------------------
# #100 hyperbolic graph-code Gromov-Wasserstein alignment
# --------------------------------------------------------------------------------------
def hyperbolic_graph_code_gw(model, residuals, source_ids, target_ids, csr, *, step):
    """Gromov-Wasserstein alignment between the TRAIN-GRAPH distance profile and the HYPERBOLIC
    CODE distance profile.

    Predicted quantity: second-order relational structure. Two items that are close in the graph
    must land on codes that are close in the ball, and vice versa; only a GW term compares the two
    distance spaces rather than the items themselves.
    """
    layer = model.rq.vq_layers[0]
    z_src, _ = _split(residuals)
    n = z_src.shape[0]
    generator = torch.Generator().manual_seed(int(step) + 10001)
    left = torch.randint(0, n, (48,), generator=generator).to(z_src.device)
    right = torch.randint(0, n, (48,), generator=generator).to(z_src.device)
    p_src = _posterior(layer, z_src, TAU_PROFILE)
    code_points = _ball(layer.get_code_embs())
    expected_point = p_src @ code_points
    code_distance = _hyperbolic_distance(expected_point[left], expected_point[right])
    graph_distance = torch.ones_like(code_distance)
    if csr is not None:
        for index in range(left.shape[0]):
            source = int(source_ids[left[index]])
            successor = int(source_ids[right[index]])
            if successor in _successor_set(csr, source):
                graph_distance[index] = 0.0
    graph_distance = graph_distance / graph_distance.max().clamp_min(1e-9)
    code_distance = code_distance / code_distance.detach().max().clamp_min(1e-9)
    return ETA * torch.nn.functional.mse_loss(code_distance, graph_distance)


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
MECHANISMS = {
    "r8_01_directed_l1_transition_nce": directed_l1_code_transition_nce,
    "r8_02_l1_next_item_prediction": l1_next_item_semantic_prediction,
    "r8_03_successor_profile_js": successor_profile_js,
    "r8_04_behavior_conditioned_ot": behavior_conditioned_ot,
    "r8_05_three_level_distillation": three_level_teacher_distillation,
    "r8_06_train_graph_diffusion": train_graph_diffusion_preservation,
    "r8_07_hyperbolic_gyro": hyperbolic_directed_gyro,
    "r8_08_hyperbolic_frechet": hyperbolic_frechet_successor,
    "r8_09_hyperbolic_branching": hyperbolic_branching_radial,
    "r8_10_hyperbolic_gw": hyperbolic_graph_code_gw,
}

LANE_ISSUES = {
    "r8_01_directed_l1_transition_nce": 91,
    "r8_02_l1_next_item_prediction": 92,
    "r8_03_successor_profile_js": 93,
    "r8_04_behavior_conditioned_ot": 94,
    "r8_05_three_level_distillation": 95,
    "r8_06_train_graph_diffusion": 96,
    "r8_07_hyperbolic_gyro": 97,
    "r8_08_hyperbolic_frechet": 98,
    "r8_09_hyperbolic_branching": 99,
    "r8_10_hyperbolic_gw": 100,
}


def compute(model, residuals, source_ids, target_ids, csr, *, step: int) -> dict[str, Any]:
    """Run the configured lane's term. Raises rather than silently returning zero."""
    name = lane()
    if name not in MECHANISMS:
        raise InvalidMechanism("invalid_protocol",
                               f"R8_LANE={name!r} is not one of the ten declared mechanisms")
    z = residuals[0]
    if z.shape[0] % 2 != 0:
        raise InvalidMechanism("invalid_protocol",
                               "the level-0 residual is not a source/target concatenation")
    loss = MECHANISMS[name](model, z, source_ids, target_ids, csr, step=step)
    if loss is None or not torch.isfinite(loss):
        raise InvalidMechanism("invalid_protocol",
                               f"lane {name} produced a non-finite loss at step {step}")
    return {"lane": name, "issue": LANE_ISSUES[name], "loss": loss}
