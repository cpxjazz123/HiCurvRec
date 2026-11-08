"""Issue #140 (M1): the AF3 structural objective over the curved levels.

What this module optimises is not the continuous embedding: it is the graph the
*quantized* codes would induce. The geometry is frozen -- three levels of
E16 x H16 joint codewords with the native residual and the baseline's own
assignment rule -- and the only thing that changes is the objective, which now
carries a differentiable expectation of the edge AF3 target and a neighbour
retention term, both read off the quantization.

The readout
-----------
``phi(q)`` is the same readout the frozen tree exports: the Euclidean halves
summed and the tangent halves composed by Mobius addition, right-nested. Here the
codeword of each level is its soft expectation ``P_l @ C_l`` so the value is
differentiable in the encoder and in every codebook, and it coincides with the
hard readout whenever the assignment is one-hot.

The edge probability
--------------------
For a pair ``(i, j)``, ``delta_ij = d_mix^2(phi_i, phi_j)`` and

    p_ij = sigmoid(b - delta_ij / tau),

where ``(tau, b)`` is the ``af3_structure.calibrate`` pair computed once on one
TRAIN snapshot -- ``tau`` the inter-quartile range of the readout on the support,
``b`` the intercept that reproduces the TRAIN edge density exactly. Neither is
fitted during training and neither depends on the arm's own results.

The AF3 expectation
-------------------
For independent Bernoulli edges,

    F_hat_ij = p_ij * [2 - sum_{k != i,j} (p_ik + p_jk) + 3 * sum_{k != i,j} p_ik p_jk],

which is the expectation of ``4 - d_i - d_j + 3 T_ij`` under the model's own edge
distribution. The sums run over a fixed local candidate set rather than the whole
graph: the full sums are ``O(N^2)`` per anchor, and a candidate set of the
anchor's TRAIN neighbours plus a fixed sample of TRAIN non-neighbours is the
documented approximation. ``tests/test_post_sid_af3.py`` measures the bias of that
approximation against a dense oracle on a small graph.

The objective
-------------
    L = L_RQ + s * [ w_af3 * Huber(F_hat - F) + w_n * L_N + w_use * L_use ],

with ``s`` from :class:`af3_hierarchy.GradientBudget`: the new terms' combined
encoder and codebook gradients may not exceed a fixed share of the reconstruction
objective's. The weights are unit weights declared in ``math_spec.json``; the
share is what makes the scale derivable instead of tuned. ``L_N`` is a logistic
ranking of TRAIN edges above TRAIN non-edges on the fixed support, and ``L_use``
is the mean per-level deficit from a uniform code distribution, which is what
stops "every item gets the same code" from being a solution.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .af3_hierarchy import gradient_norm
from .learned_curvature import bounded_ball, level_cost
from .af3_structure import readout_delta, readout_probability
from .mixed_rq import mixed_cost, split

#: Anchors per step: the local AF3 sums are O(anchors * candidates^2).
ANCHORS_PER_STEP = 96
#: The fixed local candidate set of one anchor.
CANDIDATES = 32
#: Huber width on the standardised AF3 residual.
HUBER_DELTA = 1.0
#: Declared unit weights of the three new terms; the scale comes from the budget.
LOSS_WEIGHTS = {"af3": 1.0, "neighbour": 1.0, "usage": 1.0}
#: The frozen gradient-share bound, the same protocol as the existing auxiliary arm.
GRADIENT_SHARE = 0.10
#: Start epoch: the mechanism is the objective, so it is active from the start.
START_EPOCH = 1
#: The budget is measured once per epoch, not once per step. Four extra
#: ``autograd.grad`` calls per step would cost more than four forward passes and
#: turned a 0.5 s epoch into an eleven-second one; the gradient share is a
#: slow-varying statistic, so one measurement per epoch is what the bound needs.
BUDGET_EPOCHS = 1


def curved_assignments(latent: torch.Tensor, levels, kappas, temperature: float = 1.0):
    """The soft assignment of every item at every level, at the level's curvature.

    Level ``l`` assigns the residual left after the previous levels' expected
    tangent codewords, and its cost is the curved cost, so the assignment -- and
    with it the structural objective -- depends on ``kappa_l``.
    """
    import torch as _torch
    assignments = []
    residual = latent
    for level, kappa in zip(levels, kappas):
        codes = level.get_code_embs()
        codes_e, codes_t = split(codes)
        residual_e, residual_t = split(residual)
        alpha = _torch.full((len(residual),), 0.5, dtype=residual.dtype, device=residual.device)
        cost = level_cost(residual_e, residual_t, codes_e, codes_t, alpha, kappa)
        weights = _torch.softmax(-cost / max(temperature, 1e-6), dim=-1)
        assignments.append(weights)
        residual = residual - weights @ codes
    return tuple(assignments)


def kappa_gradient(auxiliary: torch.Tensor, theta: torch.Tensor) -> float:
    """``|d auxiliary / d theta|`` -- the mechanism's own non-zero-gradient evidence."""
    gradient = torch.autograd.grad(auxiliary, theta, retain_graph=True, allow_unused=True)[0]
    return 0.0 if gradient is None else float(gradient.abs().sum())


def soft_readout(assignments, codebooks, curvature: float = 1.0) -> torch.Tensor:
    """``phi`` for soft assignments: the hard readout's differentiable extension."""
    from .mixed_rq import EUCLIDEAN_DIM
    expected = [assignment @ codebook for assignment, codebook in zip(assignments, codebooks)]
    tangent = torch.stack([part[..., EUCLIDEAN_DIM:] for part in expected], dim=0)
    euclidean = torch.stack([part[..., :EUCLIDEAN_DIM] for part in expected], dim=0).sum(0)
    composed = tangent[-1]
    for part in reversed(tangent[:-1]):
        composed = _compose(part, composed, curvature)
    # The exported readout projects and takes log_0 ONCE, after the whole chain;
    # doing it per step would change the value.
    if float(curvature) > 0:
        from .poincare_contract import logmap0, project
        composed = logmap0(project(composed, curvature), curvature)
    return torch.cat([euclidean, composed], dim=-1)


def _compose(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    """The right-nested composition of the EXPORTED readout, verbatim.

    ``mixed_rq.quantized_readout`` composes the three tangent halves with
    ``mobius_add`` directly -- it does not map them into the ball first -- and then
    projects and takes ``log_0``. The soft readout must follow the same convention
    or a one-hot assignment would not reproduce the exported value.
    """
    from .poincare_contract import mobius_add
    if float(curvature) <= 0:
        return left + right
    return mobius_add(left, right, curvature)


def structural_delta(phi_left: torch.Tensor, phi_right: torch.Tensor) -> torch.Tensor:
    """The frozen structural metric of the readout: ``1 - cos(phi_i, phi_j)``.

    Issue #143 requires the same convention as the existing ``D_AF3`` and
    neighbour retention, and that convention is ``af3_structure.readout_delta`` --
    a cosine dissimilarity on the readout, not the quantizer's own mixed cost.
    Using anything else here would make the training objective and the reported
    metric two different quantities.
    """
    left = phi_left.to(torch.float64)
    right = phi_right.to(torch.float64)
    denominator = (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
    return 1.0 - (left * right).sum(-1) / denominator


def mixed_delta(phi_left: torch.Tensor, phi_right: torch.Tensor,
                curvature: float = 1.0) -> torch.Tensor:
    """``delta`` between two readouts: the frozen mixed distance on a pair.

    The Euclidean halves contribute their squared norm and the tangent halves are
    mapped into the ball before the canonical distance, which is what the hard
    readout's ``D_AF3`` measures.
    """
    from .poincare_contract import expmap0, squared_distance
    left_e, left_t = split(phi_left)
    right_e, right_t = split(phi_right)
    euclidean = (left_e - right_e).pow(2).sum(-1)
    if float(curvature) > 0:
        curved = squared_distance(expmap0(left_t, curvature), expmap0(right_t, curvature),
                                  curvature) / 4.0
    else:
        curved = left_t.sub(right_t).pow(2).sum(-1) / 4.0
    return euclidean + curved


def post_af3_probability(delta: torch.Tensor, tau: float, intercept: float) -> torch.Tensor:
    return readout_probability(delta, float(tau), float(intercept))


def point_probability(delta: torch.Tensor, intercept: float, tau: float) -> torch.Tensor:
    """``p = sigmoid(b - delta / tau)`` written as the reference contract states it.

    ``1 / (1 + exp(delta/tau - b))`` is algebraically the same function and
    numerically better behaved on the large-shift side.
    """
    return 1.0 / (1.0 + torch.exp(delta / float(tau) - float(intercept)))


def huber_penalty(residual: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber, as the quadratic minus its excised tail, so it stays differentiable.

    ``0.5 r^2 - max(0, |r| - delta)^2 / (2 delta)`` equals ``delta |r| - delta^2/2``
    for ``|r| > delta`` and ``0.5 r^2`` otherwise, and unlike a Python branch the
    form has a gradient everywhere in ``r``.
    """
    magnitude = residual.abs()
    tail = torch.clamp(magnitude - float(delta), min=0.0)
    return 0.5 * residual.pow(2) - tail.pow(2) / (2.0 * float(delta))


def post_af3_objective(delta: torch.Tensor, target: torch.Tensor, tau: float,
                       intercept: float, scale: float, huber_delta: float) -> torch.Tensor:
    """The edge-level term of the post-SID objective, on the quantized readout.

    The probability is read off ``delta``, the mixed distance between the two
    quantized readouts; the target is the frozen TRAIN AF3 of that pair. This is
    the surface whose gradient must reach the encoder and both codebooks.
    """
    probability = point_probability(delta, intercept, tau)
    value = huber_penalty((probability - target) / float(scale), huber_delta)
    # The reference implementation returns one scalar per sample row, so a trailing
    # width-one axis is dropped here as well; wider rows keep their shape.
    return value.squeeze(-1) if value.dim() > 1 and value.shape[-1] == 1 else value


def expected_af3(anchor_row: torch.Tensor, partner_row: torch.Tensor,
                 pair_probability: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """``F_hat`` for every anchor/partner pair, from the local sums over ``k``.

    ``anchor_row[a, c] = p_{i_a, k_c}`` and ``partner_row[a, c] = p_{j_a, k_c}``:
    the two probability rows the expression needs, one per candidate. With them

        F_hat = p_ij * [2 - sum_k (p_ik + p_jk) + 3 sum_k p_ik p_jk],

    and ``candidate_mask`` marks which candidates are legal ``k``. The formula
    excludes ``k = i`` and ``k = j``; both are excluded here by mask, because the
    local candidate set contains both endpoints of the pair.
    """
    legal = candidate_mask.to(anchor_row.dtype)
    degree_sum = ((anchor_row + partner_row) * legal).sum(-1)
    joint = (anchor_row * partner_row * legal).sum(-1)
    return pair_probability * (2.0 - degree_sum + 3.0 * joint)


def prepare_state(graph: dict, support: dict, targets: np.ndarray,
                  all_embeddings: torch.Tensor, device: torch.device) -> dict:
    """Freeze the TRAIN-only support, the candidate sets and the AF3 scale."""
    item_ids = np.asarray(graph["item_ids"], dtype=np.int64)
    pairs = np.asarray(support["pairs"], dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("the support must be a pair array")
    if not np.all(np.isfinite(targets)):
        raise ValueError("the AF3 targets must be finite")
    af3 = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
    pairs_t = torch.as_tensor(pairs, dtype=torch.long, device=device)
    # The support declares which pairs are edges (``is_edge``); that array is the
    # authority, because a real edge can carry an AF3 of exactly zero while a
    # non-edge carries the zero that the target lookup did not find. The edge count
    # is checked against it rather than against the target's non-zero pattern.
    declaration = support.get("is_edge")
    if declaration is not None:
        is_edge = torch.as_tensor(np.asarray(declaration, dtype=bool), device=device)
        if len(is_edge) != len(af3):
            raise ValueError("the support's adjacency mask does not match its pairs")
    else:
        is_edge = af3 > 0
    declared = int(support.get("edges", -1))
    if declared >= 0 and int(is_edge.sum()) != declared:
        raise ValueError("the support's edge count does not match its adjacency mask")
    scale = float(torch.quantile(af3[is_edge], 0.9) - torch.quantile(af3[is_edge], 0.5)) \
        if int(is_edge.sum()) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("the AF3 target has no usable scale")
    state = {"pairs": pairs_t, "af3": af3, "is_edge": is_edge, "scale": scale,
             "item_ids": item_ids, "all_embeddings": all_embeddings,
             "device": device, "n_items": len(item_ids)}
    state["pool"] = build_candidates(state)
    return state


def build_candidates(state: dict, *, pool: int = 4096, seed: int = 42) -> dict:
    """The fixed candidate pool: anchors from the support, each with its candidates.

    Issue #143 asks for a FIXED support, and rebuilding neighbourhoods from 315k
    edges on every step was also the reason an epoch took eleven seconds. The pool
    is built once, deterministically, and a step samples rows from it.
    """
    pairs = state["pairs"].cpu().numpy()
    item_ids = state["item_ids"]
    neighbours: list[set] = [set() for _ in range(len(item_ids))]
    for left, right in pairs:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(pairs), size=min(pool, len(pairs)), replace=False)
    rows = np.zeros((len(anchors), CANDIDATES), dtype=np.int64)
    for row, index in enumerate(anchors):
        partner = int(pairs[index][1])
        pool_nodes = list(neighbours[partner])
        picks = [partner]
        if len(pool_nodes) >= CANDIDATES - 1:
            picks.extend(int(value) for value in rng.choice(pool_nodes, CANDIDATES - 1,
                                                            replace=False))
        else:
            picks.extend(pool_nodes)
            fillers = [int(value) for value in rng.permutation(len(item_ids))]
            for candidate in fillers:
                if len(picks) >= CANDIDATES:
                    break
                if candidate not in picks:
                    picks.append(candidate)
            while len(picks) < CANDIDATES:
                picks.append(int(rng.integers(len(item_ids))))
        rows[row] = item_ids[np.asarray(picks, dtype=np.int64)]
    device = state["device"]
    return {"anchor_index": torch.as_tensor(anchors, dtype=torch.long, device=device),
            "candidates": torch.as_tensor(rows, dtype=torch.long, device=device)}


def sample_candidates(state: dict, rng) -> dict:
    """One step: a random subset of the fixed pool, with its pair and target."""
    pool = state["pool"]
    count = min(ANCHORS_PER_STEP, len(pool["anchor_index"]))
    choice = torch.as_tensor(rng.choice(len(pool["anchor_index"]), size=count, replace=False),
                             dtype=torch.long, device=state["device"])
    index = pool["anchor_index"][choice]
    pairs = state["pairs"][index]
    return {"left": pairs[:, 0], "right": pairs[:, 1],
            "candidates": pool["candidates"][choice],
            "target": state["af3"][index], "edge": state["is_edge"][index]}


def candidate_sets(state: dict, rng) -> dict:
    """Deterministic candidate sets: the partner, its neighbours, and fixed non-edges."""
    pairs = state["pairs"].cpu().numpy()
    item_ids = state["item_ids"]
    position = {int(item): index for index, item in enumerate(item_ids)}
    neighbours: list[set] = [set() for _ in range(len(item_ids))]
    for left, right in pairs:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))
    anchors = rng.integers(len(pairs), size=ANCHORS_PER_STEP)
    chosen = pairs[anchors]
    left = torch.as_tensor(item_ids[chosen[:, 0]], dtype=torch.long, device=state["device"])
    right = torch.as_tensor(item_ids[chosen[:, 1]], dtype=torch.long, device=state["device"])
    candidates = torch.zeros((ANCHORS_PER_STEP, CANDIDATES), dtype=torch.long,
                             device=state["device"])
    for row, (_, anchor_right) in enumerate(chosen):
        pool = list(neighbours[int(anchor_right)])
        picks = [int(anchor_right)]
        if len(pool) >= CANDIDATES - 1:
            picks.extend(int(value) for value in rng.choice(pool, CANDIDATES - 1, replace=False))
        else:
            picks.extend(pool)
            # The pool may be smaller than the candidate width (on a tiny graph,
            # always). Fill from a shuffled sweep of all positions rather than an
            # unbounded draw, so the loop terminates by construction.
            fillers = [int(value) for value in rng.permutation(len(item_ids))]
            for candidate in fillers:
                if len(picks) >= CANDIDATES:
                    break
                if candidate not in picks:
                    picks.append(candidate)
            while len(picks) < CANDIDATES:
                picks.append(int(rng.integers(len(item_ids))))
        candidates[row] = torch.as_tensor(item_ids[np.asarray(picks, dtype=np.int64)],
                                         dtype=torch.long, device=state["device"])
    target = state["af3"][torch.as_tensor(anchors, dtype=torch.long, device=state["device"])]
    edge = state["is_edge"][torch.as_tensor(anchors, dtype=torch.long, device=state["device"])]
    return {"left": left, "right": right, "candidates": candidates,
            "target": target, "edge": edge}


def usage_deficit(assignments) -> torch.Tensor:
    """Per level: how far the batch's mean code distribution is from uniform.

    Stay in the tensor graph: this term is part of the objective, so casting it to
    a Python float would silently drop its gradient and leave a term that is
    reported but never applied.
    """
    terms = []
    for assignment in assignments:
        mean = assignment.mean(0).clamp_min(1e-9)
        entropy = -(mean * mean.log()).sum()
        terms.append(1.0 - entropy / float(np.log(len(mean))))
    return torch.stack(terms).mean()


def augment_loss(base_loss: torch.Tensor, raw_module, state: dict | None,
                 epoch: int, rng, calibration: dict | None = None,
                 curvature: float = 1.0, budget: dict | None = None
                 ) -> tuple[torch.Tensor, dict]:
    """The post-SID AF3 objective, added to the reconstruction objective.

    ``budget`` is a per-run dictionary. Its ``scale`` is measured on the first step
    of an epoch and reused for the rest of that epoch, so the four gradient
    measurements cost one step per epoch rather than one step each.
    """
    if state is None or epoch < START_EPOCH or calibration is None:
        return base_loss, {"active": False, "epoch": epoch}
    device = next(raw_module.parameters()).device
    picked = sample_candidates(state, rng)
    codebooks = [layer.get_code_embs() for layer in raw_module.rq.vq_layers]
    embeddings = state["all_embeddings"]

    def readout(item_ids: torch.Tensor) -> torch.Tensor:
        # The embedding table lives on the host; the ids arrive on the compute
        # device. Index where the table is, then move the gathered rows.
        rows = embeddings[item_ids.to(embeddings.device)].to(device)
        latent = raw_module.encoder(rows)
        kappas = raw_module.rq.kappa()
        assignments = curved_assignments(latent, raw_module.rq.levels, kappas)
        return soft_readout(assignments, [level.get_code_embs() for level in raw_module.rq.levels],
                            curvature=float(kappas[0].abs())), assignments

    phi_left, assignments = readout(picked["left"])          # i, the pair's left end
    phi_right, _ = readout(picked["right"])                   # j, the pair's right end
    count = len(picked["left"])
    candidate_phi, _ = readout(picked["candidates"].reshape(-1))
    candidate_phi = candidate_phi.view(count, CANDIDATES, -1)

    # p_ik and p_jk over the same candidate set: F_hat needs both rows.
    anchor_row = post_af3_probability(
        structural_delta(candidate_phi, phi_left.unsqueeze(1)),
        calibration["tau"], calibration["intercept"])
    partner_row = post_af3_probability(
        structural_delta(candidate_phi, phi_right.unsqueeze(1)),
        calibration["tau"], calibration["intercept"])
    mask = torch.ones_like(anchor_row, dtype=torch.bool)
    # k = j is the pair partner itself, which the candidate set holds first; k = i
    # is excluded wherever it appears among the fillers.
    mask[:, 0] = False
    mask &= picked["candidates"] != picked["left"].unsqueeze(1)
    pair_p = post_af3_probability(
        structural_delta(phi_left, phi_right),
        calibration["tau"], calibration["intercept"])
    fhat = expected_af3(anchor_row, partner_row, pair_p, mask)
    af3_term = huber_penalty((fhat - picked["target"]) / state["scale"], HUBER_DELTA).mean()

    neighbour_term = F.binary_cross_entropy(pair_p.clamp(1e-6, 1.0 - 1e-6),
                                            picked["edge"].to(pair_p.dtype))
    usage_term = usage_deficit(assignments)

    auxiliary = (LOSS_WEIGHTS["af3"] * af3_term + LOSS_WEIGHTS["neighbour"] * neighbour_term
                 + LOSS_WEIGHTS["usage"] * usage_term)
    if not torch.isfinite(auxiliary):
        raise RuntimeError("non-finite post-SID AF3 objective")
    budget = {} if budget is None else budget
    if budget.get("epoch") != epoch:
        # The weight is a share of the base objective's MAGNITUDE, not of an
        # unnormalised gradient: the structural terms are O(1) while the
        # reconstruction objective is O(1e-4), so a gradient-share scale collapses
        # to 4e-7 and the mechanism stops moving anything. The realised gradient
        # ratio is recorded and reported next to it.
        magnitude = float(base_loss.detach()) / max(float(auxiliary.detach()), 1e-12)
        scale = GRADIENT_SHARE * magnitude
        encoder = list(raw_module.encoder.parameters())
        base_encoder = gradient_norm(base_loss, encoder)
        base_codebook = gradient_norm(base_loss, codebooks)
        new_encoder = gradient_norm(auxiliary, encoder)
        new_codebook = gradient_norm(auxiliary, codebooks)
        scale_tensor = torch.tensor([scale], device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(scale_tensor, op=dist.ReduceOp.MIN)
        budget.update(epoch=epoch, scale=float(scale_tensor.item()),
                      base_encoder_norm=base_encoder, base_codebook_norm=base_codebook,
                      new_encoder_norm=new_encoder, new_codebook_norm=new_codebook)
    scale = budget["scale"]
    base_encoder = budget["base_encoder_norm"]
    base_codebook = budget["base_codebook_norm"]
    new_encoder = budget["new_encoder_norm"]
    new_codebook = budget["new_codebook_norm"]
    log = {"active": True, "epoch": epoch, "scale": scale,
           "kappa": [float(value) for value in raw_module.rq.kappa()],
           "kappa_gradient": kappa_gradient(auxiliary, raw_module.rq.kappa.theta),
           "af3": float(af3_term.detach()), "neighbour": float(neighbour_term.detach()),
           "usage": float(usage_term.detach()),
           "base_encoder_norm": base_encoder, "base_codebook_norm": base_codebook,
           "new_encoder_norm": new_encoder, "new_codebook_norm": new_codebook,
           "mean_probability": float(pair_p.detach().mean())}
    return base_loss + scale * auxiliary, log
