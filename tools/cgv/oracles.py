"""Independent mathematical oracles for CGV.

Every function here computes an expected value from DECLARED MATHEMATICS over plain data that the
adapter observed. Nothing in this module imports a trainer, a mechanism or the adapter, and
nothing here calls the implementation it is used to verify -- if it did, a wrong implementation
would be compared against itself and would pass.

That is the whole point of separating the three roles:

    adapter   executes the real mechanism and hands over raw observations
    oracle    computes what the declared mathematics says those observations must be
    engine    compares the two and reports

Conventions
-----------
* A gradient observation is a LIST of tensors, one per target parameter, in a fixed order.
* ``pcgrad(operand, reference)`` is the declared conflict projection:

      operand - (<operand, reference> / (||reference||^2 + eps)) * reference   when the dot < 0
      operand                                                                  otherwise

* An assembled encoder gradient for this family is
  ``g_base + scale * (g_target + g_other)`` where ``g_target`` may be projected.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

EPS_DEFAULT = 1e-12


# --------------------------------------------------------------------------------------
# small tensor helpers
# --------------------------------------------------------------------------------------
def _flat(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _unflatten(vector: torch.Tensor, template: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    out, offset = [], 0
    for tensor in template:
        count = tensor.numel()
        out.append(vector[offset:offset + count].view_as(tensor))
        offset += count
    return out


def _add(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [a + b for a, b in zip(left, right)]


def _scale(tensors: Sequence[torch.Tensor], factor: float) -> list[torch.Tensor]:
    return [tensor * factor for tensor in tensors]


def max_abs_deviation(left: Sequence[torch.Tensor],
                      right: Sequence[torch.Tensor]) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


# the same comparison under the name the gates use
max_abs_diff = max_abs_deviation


def l2_distance(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(left, right))))


def flat_norm(tensors: Sequence[torch.Tensor]) -> float:
    return float(torch.linalg.vector_norm(_flat(tensors)))


# --------------------------------------------------------------------------------------
# declared projection mathematics
# --------------------------------------------------------------------------------------
def pcgrad(operand: torch.Tensor, reference: torch.Tensor,
           eps: float = EPS_DEFAULT) -> torch.Tensor:
    """The declared conflict projection, on flat vectors."""
    reference_sq = float((reference * reference).sum())
    if reference_sq <= 0.0:
        return operand
    dot = float((operand * reference).sum())
    if dot >= 0.0:
        return operand
    return operand - (dot / (reference_sq + eps)) * reference


def scaled_pcgrad(operand: torch.Tensor, reference: torch.Tensor, alpha: float,
                  eps: float = EPS_DEFAULT) -> torch.Tensor:
    """Projection with a declared coefficient instead of the full removal."""
    reference_sq = float((reference * reference).sum())
    if reference_sq <= 0.0:
        return operand
    dot = float((operand * reference).sum())
    if dot >= 0.0:
        return operand
    return operand - alpha * (dot / (reference_sq + eps)) * reference


def norm_restoring(projected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Restore the original norm after projecting, as R5-06 declares."""
    original_norm = float(torch.linalg.vector_norm(original))
    projected_norm = float(torch.linalg.vector_norm(projected))
    if projected_norm <= 1e-8 * max(original_norm, 1e-12):
        return torch.zeros_like(projected)
    return projected * (original_norm / projected_norm)


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = EPS_DEFAULT) -> float:
    left_sq = float((left * left).sum())
    right_sq = float((right * right).sum())
    if left_sq <= 0.0 or right_sq <= 0.0:
        return 0.0
    return float((left * right).sum()) / (math.sqrt(left_sq) * math.sqrt(right_sq) + eps)


def ema_reference(series: Sequence[Sequence[torch.Tensor]], decay: float
                  ) -> list[torch.Tensor]:
    """The lagged EMA reference a declared recurrence produces at the final step.

    The first observation seeds the buffer; each later observation updates it. This is the
    declared mathematics of R5-09 and is recomputed here from the observed series, not read from
    the mechanism's own buffer.
    """
    if not series:
        raise ValueError("ema_reference needs at least one observation")
    buffer = list(series[0])
    for observation in series[1:]:
        buffer = [decay * old + (1.0 - decay) * new
                  for old, new in zip(buffer, observation)]
    return buffer


# --------------------------------------------------------------------------------------
# gradient assembly oracle
# --------------------------------------------------------------------------------------
REFERENCE_TERMS = ("base_loss", "l1_loss", "other_loss", "recon_loss", "quant_loss", "ema_of_base")

RULES = (
    "project_l1_against_base",
    "project_base_against_l1",
    "blockwise_l1_against_base",
    "norm_restoring_projection",
    "cosine_gated_projection",
    "unprojected",
)


def _select_reference(observations: dict[str, Any]) -> list[torch.Tensor]:
    term = observations.get("reference_term")
    if term == "base_loss":
        return observations["g_base"]
    if term == "l1_loss":
        return observations["g_l1"]
    if term == "other_loss":
        return observations["g_other"]
    if term == "recon_loss":
        return observations["g_recon"]
    if term == "quant_loss":
        return observations["g_quant"]
    if term == "ema_of_base":
        reference = observations.get("ema_reference")
        if reference is None:
            raise ValueError("reference_term 'ema_of_base' needs observations['ema_reference']")
        return reference
    raise ValueError(f"unknown reference_term {term!r}")


def assemble_gradient(observations: dict[str, Any]) -> list[torch.Tensor]:
    """The assembled target-parameter gradient the contract declares.

    ``observations`` must carry the per-term gradients (``g_base``, ``g_l1``, ``g_other``), the
    behaviour scale, and the rule parameters (``reference_term``, ``alpha``, ``cos_threshold``,
    ``blocks``, ``gate_open``, ``ema_reference``). Every value is read from the observation, so
    this oracle never inspects the mechanism.
    """
    rule = observations["rule"]
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    eps = float(observations.get("eps", EPS_DEFAULT))
    scale = float(observations["scale"])
    g_base = observations["g_base"]
    g_l1 = observations["g_l1"]
    g_other = observations["g_other"]

    if not observations.get("gate_open", True):
        # the declared activation is closed: the contract says the plain assembly must run
        return _add(g_base, _scale(_add(g_l1, g_other), scale))

    if rule == "unprojected":
        return _add(g_base, _scale(_add(g_l1, g_other), scale))

    if rule == "project_base_against_l1":
        projected = _unflatten(pcgrad(_flat(g_base), _flat(g_l1), eps), g_base)
        return _add(projected, _scale(_add(g_l1, g_other), scale))

    if rule == "blockwise_l1_against_base":
        blocks = observations.get("blocks")
        if not blocks:
            raise ValueError("rule 'blockwise_l1_against_base' needs observations['blocks']")
        projected = [tensor.clone() for tensor in g_l1]
        for block in blocks:
            operand = torch.cat([g_l1[i].reshape(-1) for i in block])
            reference = torch.cat([g_base[i].reshape(-1) for i in block])
            out = pcgrad(operand, reference, eps)
            offset = 0
            for i in block:
                count = g_l1[i].numel()
                projected[i] = out[offset:offset + count].view_as(g_l1[i])
                offset += count
        return _add(g_base, _scale(_add(projected, g_other), scale))

    if rule == "norm_restoring_projection":
        reference = _flat(_select_reference(observations))
        operand = _flat(g_l1)
        restored = norm_restoring(pcgrad(operand, reference, eps), operand)
        projected = _unflatten(restored, g_l1)
        return _add(g_base, _scale(_add(projected, g_other), scale))

    if rule == "cosine_gated_projection":
        threshold = float(observations["cos_threshold"])
        reference = _flat(_select_reference(observations))
        operand = _flat(g_l1)
        if cosine(operand, reference, eps) < threshold:
            projected = _unflatten(pcgrad(operand, reference, eps), g_l1)
        else:
            projected = list(g_l1)
        return _add(g_base, _scale(_add(projected, g_other), scale))

    # project_l1_against_base, with an optional declared coefficient
    reference = _flat(_select_reference(observations))
    operand = _flat(g_l1)
    alpha = float(observations.get("alpha", 1.0))
    if alpha != 1.0:
        projected = _unflatten(scaled_pcgrad(operand, reference, alpha, eps), g_l1)
    else:
        projected = _unflatten(pcgrad(operand, reference, eps), g_l1)
    return _add(g_base, _scale(_add(projected, g_other), scale))


def unprojected_gradient(observations: dict[str, Any]) -> list[torch.Tensor]:
    """The plain assembly, used to prove the mechanism actually changes something."""
    return assemble_gradient({**observations, "rule": "unprojected"})


# --------------------------------------------------------------------------------------
# gradient invariants
# --------------------------------------------------------------------------------------
def check_gradient_invariants(grads: Sequence[torch.Tensor],
                              invariants: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in invariants:
        name = item.get("name")
        if name == "finite":
            bad = sum(int((~torch.isfinite(g)).sum()) for g in grads)
            results.append({"name": name, "ok": bad == 0,
                            "measurement": float(bad),
                            "detail": f"{bad} non-finite entries"})
        elif name == "nonzero":
            norm = flat_norm(grads)
            results.append({"name": name, "ok": norm > float(item.get("min_norm", 0.0)),
                            "measurement": norm,
                            "detail": f"norm {norm:.6e}"})
        elif name == "norm_above":
            norm = flat_norm(grads)
            threshold = float(item.get("value", 0.0))
            results.append({"name": name, "ok": norm > threshold,
                            "measurement": norm,
                            "detail": f"norm {norm:.6e} vs {threshold:.6e}"})
        elif name == "norm_below":
            norm = flat_norm(grads)
            threshold = float(item.get("value", 0.0))
            results.append({"name": name, "ok": norm < threshold,
                            "measurement": norm,
                            "detail": f"norm {norm:.6e} vs {threshold:.6e}"})
        elif name == "orthogonal_to":
            reference = item.get("reference")
            if not isinstance(reference, list):
                raise ValueError("invariant 'orthogonal_to' needs a reference gradient list")
            dot = abs(float(_flat(grads) @ _flat(reference)))
            limit = float(item.get("max_abs_dot", 1e-6))
            results.append({"name": name, "ok": dot <= limit,
                            "measurement": dot,
                            "detail": f"|dot| {dot:.6e} vs {limit:.6e}"})
        else:
            results.append({"name": str(name), "ok": False, "measurement": float("nan"),
                            "detail": f"unknown gradient invariant {name!r}"})
    return results


# --------------------------------------------------------------------------------------
# forward invariants and checks
# --------------------------------------------------------------------------------------
def check_forward_invariants(observations: dict[str, Any],
                             invariants: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape, range, finiteness and declared-identity checks on observed forward outputs."""
    results = []
    for item in invariants:
        name = item.get("name")
        if name == "finite":
            targets = item.get("observables") or list(observations.get("tensors", {}))
            bad, checked = 0, []
            for key in targets:
                value = observations["tensors"].get(key)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    bad += int((~torch.isfinite(value)).sum())
                    checked.append(key)
            results.append({"name": name, "ok": bad == 0, "measurement": float(bad),
                            "detail": f"{bad} non-finite entries across {checked}"})
        elif name == "tokens_in_range":
            tokens = observations["tensors"].get(item.get("observable", "tokens"))
            if not isinstance(tokens, torch.Tensor):
                results.append({"name": name, "ok": False, "measurement": float("nan"),
                                "detail": "no token tensor was observed"})
                continue
            low = int(item.get("min", 0))
            high = item.get("max")
            below = int((tokens < low).sum())
            above = 0 if high is None else int((tokens > int(high)).sum())
            results.append({"name": name, "ok": below == 0 and above == 0,
                            "measurement": float(below + above),
                            "detail": f"{below} below {low}, {above} above {high}"})
        elif name == "shape":
            value = observations["tensors"].get(item.get("observable"))
            expected = item.get("expected")
            actual = list(value.shape) if isinstance(value, torch.Tensor) else None
            ok = actual == list(expected) if actual is not None else False
            results.append({"name": name, "ok": ok, "measurement": float(ok),
                            "detail": f"observed {actual}, declared {expected}"})
        elif name == "rows":
            value = observations["tensors"].get(item.get("observable"))
            expected = int(item.get("expected", 0))
            actual = int(value.shape[0]) if isinstance(value, torch.Tensor) else -1
            results.append({"name": name, "ok": actual == expected,
                            "measurement": float(actual),
                            "detail": f"{actual} rows, declared {expected}"})
        elif name == "levels":
            residuals = observations.get("residuals")
            expected = int(item.get("expected", 0))
            actual = len(residuals) if residuals is not None else -1
            results.append({"name": name, "ok": actual == expected,
                            "measurement": float(actual),
                            "detail": f"{actual} residual levels, declared {expected}"})
        elif name == "residual_identity":
            # r_{k+1} = r_k - stopgrad(q_k): the declared residual recurrence of the RQ stack.
            # The trace keeps the residual ENTERING each level, so the final remainder is absent
            # and the chain can be checked for every level whose successor was recorded.
            residuals = observations.get("residuals")
            codewords = observations.get("hard_quantized") or observations.get("codewords")
            if residuals is None or not codewords or len(residuals) < 2:
                results.append({"name": name, "ok": False, "measurement": float("nan"),
                                "detail": "observations lack the residuals/codewords pair"})
                continue
            steps = min(len(codewords), len(residuals) - 1)
            worst = 0.0
            for index in range(steps):
                worst = max(worst, float(
                    (residuals[index] - codewords[index] - residuals[index + 1]).abs().max()))
            results.append({"name": name, "ok": worst <= float(item.get("tol", 1e-5)),
                            "measurement": worst,
                            "detail": f"max |r_k - q_k - r_(k+1)| over {steps} level(s) = "
                                      f"{worst:.3e}"})
        elif name == "behavior_trace_hard_forward_exact":
            traces = observations.get("traces")
            reference = observations.get("reference_traces")
            if traces is None or not reference:
                results.append({"name": name, "ok": False, "measurement": float("nan"),
                                "detail": "the arm's trace or the untouched control walk is "
                                          "missing"})
                continue
            worst = max(float((a - b).abs().max())
                        for a, b in zip(traces, reference))
            tol = float(item.get("atol", 1e-5))
            results.append({"name": name, "ok": worst <= tol, "measurement": worst,
                            "detail": f"max |arm trace - control trace| = {worst:.3e} "
                                      f"(tol {tol:.1e})"})
        elif name == "hard_sid_unchanged_by_backward_surrogate":
            observed = observations["tensors"].get("tokens")
            reference = observations.get("reference_tokens")
            if reference is not None and observed is not None \
                    and reference.shape != observed.shape and reference.dim() == 2:
                # the control walk is level-major, the export path is item-major
                reference = reference.t()
            if observed is None or reference is None:
                results.append({"name": name, "ok": False, "measurement": float("nan"),
                                "detail": "no hard SID observation was supplied"})
                continue
            differing = int((observed != reference).sum())
            results.append({"name": name, "ok": differing == 0,
                            "measurement": float(differing),
                            "detail": f"{differing} hard SID entries differ from the control walk"})
        elif name == "jacobian_identity":
            # declared rule dr_{k+1}/dr_k = I: verified by the engine through autograd, the
            # observation carries the measured deviation
            measured = observations.get("jacobian_identity_deviation")
            if measured is None:
                results.append({"name": name, "ok": False, "measurement": float("nan"),
                                "detail": "no autograd Jacobian probe was supplied"})
                continue
            results.append({"name": name,
                            "ok": float(measured) <= float(item.get("tol", 1e-5)),
                            "measurement": float(measured),
                            "detail": f"max deviation from the identity Jacobian {measured:.3e}"})
        else:
            results.append({"name": str(name), "ok": False, "measurement": float("nan"),
                            "detail": f"unknown forward invariant {name!r}"})
    return results


def compare_forward(observations: dict[str, Any], expected: dict[str, Any],
                    atol: float) -> list[dict[str, Any]]:
    """Compare observed forward outputs with an oracle's independently computed expectation."""
    results = []
    for key, want in expected.items():
        got = observations["tensors"].get(key)
        if not isinstance(got, torch.Tensor):
            results.append({"name": f"forward_oracle:{key}", "ok": False,
                            "measurement": float("nan"),
                            "detail": f"no observed tensor {key!r}"})
            continue
        want = want.detach()
        if got.shape != want.shape:
            results.append({"name": f"forward_oracle:{key}", "ok": False,
                            "measurement": float("nan"),
                            "detail": f"shape {tuple(got.shape)} vs declared {tuple(want.shape)}"})
            continue
        deviation = float((got - want).abs().max())
        results.append({"name": f"forward_oracle:{key}", "ok": deviation <= atol,
                        "measurement": deviation,
                        "detail": f"max |observed - oracle| = {deviation:.3e} (atol {atol:.1e})"})
    return results


# --------------------------------------------------------------------------------------
# update oracle
# --------------------------------------------------------------------------------------
def adamw_first_step(param: torch.Tensor, clipped_grad: torch.Tensor, lr: float,
                     weight_decay: float, eps: float = 1e-8) -> torch.Tensor:
    """Closed-form AdamW first step, derived independently of torch.optim internals.

    With bias correction ``v_hat == g**2`` on the first step, so ``g_hat == g/(|g| + eps)`` and
    the decoupled weight decay multiplies the parameter before the update is subtracted.
    """
    direction = clipped_grad / (clipped_grad.abs() + eps)
    return param * (1.0 - lr * weight_decay) - lr * direction


def sgd_step(param: torch.Tensor, clipped_grad: torch.Tensor, lr: float,
             weight_decay: float = 0.0, momentum: float = 0.0) -> torch.Tensor:
    if momentum != 0.0:
        raise ValueError("the independent SGD oracle implements momentum 0 only")
    return param * (1.0 - lr * weight_decay) - lr * clipped_grad


UPDATE_ORACLES = {"AdamW": adamw_first_step, "SGD": sgd_step}


def optimizer_update(observations: dict[str, Any]) -> list[torch.Tensor]:
    """Expected parameters after one declared optimizer step."""
    name = observations["optimizer"]
    if name not in UPDATE_ORACLES:
        raise ValueError(f"no independent update oracle for optimizer {name!r}")
    fn = UPDATE_ORACLES[name]
    lr = float(observations["lr"])
    weight_decay = float(observations.get("weight_decay", 0.0))
    expected = []
    for param, grad in zip(observations["params_before"], observations["clipped_grads"]):
        if grad is None:
            expected.append(param.clone())
            continue
        expected.append(fn(param, grad, lr, weight_decay))
    return expected


# --------------------------------------------------------------------------------------
# distributed aggregation oracle
# --------------------------------------------------------------------------------------
def ddp_mean_pool(rank_grads: Sequence[Sequence[torch.Tensor]],
                  weights: Sequence[float] | None = None) -> list[torch.Tensor]:
    """The declared DDP aggregation: the mean over ranks, optionally batch-weighted.

    Reconstructed from the per-rank gradients handed over by the driver, never read from the
    all-reduce the run performed -- otherwise a double all-reduce would agree with itself.
    """
    if not rank_grads:
        raise ValueError("ddp_mean_pool needs at least one rank")
    if weights is None:
        weights = [1.0 / len(rank_grads)] * len(rank_grads)
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("aggregation weights must sum to a positive value")
    pooled = [torch.zeros_like(tensor) for tensor in rank_grads[0]]
    for rank, weight in zip(rank_grads, weights):
        for index, tensor in enumerate(rank):
            pooled[index] = pooled[index] + tensor * (weight / total)
    return pooled


def ddp_sum_pool(rank_grads: Sequence[Sequence[torch.Tensor]]) -> list[torch.Tensor]:
    pooled = [torch.zeros_like(tensor) for tensor in rank_grads[0]]
    for rank in rank_grads:
        for index, tensor in enumerate(rank):
            pooled[index] = pooled[index] + tensor
    return pooled


AGGREGATIONS = {"ddp_mean": ddp_mean_pool, "ddp_sum": ddp_sum_pool}


def aggregate(observations: dict[str, Any]) -> list[torch.Tensor]:
    rule = observations["aggregation"]
    if rule not in AGGREGATIONS:
        raise ValueError(f"no independent aggregation oracle for {rule!r}")
    if rule == "ddp_mean" and observations.get("weights") is not None:
        return ddp_mean_pool(observations["rank_grads"], observations["weights"])
    return AGGREGATIONS[rule](observations["rank_grads"])


# --------------------------------------------------------------------------------------
# residual-trace mechanisms: the independent reference for a declared backward rule
# --------------------------------------------------------------------------------------
TRACE_RULES = ("half_ste_residual", "direct_latent_bridge", "soft_codeword_surrogate")

# Rules whose trace-to-latent Jacobian is DATA DEPENDENT rather than a constant coefficient.
# Demanding a constant level Jacobian of these would be demanding a representation that does not
# exist, so the gradient gate verifies them by the VJP the oracle derives instead -- which is a
# different check, not a weaker one.
VJP_RULES = ("soft_codeword_surrogate",)


def _weighted_latent_pullback(obs: dict, coefficients) -> list[torch.Tensor]:
    """Sum_k c_k * J_encoder^T u_k, with the coefficients read from the DECLARATION.

    The chain rule factor that belongs to the mechanism is exactly ``c_k``: the Jacobian of that
    level's trace with respect to the latent. Everything else here is the encoder's own autograd,
    which the mechanism does not touch. So the oracle can disagree with the arm about the rule
    without sharing any of its code.
    """
    total = None
    for coefficient, cotangent in zip(coefficients, obs["u_levels"]):
        if coefficient == 0.0:
            continue
        contribution = obs["latent_to_encoder"](cotangent * float(coefficient))
        total = contribution if total is None else [a + b for a, b in zip(total, contribution)]
    if total is None:
        return [torch.zeros_like(p) for p in obs["encoder"]]
    return total


def declared_coefficients(observations: dict) -> list[float]:
    coefficients = observations.get("coefficients")
    if not coefficients:
        raise ValueError(
            "the declaration must state mechanism.mathematics.expected_level_to_encoder_jacobians")
    traces = observations["traces"]
    if len(coefficients) != len(traces):
        raise ValueError(
            f"the declaration lists {len(coefficients)} level Jacobians but the arm's trace has "
            f"{len(traces)} levels")
    return [float(value) for value in coefficients]


def half_ste_behavior_gradient(observations: dict) -> list[torch.Tensor]:
    """r6_02: q_mix = stopgrad(q) + w*(q - stopgrad(q)), r_next = r - q_mix.

    The declared Jacobian per level is read from the declaration, so declaring 1.0 (the
    full-detach rule) or 0.0 (the legacy rule) instead of 0.5 changes the expectation rather than
    quietly passing.
    """
    return _weighted_latent_pullback(observations, declared_coefficients(observations))


def direct_latent_bridge_gradient(observations: dict) -> list[torch.Tensor]:
    """r6_03: the CE-facing deep traces carry an extra b*(z - stopgrad(z)) bridge.

    Same pullback, different declared coefficients ([1, 0.5, 0.5] rather than [1, 0.5, 0.25]).
    The rule is distinguished by its coefficients and by the bridge's own check, not by a name.
    """
    return _weighted_latent_pullback(observations, declared_coefficients(observations))


def _soft_codeword(trace: torch.Tensor, codebook: torch.Tensor, floor: float):
    """The declared soft surrogate, rebuilt from observations with its OWN autograd.

    d_j = ||r - c_j||^2, s = stopgrad(mean_j d_j) clamped, p = softmax(-d_j / s),
    q_soft = sum_j p_j c_j. Rebuilding it here, rather than reading the arm's tensor, is what
    makes the expectation independent of the arm's implementation.
    """
    distances = torch.cdist(trace, codebook, p=2).square()
    scale = distances.mean(dim=-1, keepdim=True).detach().clamp_min(floor)
    weights = torch.softmax(-distances / scale, dim=-1)
    return weights @ codebook


def soft_codeword_behavior_gradient(observations: dict) -> dict[str, Any]:
    """r6_04: hard forward, soft codeword backward.

    ``r_next = r - (q_hard.detach() + (q_soft - q_soft.detach()))``, so the backward is
    ``dr_next = dr - dq_soft``: the soft surrogate REPLACES the straight-through identity. That
    makes the trace-to-latent Jacobian data dependent -- ``I - J(q_soft, r)`` -- so there is no
    constant level coefficient to declare, and asking for one would be demanding a representation
    that does not exist.

    The expectation is therefore the VJP of the DECLARED recurrence, rebuilt here from the
    observed level-0 trace, the raw codebooks and the observed hard assignment. The arm's own graph
    is never differentiated: the surrogate, the normalisation and the composition are all this
    function's, so a wrong implementation disagrees with it.
    """
    traces = observations["traces"]
    codebooks = observations["codebooks"]
    indices = observations["trace_indices"]
    cotangents = observations["u_levels"]
    floor = float(observations.get("soft_normalizer_floor", 1e-8))

    latent = traces[0].detach().requires_grad_(True)
    local_codebooks = [codebook.detach().requires_grad_(True) for codebook in codebooks]

    # The observed trace is [latent, r1, r2]: the FIRST entry is the residual entering level 0,
    # not the result of the first subtraction. Starting the rebuild one step late would pair every
    # cotangent with the wrong level, which shows up as a plausible-looking but wrong expectation.
    rebuilt = [latent]
    residual = latent
    for level, codebook in enumerate(local_codebooks):
        soft = _soft_codeword(residual, codebook, floor)
        hard = codebook[indices[level]].detach()
        residual = residual - (hard + (soft - soft.detach()))
        rebuilt.append(residual)

    encoder_total = None
    codebook_total = None
    for level, cotangent in enumerate(cotangents):
        if level >= len(rebuilt):
            break
        inputs = [latent] + local_codebooks
        found = torch.autograd.grad((rebuilt[level] * cotangent).sum(), inputs,
                                    retain_graph=True, allow_unused=True)
        grad_latent = found[0]
        grad_codebooks = [torch.zeros_like(p) if g is None else g
                          for p, g in zip(local_codebooks, found[1:])]
        if grad_latent is not None:
            encoder_part = observations["latent_to_encoder"](grad_latent)
            encoder_total = (encoder_part if encoder_total is None else
                             [a + b for a, b in zip(encoder_total, encoder_part)])
        codebook_total = (grad_codebooks if codebook_total is None else
                          [a + b for a, b in zip(codebook_total, grad_codebooks)])

    if encoder_total is None:
        encoder_total = [torch.zeros_like(p) for p in observations["encoder"]]
    if codebook_total is None:
        codebook_total = [torch.zeros_like(p) for p in observations.get("codebook_parameters", [])]
    return {"encoder": encoder_total, "codebooks": codebook_total}


def trace_gradient(observations: dict) -> Any:
    """Dispatch the declared trace rule to its independent reference."""
    rule = observations["rule"]
    if rule == "half_ste_residual":
        return half_ste_behavior_gradient(observations)
    if rule == "direct_latent_bridge":
        return direct_latent_bridge_gradient(observations)
    if rule == "soft_codeword_surrogate":
        return soft_codeword_behavior_gradient(observations)
    return assemble_gradient(observations)


def hard_residual_trace_forward(observations: dict) -> dict[str, Any]:
    """The forward the declaration promises: the legacy hard residual recurrence, bit for bit.

    The arm's backward surrogate must be invisible here. Two independent checks:

    1. the arm's trace and tokens equal the ones the untouched control walk produces;
    2. algebraically, ``traces[k] - traces[k+1]`` equals the hard codeword the quantiser selected,
       which needs the codebooks and the selected indices and nothing else.

    The control walk is used because the last level assigns codewords with Sinkhorn, whose
    assignment cannot be re-derived from distances alone by an outside observer. Check 2 does not
    depend on it.
    """
    traces = observations["traces"]
    codebooks = observations["codebooks"]
    indices = observations["trace_indices"]
    reference = observations["reference_traces"]
    # the control walk stacks per level ([levels, N]); get_indices reports per item
    # ([N, levels]), so the comparison is made in the item-major layout the export path uses
    expected = {"tokens": observations["reference_tokens"].t().contiguous()}
    if reference is not None:
        expected["residuals"] = torch.stack([trace.detach() for trace in reference], dim=0)
    return expected


def hard_assignment_algebra(observations: dict, tol: float = 1e-5) -> dict[str, Any]:
    """traces[k] - traces[k+1] == codebook_k[indices[k]], the hard subtraction with no surrogate."""
    traces = observations["traces"]
    codebooks = observations["codebooks"]
    indices = observations["trace_indices"]
    worst = 0.0
    checked = 0
    for level in range(len(traces) - 1):
        hard = codebooks[level][indices[level]]
        worst = max(worst, float((traces[level] - traces[level + 1] - hard).abs().max()))
        checked += 1
    return {"name": "hard_assignment_algebra", "ok": worst <= tol and checked > 0,
            "measurement": worst,
            "detail": f"max |r_k - r_(k+1) - hard codeword| over {checked} level(s) = {worst:.3e}"}
