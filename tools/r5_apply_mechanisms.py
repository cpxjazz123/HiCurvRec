"""Apply the ten R5 sole-mechanism deltas onto forked TIGER_r5_* trees (#61-#70).

Each arm starts as a copy of TIGER_r4_08 and receives exactly one PCGrad-coordination
change. Re-running is idempotent for already-patched markers.

    python3 tools/r5_apply_mechanisms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TREES = REPO_ROOT / "stage2_RQ-VAE"

PARENT_CONST = """R4_08_PCGRAD_ENC   = True
R4_08_EPS          = 1e-12
R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
"""

PARENT_PROJECT = '''def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
    """PCGrad step for one parameter: drop g_base when the two gradients conflict.

    ``g_l1 <- g_l1 - (<g_l1, g_base> / (||g_base||^2 + eps)) * g_base`` when the dot
    product is negative, otherwise g_l1 passes through untouched. A zero or non-finite
    base gradient disables the projection and fails closed rather than emitting NaNs.
    """
    if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
        raise RuntimeError("R4-08 received a non-finite gradient; failing closed")
    base_sq = float((g_base * g_base).sum())
    if base_sq <= 0.0:
        return g_l1
    dot = float((g_l1 * g_base).sum())
    if dot >= 0.0:
        return g_l1
    R4_08_PROJECTION_TRACE["active_count"] += 1
    return g_l1 - (dot / (base_sq + R4_08_EPS)) * g_base
'''

PARENT_APPLY = '''def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
) -> None:
    """Overwrite the encoder gradients with the conflict-projected assembly.

    ``g_base + w_b/3 * (g'_L1 + g_L2 + g_L3)`` with the averaging the loop actually uses:
    the behaviour term contributes ``behavior_weight / valid_levels * (L1 + L2 + L3)``,
    and ``g'_L1`` is the single PCGrad projection of the full encoder ``g_L1`` vector
    against the full encoder ``g_base`` vector. Decoder and codebook gradients are never
    touched.
    """
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    if R4_08_PCGRAD_ENC:
        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
        flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
        flat_proj = r4_project_conflict(flat_l1, flat_base)
        g_l1_proj = []
        offset = 0
        for grad in g_l1:
            count = grad.numel()
            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
            offset += count
    else:
        g_l1_proj = g_l1
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_proj_grad, other_grad in zip(
        encoder_parameters, g_base, g_l1_proj, g_other
    ):
        assembled = base_grad + scale * (l1_proj_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''

PARENT_CALL = """            if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
                # One backward still produces the baseline gradient for every parameter,
                # including the decoder and the codebooks. Only then is the encoder
                # gradient replaced by the conflict-projected assembly.
                loss.backward(retain_graph=True)
                _r4_08_apply_projection(
                    base_loss,
                    behavior_l1_loss,
                    behavior_other_loss,
                    list(raw_module.encoder.parameters()),
                    epoch_weight,
                    valid_levels=behavior_valid_levels,
                )
            else:
                loss.backward()
"""


def _replace_once(text: str, old: str, new: str, arm: str, label: str) -> str:
    # Prefer an exact "already applied" check on the full new blob. Do NOT use
    # ``old not in text`` alone: ``old`` is often a prefix of ``new``.
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"{arm}: missing anchor for {label}")
    return text.replace(old, new, 1)


def _write(arm: str, text: str) -> None:
    path = TREES / f"TIGER_{arm}" / "train_rqvae.py"
    path.write_text(text, encoding="utf-8")
    print(f"{arm}: wrote {path}")


def _load(arm: str) -> str:
    return (TREES / f"TIGER_{arm}" / "train_rqvae.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- helpers shared by several arms

APPLY_SIG_EXT = '''def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:'''


def _unflat(flat, like):
    return '''        g_out = []
        offset = 0
        for grad in like:
            count = grad.numel()
            g_out.append(flat[offset : offset + count].view_as(grad))
            offset += count
        return g_out
'''


# --------------------------------------------------------------------------- arm patches

def patch_r5_01(text: str) -> str:
    """Reverse PCGrad: project g_base against g_L1."""
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST
        + "R5_01_REVERSE_PCGRAD = True  # issue #61: protect L1, project base\n",
        "r5_01",
        "const",
    )
    new_project = '''def r4_project_conflict(g_operand: torch.Tensor, g_reference: torch.Tensor) -> torch.Tensor:
    """R5-01 reverse PCGrad: project the *base* operand against the L1 reference.

    When called as ``r4_project_conflict(g_base, g_l1)`` this removes the L1-conflicting
    component from g_base. The helper keeps the R4-08 name so the call sites and the
    RED suite still locate a single projection entry point.
    """
    if not bool(torch.isfinite(g_operand).all()) or not bool(torch.isfinite(g_reference).all()):
        raise RuntimeError("R5-01 received a non-finite gradient; failing closed")
    ref_sq = float((g_reference * g_reference).sum())
    if ref_sq <= 0.0:
        return g_operand
    dot = float((g_operand * g_reference).sum())
    if dot >= 0.0:
        return g_operand
    R4_08_PROJECTION_TRACE["active_count"] += 1
    return g_operand - (dot / (ref_sq + R4_08_EPS)) * g_reference
'''
    text = _replace_once(text, PARENT_PROJECT, new_project, "r5_01", "project")
    new_apply = '''def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:
    """R5-01: assemble g'_base + scale*(g_L1 + g_other); only g_base is projected."""
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    if R4_08_PCGRAD_ENC and R5_01_REVERSE_PCGRAD:
        flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
        flat_proj = r4_project_conflict(flat_base, flat_l1)
        g_base_proj = []
        offset = 0
        for grad in g_base:
            count = grad.numel()
            g_base_proj.append(flat_proj[offset : offset + count].view_as(grad))
            offset += count
    else:
        g_base_proj = g_base
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_grad, other_grad in zip(
        encoder_parameters, g_base_proj, g_l1, g_other
    ):
        assembled = base_grad + scale * (l1_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''
    text = _replace_once(text, PARENT_APPLY, new_apply, "r5_01", "apply")
    return text


def patch_r5_02(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST + "R5_02_RECON_REF = True  # issue #62: project L1 vs recon-only\n",
        "r5_02",
        "const",
    )
    new_apply = '''def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:
    """R5-02: project g_L1 against g_recon; still assemble with full unprojected g_base."""
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=True)
    )
    if recon_loss is None:
        raise RuntimeError("R5-02 requires recon_loss as the projection reference")
    g_ref = _r4_08_global_average(
        _r4_08_encoder_gradients(recon_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    if R4_08_PCGRAD_ENC and R5_02_RECON_REF:
        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
        flat_ref = torch.cat([grad.reshape(-1) for grad in g_ref])
        flat_proj = r4_project_conflict(flat_l1, flat_ref)
        g_l1_proj = []
        offset = 0
        for grad in g_l1:
            count = grad.numel()
            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
            offset += count
    else:
        g_l1_proj = g_l1
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_proj_grad, other_grad in zip(
        encoder_parameters, g_base, g_l1_proj, g_other
    ):
        assembled = base_grad + scale * (l1_proj_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''
    text = _replace_once(text, PARENT_APPLY, new_apply, "r5_02", "apply")
    text = _replace_once(
        text,
        PARENT_CALL,
        """            if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
                loss.backward(retain_graph=True)
                _r4_08_apply_projection(
                    base_loss,
                    behavior_l1_loss,
                    behavior_other_loss,
                    list(raw_module.encoder.parameters()),
                    epoch_weight,
                    valid_levels=behavior_valid_levels,
                    recon_loss=recon_loss,
                )
            else:
                loss.backward()
""",
        "r5_02",
        "call",
    )
    return text


def patch_r5_03(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST + "R5_03_QUANT_REF = True  # issue #63: project L1 vs quant-only\n",
        "r5_03",
        "const",
    )
    new_apply = '''def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:
    """R5-03: project g_L1 against g_quant = grad(base - recon); assemble with full g_base."""
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    if recon_loss is None:
        raise RuntimeError("R5-03 requires recon_loss to form the quant reference")
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=True)
    )
    quant_loss = base_loss - recon_loss
    g_ref = _r4_08_global_average(
        _r4_08_encoder_gradients(quant_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    if R4_08_PCGRAD_ENC and R5_03_QUANT_REF:
        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
        flat_ref = torch.cat([grad.reshape(-1) for grad in g_ref])
        flat_proj = r4_project_conflict(flat_l1, flat_ref)
        g_l1_proj = []
        offset = 0
        for grad in g_l1:
            count = grad.numel()
            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
            offset += count
    else:
        g_l1_proj = g_l1
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_proj_grad, other_grad in zip(
        encoder_parameters, g_base, g_l1_proj, g_other
    ):
        assembled = base_grad + scale * (l1_proj_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''
    text = _replace_once(text, PARENT_APPLY, new_apply, "r5_03", "apply")
    text = _replace_once(
        text,
        PARENT_CALL,
        """            if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
                loss.backward(retain_graph=True)
                _r4_08_apply_projection(
                    base_loss,
                    behavior_l1_loss,
                    behavior_other_loss,
                    list(raw_module.encoder.parameters()),
                    epoch_weight,
                    valid_levels=behavior_valid_levels,
                    recon_loss=recon_loss,
                )
            else:
                loss.backward()
""",
        "r5_03",
        "call",
    )
    return text


def patch_r5_04(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST + "R5_04_BLOCKWISE = True  # issue #64: per-Linear block PCGrad\n",
        "r5_04",
        "const",
    )
    new_apply = '''def _r5_04_linear_blocks(encoder_module, encoder_parameters):
    """Group each nn.Linear's weight+bias as one block; leftover params stay alone."""
    param_ids = {id(p): i for i, p in enumerate(encoder_parameters)}
    used = set()
    blocks = []
    modules = []
    if encoder_module is not None:
        modules = list(encoder_module.modules())
    for module in modules:
        if not isinstance(module, torch.nn.Linear):
            continue
        idxs = []
        for tensor in (module.weight, module.bias):
            if tensor is None:
                continue
            idx = param_ids.get(id(tensor))
            if idx is not None and idx not in used:
                idxs.append(idx)
                used.add(idx)
        if idxs:
            blocks.append(idxs)
    for i in range(len(encoder_parameters)):
        if i not in used:
            blocks.append([i])
    return blocks


def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:
    """R5-04: same PCGrad rule, applied independently per Linear weight+bias block."""
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    g_l1_proj = [g.clone() for g in g_l1]
    if R4_08_PCGRAD_ENC and R5_04_BLOCKWISE:
        for block in _r5_04_linear_blocks(encoder_module, encoder_parameters):
            flat_l1 = torch.cat([g_l1[i].reshape(-1) for i in block])
            flat_base = torch.cat([g_base[i].reshape(-1) for i in block])
            flat_proj = r4_project_conflict(flat_l1, flat_base)
            offset = 0
            for i in block:
                count = g_l1[i].numel()
                g_l1_proj[i] = flat_proj[offset : offset + count].view_as(g_l1[i])
                offset += count
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_proj_grad, other_grad in zip(
        encoder_parameters, g_base, g_l1_proj, g_other
    ):
        assembled = base_grad + scale * (l1_proj_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''
    text = _replace_once(text, PARENT_APPLY, new_apply, "r5_04", "apply")
    text = _replace_once(
        text,
        PARENT_CALL,
        """            if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
                loss.backward(retain_graph=True)
                _r4_08_apply_projection(
                    base_loss,
                    behavior_l1_loss,
                    behavior_other_loss,
                    list(raw_module.encoder.parameters()),
                    epoch_weight,
                    valid_levels=behavior_valid_levels,
                    encoder_module=raw_module.encoder,
                )
            else:
                loss.backward()
""",
        "r5_04",
        "call",
    )
    return text


def patch_r5_05(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST + "R5_05_ALPHA = 0.5  # issue #65: half-strength conflict projection\n",
        "r5_05",
        "const",
    )
    new_project = '''def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
    """R5-05 half-strength PCGrad: subtract only alpha=0.5 of the conflicting component."""
    if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
        raise RuntimeError("R5-05 received a non-finite gradient; failing closed")
    base_sq = float((g_base * g_base).sum())
    if base_sq <= 0.0:
        return g_l1
    dot = float((g_l1 * g_base).sum())
    if dot >= 0.0:
        return g_l1
    R4_08_PROJECTION_TRACE["active_count"] += 1
    return g_l1 - R5_05_ALPHA * (dot / (base_sq + R4_08_EPS)) * g_base
'''
    return _replace_once(text, PARENT_PROJECT, new_project, "r5_05", "project")


def patch_r5_06(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST + "R5_06_NORM_PRESERVE = True  # issue #66: restore ||g_L1|| after PCGrad\n",
        "r5_06",
        "const",
    )
    new_project = '''def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
    """R5-06: R4-08 PCGrad followed by norm restoration to the original ||g_L1||."""
    if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
        raise RuntimeError("R5-06 received a non-finite gradient; failing closed")
    base_sq = float((g_base * g_base).sum())
    if base_sq <= 0.0:
        return g_l1
    dot = float((g_l1 * g_base).sum())
    if dot >= 0.0:
        return g_l1
    R4_08_PROJECTION_TRACE["active_count"] += 1
    projected = g_l1 - (dot / (base_sq + R4_08_EPS)) * g_base
    if not R5_06_NORM_PRESERVE:
        return projected
    orig_norm = float(torch.linalg.vector_norm(g_l1))
    proj_norm = float(torch.linalg.vector_norm(projected))
    floor = 1e-8 * max(orig_norm, 1e-12)
    if proj_norm <= floor:
        return torch.zeros_like(g_l1)
    return projected * (orig_norm / proj_norm)
'''
    return _replace_once(text, PARENT_PROJECT, new_project, "r5_06", "project")


def patch_r5_07(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST
        + "R5_07_COS_THRESHOLD = -0.2  # issue #67: project only for strong conflicts\n",
        "r5_07",
        "const",
    )
    new_project = '''def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
    """R5-07: apply the original R4-08 projection only when cos(g_L1,g_base) < -0.2."""
    if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
        raise RuntimeError("R5-07 received a non-finite gradient; failing closed")
    base_sq = float((g_base * g_base).sum())
    l1_sq = float((g_l1 * g_l1).sum())
    if base_sq <= 0.0 or l1_sq <= 0.0:
        return g_l1
    dot = float((g_l1 * g_base).sum())
    cos = dot / ((base_sq ** 0.5) * (l1_sq ** 0.5) + R4_08_EPS)
    # Strict inequality: cos == -0.2 must NOT project.
    if cos >= R5_07_COS_THRESHOLD:
        return g_l1
    R4_08_PROJECTION_TRACE["active_count"] += 1
    return g_l1 - (dot / (base_sq + R4_08_EPS)) * g_base
'''
    return _replace_once(text, PARENT_PROJECT, new_project, "r5_07", "project")


def patch_r5_08(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST
        + "R5_08_LATE_START_EPOCH = 2250  # issue #68: PCGrad only for epoch>=2250\n",
        "r5_08",
        "const",
    )
    text = _replace_once(
        text,
        PARENT_CALL,
        """            if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
                if epoch >= R5_08_LATE_START_EPOCH:
                    loss.backward(retain_graph=True)
                    _r4_08_apply_projection(
                        base_loss,
                        behavior_l1_loss,
                        behavior_other_loss,
                        list(raw_module.encoder.parameters()),
                        epoch_weight,
                        valid_levels=behavior_valid_levels,
                    )
                else:
                    # Epochs 1..2249: full Resample-style unprojected backward inside the
                    # R4-08 tree (behavior loss still present; only the projection is off).
                    loss.backward()
            else:
                loss.backward()
""",
        "r5_08",
        "call",
    )
    return text


def patch_r5_09(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST
        + "R5_09_EMA_DECAY = 0.9  # issue #69: EMA-smoothed base as PCGrad reference\n"
        + "R5_09_EMA_STATE = {\"buffers\": None}\n",
        "r5_09",
        "const",
    )
    new_apply = '''def _r5_09_update_ema(g_base):
    """Maintain m_t = 0.9 m_{t-1} + 0.1 g_base_t; first step seeds with g_base."""
    buffers = R5_09_EMA_STATE["buffers"]
    if buffers is None:
        R5_09_EMA_STATE["buffers"] = [g.detach().clone() for g in g_base]
        return R5_09_EMA_STATE["buffers"]
    decay = float(R5_09_EMA_DECAY)
    updated = []
    for prev, cur in zip(buffers, g_base):
        updated.append(decay * prev + (1.0 - decay) * cur.detach())
    R5_09_EMA_STATE["buffers"] = updated
    return updated


def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:
    """R5-09: project g_L1 against EMA(g_base); assemble with current unprojected g_base."""
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    g_ref = _r5_09_update_ema(g_base)
    if R4_08_PCGRAD_ENC:
        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
        flat_ref = torch.cat([grad.reshape(-1) for grad in g_ref])
        flat_proj = r4_project_conflict(flat_l1, flat_ref)
        g_l1_proj = []
        offset = 0
        for grad in g_l1:
            count = grad.numel()
            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
            offset += count
    else:
        g_l1_proj = g_l1
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_proj_grad, other_grad in zip(
        encoder_parameters, g_base, g_l1_proj, g_other
    ):
        assembled = base_grad + scale * (l1_proj_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''
    return _replace_once(text, PARENT_APPLY, new_apply, "r5_09", "apply")


def patch_r5_10(text: str) -> str:
    text = _replace_once(
        text,
        PARENT_CONST,
        PARENT_CONST
        + "R5_10_INTERLEVEL = True  # issue #70: project L1 against L2+L3 behavior grads\n",
        "r5_10",
        "const",
    )
    new_apply = '''def _r4_08_apply_projection(
    base_loss: torch.Tensor,
    l1_loss: torch.Tensor,
    other_loss: torch.Tensor | None,
    encoder_parameters,
    behavior_weight: float,
    valid_levels: int = 3,
    recon_loss: torch.Tensor | None = None,
    epoch: int | None = None,
    encoder_module=None,
) -> None:
    """R5-10: project g_L1 against g_other=grad(L2+L3); assemble with full g_base + g_other."""
    encoder_parameters = list(encoder_parameters)
    if not encoder_parameters:
        return
    has_other = other_loss is not None and bool(other_loss.requires_grad)
    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
    g_l1 = _r4_08_global_average(
        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
    )
    if has_other:
        g_other = _r4_08_global_average(
            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
        )
    else:
        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
    if R4_08_PCGRAD_ENC and R5_10_INTERLEVEL and has_other:
        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
        flat_ref = torch.cat([grad.reshape(-1) for grad in g_other])
        flat_proj = r4_project_conflict(flat_l1, flat_ref)
        g_l1_proj = []
        offset = 0
        for grad in g_l1:
            count = grad.numel()
            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
            offset += count
    else:
        g_l1_proj = g_l1
    scale = float(behavior_weight) / float(max(valid_levels, 1))
    for parameter, base_grad, l1_proj_grad, other_grad in zip(
        encoder_parameters, g_base, g_l1_proj, g_other
    ):
        assembled = base_grad + scale * (l1_proj_grad + other_grad)
        if parameter.grad is None:
            parameter.grad = assembled
        else:
            parameter.grad.copy_(assembled)
'''
    return _replace_once(text, PARENT_APPLY, new_apply, "r5_10", "apply")


PATCHERS = {
    "r5_01": patch_r5_01,
    "r5_02": patch_r5_02,
    "r5_03": patch_r5_03,
    "r5_04": patch_r5_04,
    "r5_05": patch_r5_05,
    "r5_06": patch_r5_06,
    "r5_07": patch_r5_07,
    "r5_08": patch_r5_08,
    "r5_09": patch_r5_09,
    "r5_10": patch_r5_10,
}


def main() -> int:
    for arm, patcher in PATCHERS.items():
        tree = TREES / f"TIGER_{arm}"
        if not tree.is_dir():
            raise SystemExit(f"missing forked tree {tree}; run tools/r5_fork_arms.py first")
        text = _load(arm)
        # Lightweight identity check without importing torch.
        if f'EXPERIMENT_ARM = "{arm}"' not in text and f"EXPERIMENT_ARM = '{arm}'" not in text:
            # after fork the arm string uses double quotes
            if f'EXPERIMENT_ARM = "{arm}"' not in text:
                # tolerate the provenance-comment form
                if f'EXPERIMENT_ARM = "{arm}"' not in text and f'EXPERIMENT_ARM = "{arm}"  #' not in text:
                    if f'EXPERIMENT_ARM = "{arm}"' not in text:
                        pass
        if f'EXPERIMENT_ARM = "{arm}"' not in text:
            raise SystemExit(f"{arm}: EXPERIMENT_ARM identity missing before patch")
        patched = patcher(text)
        _write(arm, patched)
        # Verify the sole-mechanism marker landed.
        markers = {
            "r5_01": "R5_01_REVERSE_PCGRAD",
            "r5_02": "R5_02_RECON_REF",
            "r5_03": "R5_03_QUANT_REF",
            "r5_04": "R5_04_BLOCKWISE",
            "r5_05": "R5_05_ALPHA",
            "r5_06": "R5_06_NORM_PRESERVE",
            "r5_07": "R5_07_COS_THRESHOLD",
            "r5_08": "R5_08_LATE_START_EPOCH",
            "r5_09": "R5_09_EMA_DECAY",
            "r5_10": "R5_10_INTERLEVEL",
        }
        if markers[arm] not in patched:
            raise SystemExit(f"{arm}: mechanism marker {markers[arm]} missing after patch")
        print(f"{arm}: patched ok ({markers[arm]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
