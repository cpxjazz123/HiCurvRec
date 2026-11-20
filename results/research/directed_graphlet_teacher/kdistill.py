"""Issue #222 B: the distillation loss, written so the argument order is visible.

    L_KD = T^2 * KL( softmax(teacher/T) || softmax(student/T) )

`F.kl_div` takes (log-probabilities, probabilities) -- i.e. student first, teacher
second -- and the temperature squared keeps the gradient scale comparable across
T.  A reversed call still returns a plausible number (KL is not symmetric), which
is exactly why the wrong direction is a silent failure; `tests/test_student_interface.py`
pins the asymmetry and the sign of the gradient on the teacher's own class.

Optional masks: orbit targets exist only for nodes whose instance has a known
subtype, so padded rows carry `False` and contribute neither value nor gradient.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 1.0, mask: torch.Tensor | None = None,
            reduction: str = "batchmean") -> torch.Tensor:
    """KL from the teacher's softened distribution to the student's.

    `mask` (bool, broadcastable to the logits) restricts which rows count; the
    denominator is the number of selected rows, so a mask cannot rescale the loss
    by silently keeping its full size.
    """
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive, got %r" % (temperature,))
    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    per_row = F.kl_div(log_student, teacher, reduction="none").sum(-1)
    if mask is not None:
        selected = mask.to(per_row.dtype)
        total = float(selected.sum())
        if total == 0:
            return student_logits.sum() * 0.0
        return (per_row * selected).sum() / total * temperature ** 2
    if reduction == "batchmean":
        return per_row.mean() * temperature ** 2
    if reduction == "sum":
        return per_row.sum() * temperature ** 2
    if reduction == "none":
        return per_row * temperature ** 2
    raise ValueError("unknown reduction %r" % (reduction,))


def joint_kd(student_type, student_orbit, teacher_type, teacher_orbit, *,
             temperature: float = 2.0, alpha: float = 1.0,
             orbit_mask: torch.Tensor | None = None,
             type_mask: torch.Tensor | None = None) -> dict:
    """Type term plus `alpha` times the node-level orbit term."""
    type_term = kd_loss(student_type, teacher_type, temperature, mask=type_mask)
    orbit_term = kd_loss(student_orbit.reshape(-1, student_orbit.shape[-1]),
                         teacher_orbit.reshape(-1, teacher_orbit.shape[-1]),
                         temperature, mask=None if orbit_mask is None
                         else orbit_mask.reshape(-1))
    return {"type": type_term, "orbit": orbit_term,
            "total": type_term + float(alpha) * orbit_term}
