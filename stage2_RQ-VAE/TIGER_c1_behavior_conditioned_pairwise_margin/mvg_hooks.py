"""CGV instrumentation for this arm's mechanism.

Not part of the mathematics and not a judge: it exposes the tensors the mechanism reads, its
real assembly for the verifier to execute, and a call counter so the verifier can observe the
production loop executing it. The formula itself lives in
``mechanism_behavior_conditioned_pairwise_margin.py`` and is not duplicated here.
"""

from __future__ import annotations

import torch

from mechanism_behavior_conditioned_pairwise_margin import mechanism_loss

MECHANISM_ENABLED = True
MECHANISM_WEIGHT = 0.05
MECHANISM_LEVEL = 0
MECHANISM_PARAMETERS = {'m_neg': 1.0, 'm_pos': 0.25, 'min_valid_pairs': 1, 'similarity_temperature': 1.0}
MECHANISM_CALLS = 0


def mvg_mechanism_term(*args, **kwargs):
    """The production call site: the mechanism's own function, with its frozen parameters."""
    global MECHANISM_CALLS
    MECHANISM_CALLS += 1
    merged = dict(MECHANISM_PARAMETERS)
    merged.update(kwargs or {})
    return mechanism_loss(*args, **merged)


def mvg_mechanism_inputs(observations: dict) -> dict:
    """The tensors the mechanism reads, taken from the verifier's observations."""
    model = observations["model"]
    residuals = observations["residuals"]
    pairs = int(observations["n_src"])
    device = residuals.device
    pair = residuals[MECHANISM_LEVEL, :pairs]
    tgt = residuals[MECHANISM_LEVEL, pairs : 2 * pairs]
    affinity = torch.nn.functional.normalize(pair, p=2, dim=-1) * torch.nn.functional.normalize(
        tgt, p=2, dim=-1)
    affinity = affinity.sum(dim=-1).clamp(0.0, 1.0).detach()
    return {
        "source": pair,
        "target": tgt,
        "codebook": model.rq.vq_layers[0].get_code_embs(),
        "mask": torch.ones(pairs, device=device, dtype=residuals.dtype),
        "affinity": affinity,
    }


def mvg_mechanism_executor(
    base_loss,
    l1_loss,
    other_loss,
    encoder,
    weight,
    valid_levels: int = 1,
    mechanism_inputs: dict | None = None,
) -> None:
    """The mechanism's real assembly, as the verifier executes it."""
    if mechanism_inputs is None:
        raise RuntimeError("the CGV executor needs the mechanism's inputs")
    inputs = dict(mechanism_inputs)
    source = inputs.pop("source")
    target = inputs.pop("target")
    codebook = inputs.pop("codebook")
    mask = inputs.pop("mask")
    term = mvg_mechanism_term(source, target, codebook, mask, **inputs)
    (MECHANISM_WEIGHT * term).backward()


__all__ = [
    "MECHANISM_ENABLED",
    "MECHANISM_WEIGHT",
    "MECHANISM_CALLS",
    "mvg_mechanism_term",
    "mvg_mechanism_inputs",
    "mvg_mechanism_executor",
]
