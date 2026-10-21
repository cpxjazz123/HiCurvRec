"""CGV instrumentation for this arm's mechanism.

Not part of the mathematics and not a judge: it exposes the tensors the mechanism reads, its
real assembly for the verifier to execute, and a call counter so the verifier can observe the
production loop executing it. The formula itself lives in
``mechanism_behavior_residual_geometry_transport.py`` and is not duplicated here.
"""

from __future__ import annotations

import torch

from mechanism_behavior_residual_geometry_transport import mechanism_loss

MECHANISM_ENABLED = True
MECHANISM_WEIGHT = 0.001
MECHANISM_LEVEL = 0
MECHANISM_PARAMETERS = {
    "lambda": 0.001,
    "beta": 0.1,
    "epsilon": 1e-12,
}
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
    return {
        "source": residuals[MECHANISM_LEVEL, :pairs],
        "target": residuals[MECHANISM_LEVEL, pairs : 2 * pairs],
        "codebook": model.rq.vq_layers[0].get_code_embs(),
        "mask": torch.ones(pairs, device=device, dtype=residuals.dtype),
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
