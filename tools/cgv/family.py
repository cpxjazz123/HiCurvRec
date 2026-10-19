"""Family-wide declaration defaults for the RQ-VAE arms.

A contract states everything about ONE mechanism. This module only spares the arms of one family
from repeating the parts that are genuinely identical across them: the adapter binding, the
parameter groups, the probe case, the optimizer, the aggregation rule and the runtime
observables.

It is a COMPOSITION HELPER for whoever builds the ``spec`` argument, and nothing more: the
engine never calls it, and CGV reads no declaration from a mechanism's tree. It deliberately
contains no expectations about any mechanism's mathematics -- the rule, the reference term and
the expected propagation are the caller's to declare. Nothing here is an oracle; oracles live in
``oracles.py``.

Changing a default here changes what a caller would typically send, not what the verifier
accepts, so it does not invalidate any certificate.
"""
from __future__ import annotations

from typing import Any

# The probe case: a deterministic mini batch the gates and oracles share. Declared rather than
# implicit, because the shape invariants in the forward contract refer to it.
PROBE = {"n_src": 24, "in_dim": 32}

CODEBOOK_LEVELS = 3
CODEBOOK_MAX_INDEX = 255


def model() -> dict[str, Any]:
    return {
        "adapter": "rqvae",
        "factory": "RQVAE",
        "tokenizer_config": "_tokenizer_config",
        "forward": "forward",
        "parameter_groups": {
            "encoder": "encoder",
            "decoder": "decoder",
            "codebooks": "rq",
        },
        "probe": dict(PROBE),
    }


def forward_contract(*, declared_effect: str = "none", invariants: list | None = None,
                     jacobian: dict | None = None) -> dict[str, Any]:
    """The forward declaration for this family.

    ``declared_effect`` is ``none`` for a mechanism that only rewrites gradients: its forward must
    then satisfy the plain RQ-VAE invariants. A mechanism that changes the forward declares
    ``changes_forward`` and adds the invariants that describe the change.
    """
    base_invariants = [
        {"name": "finite"},
        {"name": "shape", "observable": "reconstructed", "expected": [2 * PROBE["n_src"],
                                                                     PROBE["in_dim"]]},
        {"name": "rows", "observable": "tokens", "expected": 2 * PROBE["n_src"]},
        {"name": "tokens_in_range", "observable": "tokens", "min": 0,
         "max": CODEBOOK_MAX_INDEX},
        {"name": "levels", "expected": CODEBOOK_LEVELS},
        {"name": "residual_identity", "tol": 1e-4},
    ]
    contract = {
        "observables": ["reconstructed", "quant_loss", "base_loss", "recon_loss", "tokens"],
        "declared_effect": declared_effect,
        "invariants": base_invariants + list(invariants or []),
    }
    if jacobian:
        contract["jacobian"] = jacobian
    return contract


def update() -> dict[str, Any]:
    """The optimizer the real training loop uses, with an independent update oracle."""
    return {
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "atol": 1e-6,
        "requires_parameter_change": True,
        "oracle": "oracles:optimizer_update",
    }


def distributed() -> dict[str, Any]:
    """DDP's declared aggregation, checked against an independent pooled reference."""
    return {
        "world_size": 2,
        "aggregation": "ddp_mean",
        "atol": 1e-5,
        "oracle": "oracles:aggregate",
    }


def runtime(*, observe_extra: list | None = None, force_extra: dict | None = None,
            reset: str | None = None) -> dict[str, Any]:
    """The real entry point and what must be observable when it runs.

    ``force`` opens the declared activation so the mechanism is actually reached by a two-epoch
    probe: without it the behaviour ramp keeps the weight at zero and the probe would prove
    nothing. ``reset`` clears a declared module-global state so the probe starts from a known one.
    """
    force = {"BEHAVIOR_WEIGHT_MAX": 1.0, "RAMP_START_EPOCH": 0}
    force.update(force_extra or {})
    contract = {
        "entrypoint": "main",
        "isolation": "required",
        "force": force,
        "observe": [
            {"name": "mechanism_calls", "target": "r4_project_conflict", "min": 1},
            {"name": "optimizer_steps", "min": 1},
            {"name": "parameter_updates", "min": 1},
        ] + list(observe_extra or []),
    }
    if reset:
        contract["reset"] = reset
    return contract


def gradient(*, rule: str, reference_term: str, mode: str = "formula",
             loss_terms: dict | None = None, expected_paths: dict | None = None,
             extras: dict | None = None) -> dict[str, Any]:
    """The gradient declaration, with the family's assembly convention filled in.

    The assembly the whole family shares is
    ``g_base + probe_weight / valid_levels * (g_target + g_other)``; ``rule`` says how
    ``g_target`` is produced, and ``reference_term`` says what it is projected against. Both are
    the mechanism's own claim and are checked by G3 against the independent oracle.
    """
    contract: dict[str, Any] = {
        "target": "encoder",
        "mode": mode,
        "rule": rule,
        "reference_term": reference_term,
        "oracle": "oracles:assemble_gradient",
        "eps_flag": "R4_08_EPS",
        "probe_weight": 0.5,
        "probe_seeds": [1, 4, 5, 12, 16, 30, 32, 36],
        "atol": 1e-5,
        "effect": {"min_deviation": 1e-6, "min_probes": 1},
        "loss_terms": loss_terms or {
            "base_loss": "base_loss",
            "l1_loss": "behavior_l1",
            "other_loss": "behavior_other",
        },
        # Declared propagation, measured for this family: the reconstruction and quantisation
        # terms reach the encoder, and so does the LEVEL-0 behaviour term. The L2+L3 terms do
        # not, because the residual chain stops the gradient at the first quantiser, so the
        # family declares that instead of hiding it. A mechanism whose premise is that the L2+L3
        # gradient exists must declare `nonzero` for `other_loss` and will then be refused by G2
        # -- which is the point: the declaration is checked, not assumed.
        "expected_paths": expected_paths or {
            "base_loss": "nonzero",
            "l1_loss": "nonzero",
            "other_loss": "zero",
        },
    }
    contract.update(extras or {})
    return contract
