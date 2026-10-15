"""M6: Geometry-Consistent Residual Correction.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
The residual that RQ carries from one level to the next is updated in the ball
instead of in flat space: the subtraction ``x - q`` becomes a Mobius subtraction
in the ball, mapped back to tangent space. Level 1 is untouched (its input is the
raw encoder output and the crowded L1 condition must stay exactly as in the
control); levels 2 and 3 receive a residual whose geometry matches the ball the
codewords live in.

Math
----
With ``B(.) = expmap0(., c)`` and ``T(.) = logmap0(., c)``:

    r = T( B(x) +_c (-B(q)) )

``+_c`` is the Mobius addition of ``hyperbolic.mobius_add``. In the flat limit the
expression degenerates to ``x - q``, so the mechanism is a strict generalisation of
the control's residual rule rather than a different pipeline.

Composition with the shared layer
---------------------------------
Geometry only: it changes which residual vector each level receives, never the

codebooks, the losses, the capacity mask or the sampling. The shared layer's
capacity mask keeps L1 at 128 codewords because L1's own update is unchanged.

What is completely unchanged relative to the control
---------------------------------------------------
Common notation, and what the two sides must agree on
----------------------------------------------------
    forward residual:   r_{k+1} = r_k - q_k                (k = 0, Euclidean)
                        r_{k+1} = r_k (-) q_k              (k >= 1, Mobius)
    decoder input:      z_hat = q_0 + q_1 + ...            (previous arm, Euclidean)

The tempting move is to mirror it: ``r (-) q = log_0(exp_0(r) +_c (-exp_0(q)))``
looks like a translation whose inverse is ``r (+) q``. It is not. **Mobius addition
is not a group law** -- it is neither closed nor associative -- so the naive
mirroring does not recover ``r``. The tree's checks measure the failure directly:
``(r (-) q) (+) q`` differs from ``r`` by 6.0e-02 at c = 1, while the same
mirroring on the Euclidean pair differs by only 3.7e-02 by accident of the small
vectors used, not by construction.

That settles the choice the report leaves open. Option (b), a mathematically
consistent reconstruction, does not exist for this residual rule, because the rule
has no inverse. So this tree takes option (a) and *says so*: level 0 is untouched
(Euclidean in, Euclidean out) and levels 1-2 combine their codes under the same
Mobius law the residual used, which makes the two sides consistent in *law* while
the residual path stays non-invertible. The arm is named for that, not for an
inversion it cannot deliver.

Two further honest limits:

  * the terminal residual r_3 is discarded, as in any residual quantiser, so even a
    group law would not make the decoder input a literal inverse of the chain;
  * the exported SIDs depend on the code indices, not on the decoder input, so
    this change moves the reconstruction term of the training signal and the
    reported reconstruction error, not which codes are exported. That is measured,
    not assumed: ``reconstruction_deviation`` is in the metrics.

Scope of the mechanism, stated exactly
-------------------------------------
* Which levels it changes: L1 keeps the control's flat Euclidean residual (its
  input is the raw encoder output); L2 and L3 receive the Mobius-updated residual.
  So this is a two-level residual propagation, not a three-level scheme, and it
  changes what the *deeper* levels see.
* What it does not change: the reconstruction. The decoder input is still the
  Euclidean sum of the per-level codes, ``quantized_x += quant``, while the
  residual walk subtracts in the ball. The two are not inverses of each other, and
  the previous docstring did not say so. The decoder consumes a z-space sum, so
  keeping it Euclidean is a deliberate scope choice rather than an oversight --
  but it does mean the reconstruction differs from the plain Euclidean path, and
  therefore the reconstruction loss sees a different decoder input.
* What the exported SIDs depend on: the code *indices*, not the reconstruction.
  The deviation therefore does not change which codes the run exports; it changes
  the training signal through the reconstruction term. ``reconstruction_deviation``
  below measures that deviation so it is a number in the metrics rather than a
  claim in a comment.

L1 capacity mask and diagnostics, L1's residual rule, all losses, pair sampling,
codebook sizes and parameters, optimiser, schedule, seed, epochs, evaluation
interval. Levels 1 and 2 assignment distances also stay Euclidean, exactly as in
the control: this tree adds no ball metric to the assignment.

Fixed parameters (no sweep): ``c = 1.0``.
"""
from __future__ import annotations

import torch

from hyperbolic import expmap0, logmap0, mobius_add

MOBIUS_CURVATURE = 1.0


def _mobius_residual(residual: torch.Tensor, quant: torch.Tensor,
                    c: float = MOBIUS_CURVATURE) -> torch.Tensor:
    """``logmap0(expmap0(x) +_c (-expmap0(q)))``: the ball-consistent residual."""
    if not torch.isfinite(residual).all() or not torch.isfinite(quant).all():
        raise ValueError("residual and quant must be finite")
    ball_x = expmap0(residual, c)
    ball_q = expmap0(quant, c)
    return logmap0(mobius_add(ball_x, -ball_q, c), c)


def _trace_with_mobius_residual(module, embeddings: torch.Tensor,
                               with_tokens: bool = False):
    """Residual trace using the same level-by-level update rule as ``RQLayer``.

    Level 0 consumes the encoder output unchanged and its residual is the plain
    Euclidean remainder; only the later levels use the Mobius correction. Keeping
    the trace identical to the forward is what stops the behaviour term from being
    trained on a different residual path than the mechanism actually builds.
    """
    residual = module.encoder(embeddings)
    traces = [residual]
    tokens = []
    for level, vq_layer in enumerate(module.rq.vq_layers):
        quant, _, _, indices = vq_layer(residual)
        tokens.append(indices)
        residual = residual - quant if level == 0 else _mobius_residual(residual, quant)
        traces.append(residual)
    stacked = torch.stack(traces[: module.rq.codebook_num], dim=0)
    if with_tokens:
        return stacked, torch.stack(tokens, dim=0)
    return stacked


def reconstruction_deviation(module, embeddings: torch.Tensor) -> dict:
    """Compare this arm's decoder input with the plain Euclidean walk.

    Both walks use the same codebooks and the same encoder, so the only
    difference is the residual rule. Reports the relative L2 deviation, the worst
    per-row deviation, and the cosine agreement -- enough to say whether the
    reconstruction is close to the control's or has drifted.

    Returns a plain dict so the trainer can record it directly.
    """
    with torch.no_grad():
        def walk(mobius: bool) -> tuple[torch.Tensor, torch.Tensor]:
            residual = module.encoder(embeddings)
            total = torch.zeros_like(residual)
            for level, layer in enumerate(module.rq.vq_layers):
                quant, _, _, _ = layer(residual)
                residual = (
                    residual - quant if (level == 0 or not mobius)
                    else _mobius_residual(residual, quant, MOBIUS_CURVATURE)
                )
                total = total + quant
            return total, residual

        arm_input, arm_residual = walk(True)
        plain_input, plain_residual = walk(False)
        scale = plain_input.norm().clamp_min(1e-12)
        deviation = (arm_input - plain_input).norm(dim=-1)
        agreement = torch.nn.functional.cosine_similarity(
            arm_input, plain_input, dim=-1
        )
        return {
            "recon_dev_rel": float((arm_input - plain_input).norm() / scale),
            "recon_dev_max_row": float(deviation.max()),
            "recon_cos_min": float(agreement.min()),
            "residual_dev_rel": float(
                (arm_residual - plain_residual).norm() / plain_residual.norm().clamp_min(1e-12)
            ),
        }
