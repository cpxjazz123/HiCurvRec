# Iteration bridge — Iter27 (ABORTED) → Iter28 (running)

## Status

**Iter28: Stage2 GPU training in progress** as of 2026-09-26 23:43 +1000.

- Mechanism: Sinkhorn-weighted Riemannian EMA codebook (SREMA) replacing AdamW on `layers.X.embedding.weight`.
- Codebook excluded from AdamW entirely; SREMA updates codebook on the Poincaré ball.
- Curvature unchanged from iter18 (cyclic learnable, `c_layer_scale` per-layer).
- β = 0.99 fixed.
- 4×A40 DDP, 100,000 global steps, ~22 min ETA.

## iter27 → iter28 carryover (already documented in iter27/logs/iteration_bridge.md)

- iter27 ABORTED at MVG (S08_MVG) because user-fixed `trust_radius_fraction = 0.5` produced `clip_frac = 0` across 200 MVG steps + 1000-step diagnostic probe.
- iter28 picks up the canonical Sinkhorn + hyperbolic geometry machinery but **changes the codebook update mechanism** (AdamW → SREMA), not just the AdamW hyperparameter.

## Forbidden in iter28 (carried forward)

- Do not silently inherit iter27's `trust_radius_fraction` — iter27 was a different mechanism.
- Do not modify Stage1 embeddings or Stage3 trainer.
- Do not stack a second new mechanism on top of the SREMA.
- Do not retune `β = 0.99` inside iter28 (SKILL.md §2.9).

## Post iter18 → iter28 (active contract)

Active research contract: **CAO-1 (Curvature-Aware Optimization)**.

The single new mechanism is **SREMA**: codebook `embedding.weight` is
**excluded** from the AdamW optimizer and is updated by a
Sinkhorn-weighted Riemannian EMA after every AdamW step.  Every
other component (encoder, decoder, `c_layer_scale`, Sinkhorn ε,
cyclic curvature, warm-start, Stage3 config) is inherited verbatim
from iter18.

## Active contract transition status

The user explicitly transitioned FCCR-1 → CAO-1 between iter26 and
iter27 (see iter27 directive).  Iter28 keeps CAO-1 unchanged; it does
not re-open the FCCR-1 question.

## iter28 hard target

`test_R@10 > 0.065` (downstream Stage3).

Iter28 vs canonical iter18 (`R@10 = 0.05988962203380978`) is the
direct comparison (single mechanism change, same protocol).