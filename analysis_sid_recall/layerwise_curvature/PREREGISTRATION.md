# Pre-registration for the variable-curvature arm (Issue #129, WP4)

Written before any Stage 2 or Stage 3 run; its constants are fixed and must not be
tuned against a test recall.

Mapping from the measured profile to the model curvature, per level l:

```
c_l = clip(c0 * exp(-alpha * z_l), c_min, c_max)
c0 = 1.0, alpha = 0.5, c_min = 0.05, c_max = 4.0
```

* `z_l` is the size-preserving-null-standardised curvature of the **frozen pilot
  SID** (the current Euclidean baseline export), measured on the TRAIN-only
  graph, refreshed on a detached schedule (every 20 Stage 2 epochs) and cached;
  it never reads the arm's own test outcome.
* radius/scale: the Poincare ball radius is 1; the curvature enters the distance
  as `d_c(x, y) = (1/sqrt(c)) * d_1(sqrt(c) x, sqrt(c) y)`; `c_l` is per level.
* the geometry participates in the encoder/codebook/quantizer training through
  differentiable Poincare/Lorentz operations, not as a post-hoc score;
* ablation switches live in the candidate tree (behaviour-off, geometry-off,
  both-on) and, per repo rule §19g, no closed-switch arm is trained.

Matched arms if an arm were justified: (E) c_l = 0; (S) one shared c; (G) a
pre-registered one-factor sweep of c; (V) the mapping above. Same Stage 2 and
Stage 3 budgets, same protocol, byte-exact SID attestation.

## Why no arm is launched

The diagnostics of this run do not produce a level-dependent signal that survives
the required checks: see `REPORT.md` (Q1-Q3) and `../CONVERGENCE_FINDINGS.md`.
Under the issue's own rule ("only if the preceding diagnostics support an
interpretable level-dependent signal"), spending four matched Stage 2 and Stage 3
runs would be a claim-producing experiment without a pre-registered reason, so
the verdict is negative and the mapping above is recorded, not executed.

Current diagnostics feeding this decision: joint out-of-series RMSE change
-165.3%.
