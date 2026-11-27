# Issue #271 diagnosis (before mechanism design)

## 1. Phenomenon (kept separate from causes)

#269 trained a frozen per-triple κ_t with the original A.1 origin-value objective on the hard q. Its
local proxy improved (`#270` common Σ ruler: A.1 MAE 2.246 → 1.997, Δ = −0.249), but the same export
worsened the global graph-distance distortion on #270's relative ruler (J 0.183 → 0.326), compressed the
L2 prefix space (170 → 113 groups, 127 splits, 93 merges) and lost recall (R@10 0.052595 → 0.045231,
Δ = −0.00736). #270's 42-arm observational analysis found no stable monotone curvature→R association
(Spearman −0.068; CI [−0.51,+0.35]), no conditional interaction with SID preservation, and a strong
family/protocol variance, so the loss cannot be attributed to the local curvature term alone.

The new question is narrower and interventional in intent: **if global behavioural-distance fidelity is
held at the frozen TIGER level, does local sectional-curvature supervision still reach the discrete SID
and improve recommendation?** It is not assumed to be true.

## 2. Competing explanations and the measurement that separates them

| id | explanation | measurement in this issue | status before the run |
|---|---|---|---|
| H1 | The local κ gain was bought by degrading the global distance field; the SID assignment then compressed prefixes and lost recall | full-graph J at the frozen TIGER checkpoint and at the candidate; `J_max = J_TIGER + ε`; per-stratum/tail errors; L2 split/merge audit | UNTESTED |
| H2 | The local A.1 proxy is simply not recommendation-relevant (ruler mismatch), so a constrained run will preserve J and SID but not improve R@10 | independent common-ruler A.1 on the candidate vs TIGER; Stage3 R@10 with the SID frozen before test | UNTESTED |
| H3 | The #269 loss is a protocol confound (single-card #269 vs four-card history; family variance dominates) | single-card flag and world size logged; comparison to canonical TIGER disclosed as not fully matched; no claim of a pure protocol effect | UNTESTED (cannot be isolated on this host) |
| H4 | The SID/quantisation layer, not geometry, determines recall; any geometry change is invisible after the hard SID | SID split/merge and prefix purity, collision, codebook usage; Stage3 with a frozen SID | UNTESTED |

A null result is an allowed outcome. The run must be able to distinguish H1 from H4 even when R@10 does
not move: if J ≤ J_max, the SID is structurally TIGER-like, and R@10 is still below TIGER, then H2/H4 gain
evidence; if J ≤ J_max but the SID still compresses, the global constraint does not reach the assignment.

## 3. Six admission questions (Stage2)

1. **Which problem?** #269's local curvature gain came with global graph-distance degradation and prefix
   compression; the next run must hold the global distance field fixed while keeping the local objective.
2. **Evidence it caused the failure?** #270: J 0.183 → 0.326, L2 170 → 113, ΔR −0.00736, while local A.1
   improved by 0.249; #270's mediation (curvature→SID ρ −0.28, SID→ΔR ρ −0.47) is a descriptive
   concomitant, not a proof. It is enough to justify an intervention, not to assert the cause.
3. **Why does the new mechanism act on it?** The dual term `μ (Ĵ − J_max)` adds a gradient that opposes
   any increase of the global distance error; μ rises only while the estimated constraint is violated,
   so at convergence the normal cone condition holds instead of a fixed-weight compromise.
4. **Substantive difference from #269?** #269 minimised only the local proxy; #271 adds an explicit
   inequality constraint over the global behavioural distance with a non-negative multiplier and a
   full-graph acceptance test. It also inherits the exact same local term, so the difference is the
   constraint, not a new loss shape.
5. **Which metric decides?** Full-graph `J ≤ J_max`; independent common-ruler local A.1 error < TIGER;
   three-code collision < 0.40; Stage3 R@10 vs `0.05520639286895663` and the `0.065` target; SID split/merge
   and prefix diversity reported alongside.
6. **Knowledge if it fails?** If J passes but R@10 does not improve, the local-curvature→recommendation
   channel is not established even under a controlled global field (H2/H4). If J fails, the dual is not
   effective at this budget; the learned μ trajectory and violation log localise whether the primal or
   the dual is at fault. Either outcome is informative and rules out a class of next mechanisms.

## 4. Low-cost probes before the official run

- Recompute `J_TIGER` and `J_#269` on the frozen checkpoints with the exact common ruler; freeze ε from
  the bootstrap standard error of the fixed pair estimator (no tuning after seeing the candidate).
- Probe the pair sample: stratum sizes vs `assets/distance_histogram.npz`, coverage and weights; verify the
  sample's full-J estimate on TIGER is within ε of the exact full J.
- On a real batch, check the local/global gradient norms and cosine, that both reach the encoder and the
  three codebooks, and that the κ/S*/graph tensors require no gradient.
- Verify the κ=0 limit of the common ruler against the Euclidean distance, and the mask behaviour for
  self-pairs, unreachable pairs and coincident points.

## 5. Still unknown before the run

- Whether the frozen pair sample is representative enough for the full-graph J at the 1e-3 level.
- Whether the dual step can move J without destroying the local A.1 fit at the fixed 3000-epoch budget.
- Whether the single-card run is comparable to canonical TIGER; it is disclosed, not corrected by
  retraining TIGER.
