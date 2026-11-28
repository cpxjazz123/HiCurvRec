# Issue #287 — DIAGNOSIS

## The failure this round must attack (#274, plus the #277–#286 measurements)

Issue #274's curvature-selected pair ranking reached a hard-q ranking accuracy of ~80 % and lowered the
L1 negative sharing (N = 0.0087523 for R2 `9f6bf673…`, 0.0090166 for R3 `9d9f3d68…`), but on the same
checkpoint exports the **L1 positive sharing stayed below the frozen TIGER value** (P = 0.0783614 /
0.0809729 vs TIGER 0.09253295) and the **three-code collision stayed at ~15 %** (C = 0.1511368 /
0.1543499 vs TIGER's historical high-recall band 3.03–8.71 %). The behavioural supervision therefore
did not move the quantity that P, N and C actually measure.

## Candidate root causes (distinguishable before the run)

| id | explanation | discriminating measurement |
|---|---|---|
| **E1** | **L2/L3 absorption**: the ranking acted on `q = Σ_l c_l`, so the deeper codebooks satisfied the ordering while the exported L1 prefix stayed frozen. | the pair gradient's per-codebook norms (L1 vs L2/L3) and the L1 assignment change rate after the first step |
| **E2** | **Pair semantics**: the positive/negative labels are geometric (closed vs open outer pair), not the behavioural relation that P measures (an observed edge vs an unobserved pair). | the structural/behavioural profile of the selected pairs against their class (screen287) and the pair index's coverage |
| **E3** | **Scale / gradient budget**: the margin or the 2 % budget makes the term too small to change the code geometry. | the margin's scale on the L1 distance, the gradient ratio, the calibration's binding ratio and the measured code movement |
| **E4** | **L1 collapse**: pulling L1 codes together raises the L1 collision, so P and C move in opposite directions. | the per-epoch C and the L1 codebook usage (the export's collision) |
| **E5** | **Coverage**: the frozen index selects only 26,583 of 79,185 groups (10,782 centres), so the supervision touches a small part of the catalogue. | the index coverage and the number of items the sampled groups touch |

E1 is the mechanism this round tests (the single main change: the supervised distance becomes the
**actually selected L1 codeword**); E2 is the pre-registered round-2 change (same-anchor behavioural
pairs); E3/E4/E5 are measured in the same run and decide the next step if round 1 fails.

## Cheap evidence available before any training (no retraining)

- The #274 metrics and the R2/R3 checkpoint-export identities with their P/N/C (the frozen
  thresholds).
- The frozen #277 predictor and the #275/#276 evaluator: the metrics are computed on the 57,439
  validation rows with the legal uniform negative population, so P/N/C are reproducible for any SID.
- The #283–#286 analyses: the L1 prefix dominates all three metrics, and the L1 codebook geometry is
  the only channel through which a ranking term can move them.
- The frozen triplet/curvature asset and the frozen #274 index (no rebuild, no re-selection).

## Formal-training admission six questions

1. **Problem**: the curvature-selected ranking must raise P, lower N and lower C on one checkpoint
   export; #274 lowered N but left P below TIGER and C at ~15 %.
2. **Evidence**: the #274 supervised distance was the sum of the three hard codewords, so nothing
   forced the L1 codebook to move; the exported L1 prefix stayed at the TIGER-like level while the
   deeper levels absorbed the ranking (E1). The diagnosis is cheap: the per-codebook gradient norms
   and the L1 assignment change are measured before the formal run.
3. **Mechanism**: the ranking supervises the distance between the **actually selected L1 codewords**
   (a hard selection through the legal codebook path), so the gradient can only reach the L1
   codebook; the L1 geometry, and therefore the exported L1 prefix, must change.
4. **Difference from #274**: one main factor — the supervised distance (L1 instead of the three-level
   sum). The frozen index, margin, groups, budget, ramp and the whole Stage2 protocol are unchanged.
5. **Metrics**: the three frozen thresholds on the checkpoint export, measured with the frozen
   evaluator; the per-epoch trajectories (P/N/C proxies, collision, occupancy, Hard ER, gradients,
   epoch time) and the L1 probe.
6. **Knowledge if it fails**: which of E1–E5 is responsible — the L1 gradient share and the L1
   assignment change (E1), the pair semantics (E2), the scale (E3), the collapse (E4) or the
   coverage (E5) — which selects the next single legal change.

## What rounds 1 and 2 measured (the root cause of the N failure)

* Round 1 (the L1 supervision) reached the intended module: the pair gradient reached the **L1
  codebook only** (0.1586; L2/L3 and the encoder exactly 0), the L1 assignments moved (0.85 % after the
  first step) and the exported prefix improved on **P** (0.08353) and **C** (0.15093) — but **N rose**
  (0.00939). The inherited margin (0.2529) was calibrated on the **final-`q`** distance scale while the
  L1 codeword scale is 0.19–0.25, so the hinge was saturated (margin satisfaction 0.113) and the margin
  itself did nothing.
* Round 2 (the same-anchor behavioural pairs, the pre-registered candidate, with the margin
  re-derived on the L1 scale, 0.006438) raised **P** again (0.08506) but raised **N** (0.01014) and
  **C** (0.15838) with it.
* **Root cause (measured):** the ranking term reshapes the L1 *codebook geometry* but cannot refine the
  L1 *partition*. The module that decides which items share an L1 code is the **encoder**, and the level
  selection is an argmin, so the pair gradient stops at the codebook. Every push towards behavioural
  grouping therefore coarsens the partition (N and C rise together) instead of sharpening it. This is
  the issue's own hypothesis in #287's terms: the L2/L3 levels cannot continue to separate the items
  that the L1 prefix merged, and the mechanism never reaches the module that could separate them.
* **Round 3 (the pre-registered next single change: the gradient path).** The forward value stays the
  **hard selected L1 codeword** (the real SID semantics); the backward passes through the level's
  Sinkhorn soft assignment, so the encoder also receives the ranking gradient. Implementation detail
  found and fixed during the round: the production `sinkhorn` is `@torch.no_grad()`, so the first
  attempt had **zero** encoder gradient (measured: `pair_grad_norm_encoder = 0.0`) and was stopped; the
  loss module now carries a differentiable twin of the Sinkhorn whose agreement with the production is
  asserted by a test, and the formal run was restarted with the fix.

## The next-round candidate is pre-registered

If round 1 does not reach the three thresholds, the **only** pre-registered follow-up is the pair
construction: same-anchor behavioural pairs (an observed training-graph relation against an
unobserved one), curvature-screened inside comparable structure-role / edge-direction /
graph-distance / degree strata, with the same L1 supervision, margin and budget. No scale sweep, no
new loss and no threshold change is allowed.
