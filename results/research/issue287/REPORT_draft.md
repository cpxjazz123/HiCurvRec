# Issue #287 — REPORT (draft; the final numbers are filled in after the formal run)

Analysis-only infrastructure, **Stage2-only mechanism run**. INFRA_MODE=FROZEN. No Stage3, no new
test, no new real Recall. One A100-80GB was visible, so the run used the single-card branch with the
frozen global batch 4096 (§19d0b); the four-card reference of ~0.4 s/epoch is quoted for comparison
and the measured single-card epoch time is reported separately.

## C1 — the frozen thresholds, re-verified with the #277 evaluator

The frozen evaluator (`tools/issue252.layer_metrics`, sha256 in `EVAL_287.json`) on the 57,439
validation rows reproduces the issue's R2/R3 values to the last digit:

| SID | P | N | C |
|---|---|---|---|
| R2 `9f6bf673…` (pairrank_enc checkpoint export) | 0.0783613920855168 | 0.008752316503966327 | 0.1511367795989751 |
| R3 `9d9f3d68…` (pairrank_cover checkpoint export) | 0.08097285816257246 | 0.00901661164455637 | 0.1543498596819457 |

so the acceptance thresholds are exactly **P > 0.08097285816257246**, **N < 0.008752316503966327**,
**C < 0.1511367795989751** on one and the same checkpoint export.

## C2 — what the curvature screen contributes (`SCREEN_287.json`)

- The frozen #274 index: 26,583 groups over 10,782 centres, 21,221 training groups, 5,362 hold-out
  groups, 11,978 catalogue items touched (48.7 %).
- The selected positives are edges by construction (edge share 1.0) with a median curvature of
  0.689; the selected negatives are non-edges (edge share 0.0) with a median curvature of 3.795 —
  the cross-structure incomparability that #274 measured.
- Within a structure class the curvature is largely a **scale proxy**: Spearman(κ, degree sum)
  = 0.480 (closed) / 0.492 (open) and Spearman(κ, common neighbours) = 0.485 / 0.478, while
  Spearman(κ, reciprocal flag) = 0.253. The curvature-selected pairs are therefore strongly
  distinguished from their class by the degree/common-neighbour scale (the selected positives have a
  degree sum of 306 against the class-wide 767 and 6.7 common neighbours against the negatives' 46.9),
  i.e. much of the screen is a structural-scale effect rather than an independent curvature signal.

## C3 — the round-1 mechanism and its measured path

The single main change: the ranking supervises the distance between the **actually selected L1
codewords** (the #274 index, margin, groups, budget and ramp are unchanged).

Measured on the real batches (the run's `pair_calibration` and `l1_probe` records):

| measurement | value |
|---|---|
| pair gradient norm per codebook (L1, L2, L3) | **0.1586, 0.0, 0.0** |
| pair gradient norm, encoder | **0.0** |
| reconstruction gradient norm per codebook | 9.72e−6, 9.92e−6, 9.20e−5 |
| reconstruction gradient norm, encoder | 2.93e−4 |
| reconstruction ↔ pair gradient cosine | **0.0295** (near-orthogonal) |
| binding ratio (reconstruction ÷ pair) | 5.88e−5 → λ = 0.02 × 5.88e−5 = **1.175e−6** |
| L1 assignment change rate after the first step | **0.85 %** of 4,096 probed items |
| L1 codebook step norm after the first step | 2.44e−4 |

So the mechanism acts on the intended path (the L1 codebook only, the encoder untouched) and the L1
assignments do move — the hypothesis that the #274 final-`q` ranking let the deeper codebooks absorb
the ordering is confirmed as a mechanism fact and is now bypassed.
