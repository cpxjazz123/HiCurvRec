# Issue #288 — R01 CONTRACT (frozen 2026-11-28 before the R01 launch)

## 1. The targeted root cause

The in-bucket behavioural purity ceiling (DIAGNOSIS.md, H2): the L1 buckets contain only ~33–35 %
behaviourally related pairs across five very different mechanisms, so P (the target-in-bucket rate)
is capped and every bucket-size reduction (N) removes related and unrelated items alike.  The #287
same-anchor arm lost C (0.158376) because its L1-only supervision left the L2/L3 refinement without
the contrast; the #274 final-`q` arm had the best N/C (0.008752 / 0.151137) but never shared an
anchor between its positive and negative pairs.

## 2. The causal path

* **The single main change (R01):** the pair construction.  The #274 outer pairs (a closed-triangle
  outer pair as the positive, an open-wedge outer pair as the negative) become the **same-anchor**
  pairs (`results/research/issue287/pairs_same_anchor.npz`, reused unchanged: the anchor, its
  curvature-screened direct partner, and its distance-two co-item as the relative-association
  negative).
* **The supervised representation (unchanged from the reference arm):** the **final hard `q`** =
  the sum of the three levels' hard selected codewords:
  `q(x) = c1(x) + c2(x) + c3(x)`.
* **The loss (unchanged form):** `L_geo = mean over the groups of relu(||q(a)-q(pos)|| - ||q(a)-q(neg)|| + m)`
  with the frozen margin `m = 0.00470653947787214` (`MARGIN_R01.json`: the median of
  `d_negative - d_positive` on the frozen TIGER final `q` over the same frozen pairs).
* **The gradient path:** the hard selection is an argmin per level, so the loss reaches the three
  **codebooks** and never the encoder.
* **The coupling with the curvature:** the curvature only screened the pairs offline inside the
  comparability group; it never enters the loss, the margin, a weight or a distance.

Unchanged frozen protocol: 768-dim item semantic input, the 32-dim bottleneck, three levels of 256
codes, the Euclidean selection, the original residual quantisation, Sinkhorn, reconstruction,
commitment, the codebook loss, the SID export protocol; seed 42; the original global batch 4096;
3000 epochs; the training samples from the training graph only; the validation only for evaluation.

## 3. The falsifiable prediction

On the checkpoint-bound export with the frozen #277 evaluator: the same-anchor + final-`q` arm must
keep the same-anchor P gain (#287 r2's P 0.085064 ± the ranking noise) **while regaining** R2's
L2/L3 shaping (C ≤ 0.154 and N ≤ 0.0092), and the in-bucket purity must rise above the 0.354
observed maximum.  Any of these failing refutes the H4/H5 combination and directs R02 to the next
factor (the pair set, the budget, or the schedule).

## 4. The metrics and the decision

P > 0.08097285816257246, N < 0.008752316503966327, C < 0.1511367795989751 (the frozen #277
definitions, the 57,439 validation rows), on one export; the in-bucket purity, the per-level
distinctness, the false-negative rate and the bucket sizes as the mechanism diagnostics; Hard ER
reported without a threshold; the #277 prediction only as a diagnostic.
