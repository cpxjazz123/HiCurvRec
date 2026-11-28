# Issue #288 — R01 DIAGNOSIS (offline, before the R01 run)

Evidence: `results/research/issue288/DIAGNOSIS_DATA.json` (sha256 in the RUN_LOG commit),
computed by `diagnose_r01.py` from the frozen #277 evaluator, the five frozen SID exports, the
visible training graph and the sectional-curvature triplet table.

## The observation (not a root cause)

Seven #287 rounds + the #274 references all fail the three-threshold conjunction (P > 0.080973,
N < 0.008752, C < 0.151137 on one export). The P/N/C trade-off is monotone along the mechanism
strength; the best per-axis values belong to three different arms.

## The measured decomposition (the #277 definitions, not a uniform-pair formula)

| export | P | N | C | in-bucket behavioural purity | unshared related pairs (false negatives) | L1 bucket size (mean / median / max) |
|---|---|---|---|---|---|---|
| TIGER | 0.092533 | 0.011457 | 0.169276 | **0.3450** | 0.9458 | — |
| R2 (#274) | 0.078361 | 0.008752 | 0.151137 | 0.3342 | 0.9634 | — |
| R3 (#274) | 0.080973 | 0.009017 | 0.154350 | 0.3331 | 0.9626 | — |
| #287 r1 | 0.083532 | 0.009392 | 0.150933 | 0.3534 | 0.9543 | — |
| #287 r7 | 0.085935 | 0.009970 | 0.153821 | 0.3542 | 0.9470 | — |

(the exact bucket-size columns are in `DIAGNOSIS_DATA.json`; the N column is the #277
anchor-conditioned probability `remaining/population`, never `sum_k n_k^2 / N^2`.)

Two measured facts separate the failure from its causes:

1. **The in-bucket behavioural purity is ~33–35 % in every export.** The pair supervision changed
   the bucket *sizes* (N) and the codeword geometry, but it barely changed *which* co-sharers are
   behaviourally related (purity range across five very different mechanisms: 0.333–0.354, +2 pp).
   P is the target-in-bucket rate; the purity is its ceiling. This is why every mechanism that
   lowered N (by shrinking buckets) also lowered P: it removed related and unrelated items alike.
2. **The curvature's within-stratum information is weaker than the degree sum.** On the triplet
   table's distance-1/2 pairs, with the label "the pair is an observed edge" and the strata
   (the direction class, the distance class, the centre's degree decile): the pooled within-stratum
   AUC of the curvature is **0.7291**, of the degree-sum control **0.8707** (2.84 M pairs). The
   curvature's independent selection information is therefore not the lever the #274/#287 indices
   assumed; the degree/co-occurrence structure carries more.

## Competing hypotheses and their status

| id | statement | status | evidence |
|---|---|---|---|
| H1 | The failure is the bucket-size distribution alone (N) | REFUTED | N falls in R2 (0.008752) while P falls (0.0784); the sizes alone do not explain P |
| H2 | The failure is the in-bucket purity: the pair term cannot make co-sharers behavioural | SUPPORTED | purity 0.333–0.354 across five mechanisms; P ≤ purity always |
| H3 | The curvature adds independent information beyond the structure | REFUTED | within-stratum AUC 0.7291 < the degree control 0.8707 |
| H4 | The final hard `q` (all-levels) supervision shapes the L2/L3 refinement better than the L1-only supervision | SUPPORTED | R2's C 0.151137 / N 0.008752 vs #287 r1's C 0.150933 (similar) but the L1-only form's C degrades in the same-anchor construction (round 2: 0.158376) |
| H5 | The same-anchor pair construction raises P relative to the #274 outer pairs | SUPPORTED (weak) | #287 r2 (same-anchor, L1): P 0.085064 vs #287 r1 (outer pairs, L1): 0.083532 |

## The R01 decision (one main change, evidence-driven)

R01 = **the same-anchor pairs ranked on the final hard `q`**. The single main change against the
#274 reference (R2, the best N/C arm) is the **pair construction** (the closed/open outer pairs →
one anchor vs its own positive and negative); the supervised representation stays the #274 final
hard `q` (the best measured N/C form). The #287 same-anchor arm (r2) used the L1 codeword and lost C
(0.158376); the #274 final-`q` arm had the best N/C but the outer-pair construction; the two were
never combined.

## The six pre-training questions

1. **Which specific problem?** The in-bucket purity ceiling (H2): P cannot rise without the
   co-sharers becoming behavioural, and the L1-only same-anchor form lost the L2/L3 shaping.
2. **What evidence?** The purity table (0.333–0.354) and the #287 r2 C (0.158376) vs R2's
   (0.151137) at nearly the same P band.
3. **Why will the new mechanism act on it?** The same-anchor contrast states "this anchor's code is
   closer to its behavioural partner than to its distance-two co-item"; ranking on the final `q`
   (all three levels) lets the gradient reach the L2/L3 codebooks as well, so the refinement carries
   the contrast instead of only the L1 prefix.
4. **What is substantively different from the failed mechanisms?** #274's outer pairs never shared
   an anchor (the positive and the negative pairs were disjoint); #287 r2 shared the anchor but
   supervised the L1 codeword only. R01 has both the shared anchor and the all-levels `q`.
5. **Which metrics decide it?** P/N/C of the checkpoint-bound export with the frozen evaluator; the
   in-bucket purity and the per-level distinctness as the mechanism diagnostics.
6. **If it fails, what is learned?** Whether the purity ceiling is attackable through the pair
   form at all, and whether the final-`q` form preserves the same-anchor P gain — the knowledge
   that decides R02's factor (pair set vs budget vs schedule).

## Low-cost mechanism checks before the launch

The tests (12 known-answer, all pass; 8 mutations, all caught by a named assertion) and the trainer's
own calibration record the gradient routing on the real batch: the pair term must reach all three
codebooks and not the encoder (`pair_grad_norm_per_codebook` nonzero, `pair_grad_norm_encoder` = 0),
the reconstruction/pair cosine and the L1 assignment change rate.

## What remains uncertain

Whether the purity is attackable by *any* pair-level term (H2 says the ceiling is structural). If
R01's purity stays ~0.35, R02 must change the factor (the pair set / the budget / the schedule), not
the pair form; if the purity rises but P does not, the L2/L3 shaping is the misplaced lever.
