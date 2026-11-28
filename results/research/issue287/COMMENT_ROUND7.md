## Issue #287 — round 7 (the bucket-size-weighted separation): the frontier moves, the acceptance point does not

Authorized by the owner's "继续"; this runs the change pre-registered in the round-6 verdict exactly
as written (`CONTRACT.md` addendum B, hypothesis H68).

**The change:** round 6's mechanism (the curvature-screened positive, the mined false positive, the
item representation, the frozen margin) with **one change in the reduction**:
`L = sum_g w_g * relu(d_pos - d_neg + m) / sum_g w_g`, where `w_g` is the size of the anchor's L1
bucket. The weights and the negatives are re-mined from the current partition every 500 epochs; the
Sinkhorn balanced assignment is untouched.

**The result** (the round's own checkpoint export, the frozen #277 evaluator):

| metric | round 7 | round 6 | R2 (the N/C ceiling) | threshold | verdict |
|---|---|---|---|---|---|
| P | 0.08593464370897821 | 0.08819791430909313 | 0.0783613920855168 | > 0.08097285816257246 | **PASS** |
| N | **0.009969683904863316** | 0.010532514090149122 | 0.008752316503966327 | < 0.008752316503966327 | FAIL |
| C | **0.1538211249847481** | 0.15886444055801852 | 0.1511367795989751 | < 0.1511367795989751 | FAIL |

**The direction is confirmed, the magnitude is not.** The weighting did even the buckets out - the
largest L1 bucket fell from **718** (frozen TIGER) to **460** at epoch 1000 and the mean from 270.2 to
222.6 - and the export improves on round 6 on **both** N (−5.3 %) and C (−3.2 %) at a 2.6 % P cost.
Round 7 now dominates round 2 on all three axes and round 4 on N and C. But N is still **13.9 %**
above its ceiling and C **1.8 %** above its own; only round 1 (0.150933) ever cleared C, and no round
ever reached R2's N.

**The measured frontier after seven rounds** (P, N, C): the family's best P (round 6: 0.088198), best
C (round 1: 0.150933), best N (round 3: 0.009308) are three different arms - the acceptance point
(P > 0.080973, N < 0.008752, C < 0.151137) lies outside the frontier, and the score is **0 of 7
candidates meeting all three**. The scorecard: rounds 1 (P,C), 2 (P), 3 (none), 4 (P), 5 (P), 6 (P),
7 (P).

**The root cause the seven rounds measure:** N is the probability that a random item shares the
anchor's L1 code; it is governed by the bucket-size distribution (`N = sum_k n_k^2 / N^2`), and every
**pair-level** term - whatever its space, margin, gradient path or weighting - re-equilibrates it
under the Sinkhorn balanced assignment. The weighting proved the mechanism can *move* the
distribution (the buckets evened out); the residual imbalance is what keeps N above the ceiling.

**The next change is a contract matter, not another pair form:** a global usage-balance
(entropy/usage-deviation) term on the encoder's assignment, which the frozen contract's "no auxiliary
loss" clause forbids. Running it needs an explicit re-scope (an eighth change and a different loss
class). Its target is measured: shrink the largest buckets enough that `sum_k n_k^2 / N^2` falls
below 0.008752 with P > 0.080973 and C < 0.151137 on one export.

**Delivery:** the round-7 candidate tree `stage2_RQ-VAE/TIGER_bucketsize_pairrank_rq/` (23
deterministic tests, 9 mutations all caught by a named assertion), the round-6 tree archived, all
records under `results/research/issue287/` (REPORT, VERDICT with the seven-round frontier, H68,
`MARGIN_287_R7.json`, `pairs_bucketsize.manifest.json`, `results/round7.json`), commit `5489a4db`
pushed to `main`, infra drift 0, no Stage3 started, no Recall claimed. **The issue stays OPEN.**
