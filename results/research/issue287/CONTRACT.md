# Issue #287 — CONTRACT (frozen 2026-11-28, before the diagnostics and the formal run)

## 1. Frozen acceptance thresholds (issue §1)

The candidate is judged on the **SID exported from the final checkpoint** (never the training-time
export), evaluated with the frozen #277/#275 evaluator
(`tools.issue252.layer_metrics.validation_metrics` + `catalogue_metrics` on
`results/stage0_build_parquet/valid.parquet`, 57,439 rows, last history item vs true target, the
legal uniform negative population) and

* `P = l1_pos_hit` — L1 positive sharing rate,
* `N = l1_neg_hit` — L1 negative sharing probability,
* `C = 1 − unique(first 3 codes)/Nitems` — the three-code collision rate.

The three thresholds, from the frozen R2/R3 checkpoint exports (re-verified before the run):

| symbol | value | source |
|---|---|---|
| P floor | **> 0.08097285816257246** | R3 `9d9f3d68a8e97d57d7415318d7cf1e2563926044c05d15c99643488d808be0d7` (P=0.08097285816257246, N=0.00901661164455637, C=0.1543498596819457) |
| N ceiling | **< 0.008752316503966327** | R2 `9f6bf6731e88f5ec3e83bdf54b99784a5eafbf9412efe5e63f3b46b12cecaf97` (P=0.0783613920855168, N=0.008752316503966327, C=0.1511367795989751) |
| C ceiling | **< 0.1511367795989751** | R2 (same row) |

All three must hold **simultaneously on one and the same checkpoint export**; the best metrics of
different candidates must never be stitched. Stretch goals (not closure gates): `P ≥ 0.09253295`
(the frozen TIGER same-mouth value) and `C` inside the historical high-recall band 3.03–8.71 %.
`Hard ER = P/N` is reported without a threshold. `0.065` has no positive labels → NOT_EVALUABLE.

## 2. Mechanism boundary (issue §2)

* The cross-sectional curvature is used **only** by the offline index builder. It never enters the
  loss value, a weight, the margin, a distance or the code selection (asserted by the tests).
* The supervised loss is the distance ranking only:
  `L_geo = mean over groups of relu(d_pos − d_neg + m)` with the frozen margin
  `m = 0.2529585361480713` (identical for every sample).
* No new curvature regression, no global-distance loss, no collision loss, no other auxiliary loss.
* If the L1 distance is used it is the **actually selected L1 codeword** through the legal codebook
  path (an argmin selection), never a soft assignment.
* The frozen Stage2 protocol is unchanged: 768-dim input, 32-dim bottleneck, three levels of 256
  codes, Euclidean selection, the original residual quantisation, Sinkhorn, reconstruction,
  commitment, codebook loss and the SID export protocol.
* Training samples come from the training graph only; the validation behaviour is used for
  evaluation and for the pre-registered candidate screening only; the test split is never touched.

## 3. Round 1 — the single main change (frozen before the run)

**Change:** the ranking supervises the distance between the **L1 codeword that the model actually
selected**, instead of the sum of the three levels' hard codewords that Issue #274 used.

Everything else is Issue #274's frozen configuration: the frozen pair index
`results/research/issue274/pairs_index.npz` (the R2 rules: comparability group = centre plus the
unordered multiset of the two centre-edge directions; closed = `bc_dist_sq == 1`; open =
`bc_dist_sq == 4`; minimum support 2/2; exact ties excluded; the hold-out is `centre % 5 == 0`),
the frozen margin, `PAIR_GROUPS = 256` groups per optimizer step, the gradient budget
`PAIR_BUDGET = 0.02` (the weighted pair gradient stays below 2 % of the reconstruction gradient per
codebook) and `PAIR_RAMP_STEPS = 200`.

*Why this change:* Issue #274's ranking acted on the final hard `q = Σ_l c_l`, so the L2/L3 codebooks
could absorb the ordering while the exported L1 prefix barely moved (P below TIGER, N down,
collision ≈ 15 %). Ranking the L1 codeword forces the L1 codebook to carry the behavioural ordering
and therefore to move the L1 prefix.

*Falsifiable prediction:* the pair term's gradient must reach the **L1 codebook only** (the L2/L3
codebooks and the encoder must receive none), the L1 codebook's gradient share must dominate the
pair term, the L1 assignments must change during training, and the exported SID must satisfy
P ↑, N ↓, C ↓ against the three frozen thresholds. If the L1 codebook's gradient share is
negligible or the L1 assignments do not move, the hypothesis is refuted and the next round changes the
pair construction (same-anchor behavioural pairs, pre-registered below) instead of the scale.

*Pre-registered round 2 (only if round 1 fails and the diagnosis supports it):* the positive/negative
pair construction changes to **same-anchor behavioural pairs** (a positive item with an observed
training-graph relation to the anchor, a negative item without one), screened by curvature only
inside comparable structure-role / edge-direction / graph-distance / degree strata, with the same
L1 supervision, margin and budget. No other change is allowed in that round either.

## 4. Diagnostics before the formal run (issue §3)

On real batches: the curvature screening's discriminative signal (and its redundancy against the
closed/open label, the direct edge, the common neighbours and the degrees); the pair term's gradient
norms for the encoder and each of the three codebooks, the weighted/unweighted norms, the
reconstruction-to-pair cosine and the binding ratio; the actual L1 code changes after one optimizer
step; the margin scale on the L1 distance; the reconstruction gradient budget; the STE/codebook
routing; the positive/negative flips; the curvature-invariance of the loss; the index overlap and
conflicts; the NaN/collapse guards; the real hard forward.

## 5. Formal run and delivery

One formal candidate: 3000 epochs, seed 42, global batch 4096, the frozen configuration, no early
stopping. The SID is re-exported from the final checkpoint and evaluated with the frozen evaluator.
Deliverables: per-round DIAGNOSIS/HYPOTHESES/CONTRACT, the sample/asset manifest, the code and the
tests, the persistent commands/exit codes/logs, the real gradient and parameter-update evidence, the
checkpoint/SID/export identity, the per-round metrics and the final PASS table, the frozen #277
predictions (diagnostic only), the performance record, the VERDICT, the infra drift 0 and the
commit+push. No Stage3, no new test, no new real Recall, and no faked four-piece Stage3 archive.

---

# ADDENDUM A — round 6 (owner-authorized, frozen 2026-11-28 before the round-6 run)

The owner proposed the mechanism that rounds 1–5 could not reach, and it is recorded here before the
round-6 run:

> 假设商品 A 与 B 有购买关系，与 C、D、E 没有已知行为关系…… 拉近两个商品，可能改变它们与整个码字
> 分组中其他商品的关系。所以，我建议选择负样本时，优先选择 C、D、E 这种**已经发生错误共享**的商品，
> 而不是随机选择距离很远、本来就不会共享码字的商品。这样，Loss 就会同时要求：A 与 B 更接近；B 与
> C、D、E 保持区分。…… 我们希望通过截面曲率选择具有结构关联的正样本，再通过**困难负样本**保护商品
> 区分能力，从而解决正共享率和负共享率同步上升的问题。

Two measured facts make this impossible in the frozen round-2 form, and the addendum therefore
changes exactly two things:

1. **The negative construction.** The negative of an anchor is an item that **already shares the
   anchor's L1 code** and has no observed training-graph relation with it - a *measured false
   positive*, the population that the metric N counts. The negatives are mined offline from the
   frozen TIGER checkpoint export and refreshed from the **current** partition every
   `HARDNEG_REFRESH_EPOCHS = 500` epochs with the same frozen rule.
2. **The distance space.** The ranking is stated on the **item representation** (the encoder's
   bottleneck), not on the selected L1 codewords. This is forced by the mechanism itself: two items
   that share their L1 code have a codeword distance of exactly zero, so a codeword-distance hinge
   between them has **no gradient** (the codebook gradient is proportional to
   `c_anchor - c_negative = 0`) and "push the false positive away" cannot be written in that space.
   The representation is the space in which the assignment (the module that decides which items
   share an L1 code) is decided.

Unchanged from the frozen contract: the cross-sectional curvature is used **only** by the offline
index builder (never in the loss, a weight, the margin or a distance); the loss is the distance
ranking only, `mean over groups of relu(d_pos - d_neg + m)` with one frozen margin for every sample;
no curvature regression, no global-distance loss, no collision loss, no other auxiliary loss; the
frozen Stage2 protocol (768-dim input, the 32-dim bottleneck, three levels of 256 codes, the
Euclidean selection, the original residual quantisation, **the Sinkhorn balanced assignment**,
reconstruction, commitment, codebook loss, the SID export protocol); the training samples come from
the training graph only and the test split is never touched.

The margin rule keeps the frozen form with one documented adaptation: the margin is
`|median(d_negative - d_positive)|` on the frozen TIGER checkpoint (the median gap is **negative**
here, because the mined false positives are already closer than the behavioural positives - which
is the mechanism's premise), so the hinge is inactive for the median pair exactly as in the earlier
rounds. The frozen value: **0.04035949200819747** (`MARGIN_287_R6.json`).

**Falsifiable prediction (round 6):** the pair gradient must reach the **encoder** (and no codebook),
the L1 assignments must move during training, and the exported checkpoint's SID must satisfy
P > 0.08097285816257246, N < 0.008752316503966327 and C < 0.1511367795989751 **simultaneously**.
Any of these failing refutes the hypothesis and is recorded as such.
