# Issue #287 — REPORT

Stage2-only mechanism work under `INFRA_MODE=FROZEN`. No Stage3, no new test, no new real Recall.
One A100-80GB was visible, so both formal runs used the single-card branch with the frozen global
batch 4096 (§19d0b); the four-card reference of ~0.4 s/epoch is quoted for comparison and the measured
single-card epoch time is reported separately.

## C1 — the frozen thresholds, re-verified with the #277 evaluator

The frozen evaluator (`tools.issue252.layer_metrics`, sha256 in `EVAL_287.json`) on the 57,439
validation rows reproduces the issue's R2/R3 values to the last digit:

| SID | P | N | C |
|---|---|---|---|
| R2 `9f6bf673…` (pairrank_enc checkpoint export) | 0.0783613920855168 | 0.008752316503966327 | 0.1511367795989751 |
| R3 `9d9f3d68…` (pairrank_cover checkpoint export) | 0.08097285816257246 | 0.00901661164455637 | 0.1543498596819457 |

so the acceptance thresholds are exactly **P > 0.08097285816257246**, **N < 0.008752316503966327**,
**C < 0.1511367795989751**, all three on one and the same checkpoint export.

## C2 — what the curvature screen contributes (`SCREEN_287.json`)

- The frozen #274 index: 26,583 groups over 10,782 centres, 21,221 training groups, 5,362 hold-out
  groups, 11,978 catalogue items touched (48.7 %).
- The selected positives are edges by construction (edge share 1.0) with a median curvature of 0.689;
  the selected negatives are non-edges (edge share 0.0) with a median curvature of 3.795 — the
  cross-structure incomparability #274 measured.
- Inside a structure class the curvature is largely a **scale proxy**: Spearman(κ, degree sum) = 0.480
  (closed) / 0.492 (open), Spearman(κ, common neighbours) = 0.485 / 0.478, Spearman(κ, reciprocal
  flag) = 0.253. The curvature-selected pairs differ from their class mainly in scale (the selected
  positives have a degree sum of 306 against the class-wide 767; the selected negatives carry 46.9
  common neighbours against the positives' 6.7). **The curvature screen therefore adds little beyond
  the closed/open label, the direct edge, the common neighbours and the degrees** (H60 → the
  redundancy side); it is reported as such and is not claimed as an independent contribution.

## Round 1 — the L1 supervision (`TIGER_l1_pairrank_rq`)

**The single main change against #274:** the ranking supervises the distance between the **actually
selected L1 codewords** instead of the sum of the three levels' hard codewords. The frozen #274 index,
the margin, `PAIR_GROUPS = 256`, the 2 % gradient budget, the 200-step ramp, the protocol, seed 42 and
the global batch 4096 are unchanged.

### The measured path (`pair_calibration`, `l1_probe`)

| measurement | value |
|---|---|
| pair gradient norm per codebook (L1, L2, L3) | **0.1586, 0.0, 0.0** |
| pair gradient norm, encoder | **0.0** |
| reconstruction gradient norm per codebook | 9.72e−6, 9.92e−6, 9.20e−5 |
| reconstruction ↔ pair gradient cosine | 0.0295 (near-orthogonal) |
| binding ratio → λ | 5.88e−5 → **1.175e−6** (= 2 % budget) |
| L1 assignment change rate after the first step | **0.85 %** of 4,096 probed items |
| L1 codebook step norm after the first step | 2.44e−4 |

The supervision reached the intended module and the L1 assignments moved (H56 SUPPORTED, H58
SUPPORTED: the #274 final-`q` ranking indeed let the deeper codebooks absorb the ordering).

### The formal run

3000 epochs, 913 s wall time, 0.293 s/epoch mean (single card), final training collision 0.15101,
L1 ranking accuracy 0.6152, margin satisfaction 0.1126. The checkpoint-bound export
(`item_sids.json`, sha `eac50f49…`, `tokens_match=true`, `differing_items=0`; the trainer's table is
kept beside it and differed on 64 items) evaluated with the frozen evaluator:

| metric | round 1 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|
| P | **0.08353209491808701** | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **PASS** |
| N | **0.009391703948852751** | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | **FAIL** |
| C | **0.15093342010005284** | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | **PASS** |

Diagnostics: Hard ER 8.8942, L2 prefixes 7,924, L3 refinement 2.0761 bits; the frozen #277 predictor
scores this SID at **0.0535488** (diagnostic only, not a real Recall).

**Round-1 verdict: P ↑ and C ↓ pass, N fails.** The L1 supervision did what it was designed to do
(P rose 0.0056 above R2 and 0.0026 above R3; C fell below both), but the L1 partition also became
**coarser overall**, so unrelated items now share an anchor's L1 code more often (N rose 0.00064 above
R2 and 0.00037 above R3). Two measured reasons:
1. the geometry is a *pairwise* ranking of two outer pairs, not a per-anchor contrast against an
   unrelated item — nothing in it opposes the anchor's sharing with a random item (E2);
2. the inherited margin (0.2529, calibrated on the **final-`q`** distance scale) is far larger than the
   L1 codeword scale (the frozen TIGER's L1 distances are 0.19–0.25), so the hinge was saturated
   (margin satisfaction 0.113) and the margin itself contributed nothing (E3).

## Round 2 — the same-anchor behavioural pairs (pre-registered)

**The single main change against round 1:** the pair construction and the pair form. The frozen index
is now *same-anchor* (`build_pairs287.py`, manifest `pairs_same_anchor.manifest.json`): for every
anchor item the **positive** candidate is a direct neighbour (an observed training-graph relation)
and the **negative** candidate is a distance-two co-neighbour (no edge — an unobserved pair, recorded
as a potential false negative, not as proof of irrelevance). The curvature screens inside the
comparability group (the anchor, the unordered multiset of the two centre-edge direction flags and
the anchor's degree decile): the highest-curvature positive and the lowest-curvature negative, with
#274's tie rule. The margin is re-derived for this pair on the frozen TIGER L1 codebook, once, before
training: **m = 0.00643804669380188** (`MARGIN_287.json`; the frozen TIGER's L1 distances are
0.228/0.248 on average, the median gap 0.0064, 53.0 % of the pairs already have the positive
closer). Everything else is round 1's frozen configuration (the L1 supervision, the 2 % budget, 256
groups, the 200-step ramp, seed 42, global batch 4096, 3000 epochs).

Index coverage: 40,835 groups with both candidates (of 73,116 with a negative), 15,998 anchors,
32,666 training groups, 8,169 hold-out groups, 16,616 items touched (67.6 % — more coverage than #274's
48.7 %).

Tests: 7 deterministic known-answer tests (the hand-computed hinge mean, the independent numpy
reference through the squared-norm expansion and the layer-cake identity, the hard L1 forward, the
gradient routing, the curvature invariance, the frozen margin, the deterministic sampling) and **8
mutations, all caught by a named assertion** (`MUTATIONS.json`).

### The formal run

3000 epochs on the single card; the measured path: the pair gradient reaches the **L1 codebook only**
(0.0878; L2/L3 = 0; the encoder 0), the reconstruction↔pair cosine 0.0443, λ = 2.026e−6, the L1
assignment change rate 0.85 % after the first step. The checkpoint-bound export evaluated with the
frozen evaluator:

| metric | round 2 | round 1 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|---|
| P | **0.08506415501662633** | 0.08353209491808701 | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **PASS** |
| N | **0.0101443625362726** | 0.009391703948852751 | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | **FAIL** |
| C | **0.1583763777606052** | 0.15093342010005284 | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | **FAIL** |

**Round-2 verdict: P rises further, but N and C rise with it.** The same-anchor contrast does exactly
what it says (the anchor's L1 code is pulled towards its direct neighbour), and that raises P; but the
L1 partition becomes coarser again (C 0.1509 → 0.1584, N 0.00939 → 0.01014), because the term cannot
make the L1 assignment *finer*: the module that decides which items share an L1 code is the **encoder**,
and the pair term does not reach it (the level selection is an argmin, so the gradient stops at the
codebook). Rounds 1 and 2 therefore share one root cause: **the ranking can reshape the L1 codebook
geometry but cannot refine the L1 partition**, so every push towards behavioural grouping coarsens the
partition instead of sharpening it.

## C5 — what is still open

The curvature selection's contribution beyond the structural label and the degrees is weak (C2), so
the honest reading of this round is: **the L1 supervision moves the exported L1 prefix (P ↑, C ↓) and
the remaining obstacle is the negative sharing (N)**, which the pre-registered same-anchor contrast
addresses directly. The real recommendation effect is a future Stage3 matter and is **not** claimed
here.

## Round 3 — the straight-through gradient path (`TIGER_ste_pairrank_rq`)

**The single main change against round 2:** the gradient path. The forward value of the supervised L1
codeword stays the **hard selected codeword** (the real SID semantics); the backward passes through
the level's Sinkhorn soft assignment, so the **encoder** — the module that decides which items share
an L1 code — also receives the ranking gradient. The index, the margin, the budget, the ramp, the
groups, the protocol, seed 42 and the global batch 4096 are round 2's frozen values.

Implementation note (a mechanism bug found and fixed inside this round, not a silent retry): the
production `sinkhorn` is decorated `@torch.no_grad()`, so the first attempt produced
`pair_grad_norm_encoder = 0.0` — the STE path was dead. That run was stopped; the loss module now
carries a differentiable twin of the Sinkhorn whose agreement with the production implementation is
asserted by a test (`test_the_differentiable_sinkhorn_agrees_with_the_production_one`), and the
formal run was restarted with the fix. Both runs are on record.

Measured path after the fix: the pair gradient reaches the L1 codebook (0.0879), **the encoder
(0.0268)** and neither L2 nor L3 (0.0); the reconstruction ↔ pair cosine is 0.0458; λ = 2.0266e−6
(the 2 % budget); the L1 assignment change rate after the first step is 0.85 %.

### The formal run

3000 epochs; the measured path: the pair gradient reaches the L1 codebook (0.0879) **and the encoder
(0.0268)**, never L2/L3 (0.0); the reconstruction ↔ pair cosine 0.0458; λ = 2.0266e−6; the L1
assignment change rate after the first step 0.85 %. The checkpoint-bound export evaluated with the
frozen evaluator:

| metric | round 3 | round 1 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|---|
| P | **0.08088580929333727** | 0.08353209491808701 | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **FAIL** |
| N | **0.009307922913861833** | 0.009391703948852751 | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | **FAIL** |
| C | **0.15601740757310778** | 0.15093342010005284 | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | **FAIL** |

**Round-3 verdict: 0 of 3.** The straight-through path did what it was designed to do (the encoder
gradient is non-zero for the first time) and the training-time collision fell to 0.130–0.138 in the
middle of the run, but the **final-epoch** state drifted back to a coarser partition (the exported
C = 0.156): the extra freedom in the encoder lets the pair term reshape the partition, yet the late
optimisation phase (the annealed learning rate) gives the reconstruction the last word, and the
export is the final epoch by protocol. The negative sharing stayed above the threshold as well
(0.00931 vs 0.008752), so the mechanism still cannot lower N below R2's level.

## Round 4 — the margin on the L1 scale (`TIGER_l1margin_pairrank_rq`)

**The single main change against round 1 (the best round):** the **margin**. Round 1 inherited Issue
#274's margin (0.2529), which was calibrated on the **final-`q`** distance scale, while the L1
codeword scale is 0.19–0.25: the hinge was therefore saturated (margin satisfaction 0.113) and the
term behaved like an **unhinged contraction** of the whole L1 codebook - every positive pair was
dragged together whether it satisfied the ranking or not, which is exactly the mechanism that
coarsens the partition (N up). Round 4 keeps round 1's index, its L1 supervision, its hard gradient
path, its budget, its ramp and the whole frozen recipe, and only re-derives the margin on the L1
scale with the same frozen rule (the median of `d_negative - d_positive` over the frozen TIGER's L1
codebook and the frozen #274 index): **m = 0.022925138473510742** (`MARGIN_274_L1.json`; the frozen
TIGER's L1 distances are 0.2186 / 0.2491 on average, the median gap 0.0229, 56.0 % of the pairs
already have the positive closer). With this margin the hinge is selective: only the violated pairs
are pulled, so the ranking sharpens instead of contracting the codebook.

Tests: the same 20 deterministic tests as round 1 (with the frozen margin updated) pass; the mutation
report is unchanged in structure.

### The formal run

3000 epochs, 864.509 s wall time, 0.2882 s/epoch mean (single card), final training collision 0.16431447512913333. The checkpoint-bound export (`item_sids.json`, sha `9829661e8301a193…`) evaluated with the frozen evaluator:

| metric | round 4 | round 1 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|---|
| P | **0.0862480196382249** | 0.08353209491808701 | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **PASS** |
| N | **0.010293984868672328** | 0.009391703948852751 | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | **FAIL** |
| C | **0.16415178752999549** | 0.15093342010005284 | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | **FAIL** |

Diagnostics: Hard ER 8.3785, L2 prefixes 7923, L3 refinement 2.0807 bits.

**Round verdict: **1 of 3 thresholds pass.****


## Round 5 — the non-adjacent behavioural positive (`TIGER_sameanchor_r5_pairrank_rq`)

**The single main change against round 2:** the frozen pair index (and the margin it implies). The
measurement that motivates it is the round 1–4 trade-off: on the (P, N, C) plane the term's strength
moves P, N and C **together** (the frozen TIGER baseline sits at P 0.09253 / N 0.01146 / C 0.16928;
R2 at 0.07836 / 0.008752 / 0.15114; round 1 at 0.08353 / 0.009392 / 0.15093), so the issue's three
thresholds — P above R3, N below R2, C below R2 — ask for a point **off** that curve, not for another
point on it. The measured reason for the curve is that round 2's positive is an **adjacent** item:
pulling the anchor towards its direct neighbour drags that neighbour's whole direct group into one
L1 bucket, so the anchor's L1 group grows with unrelated items (N up).

Round 5 therefore keeps the same-anchor contrast but changes **what the positive is**: the positive
candidate is an item that **shares a link with the anchor but is not directly connected to it** (the
open-wedge outer pair of Issue #274, used here as the *positive* instead of the *negative*), and the
negative candidate is an item at the same distance through **a different centre** (a relative
association that is not the positive's context). The index is built offline from the same frozen
assets with the same comparability group and the same curvature screen (`build_pairs287_r5.py`,
manifest `pairs_same_anchor_r5.manifest.json`, sha256 in the manifest): 70,354 groups, 24,299 anchors,
56,302 training groups, 14,052 hold-out groups, 24,413 items touched (99.3 % of the catalogue).
The margin is re-derived with the same frozen rule on the frozen TIGER L1 codebook, once, before
training: **m = 0.002109222114086151** (`MARGIN_287_R5.json`; the median gap 0.0021, 51.0 % of the
pairs already have the positive closer). Everything else is round 2's frozen configuration (the L1
supervision, the hard gradient path, the 2 % budget, 256 groups, the 200-step ramp, seed 42, global
batch 4096, 3000 epochs).

Tests: the tree's own suite (11 tests: the 7 known-answer tests of the same-anchor loss plus the
repository-root test plus the mutation harness) passes, and **8 mutations, all caught by a named
assertion** (`MUTATIONS.json`, schema `genrec.issue287.round5.mutations/1`).

### The formal run

3000 epochs, 868.302 s wall time, 0.2894 s/epoch mean (single card), final training collision 0.15772562736405416. The checkpoint-bound export (`item_sids.json`, sha `b10e4774ff490c83…`) evaluated with the frozen evaluator:

| metric | round 5 | round 1 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|---|
| P | **0.08245268893957068** | 0.08353209491808701 | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **PASS** |
| N | **0.009581459175042285** | 0.009391703948852751 | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | **FAIL** |
| C | **0.1578883149631919** | 0.15093342010005284 | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | **FAIL** |

Diagnostics: Hard ER 8.6054, L2 prefixes 7894, L3 refinement 2.0812 bits.

**Round verdict: **1 of 3 thresholds pass.****


## Round 6 — the hard-negative construction in the item representation (`TIGER_hardneg_pairrank_rq`)

**Owner-authorized** (the owner's proposal is quoted in `CONTRACT.md` addendum A). The mechanism:
the **positive** is round 2's frozen, curvature-screened behavioural partner; the **negative** is a
**measured false positive** - an item that *already shares the anchor's L1 code* and has no observed
training-graph relation with it, which is exactly the population the metric N counts. The owner also
named the failure mode of the earlier rounds: "拉近两个商品，可能改变它们与整个码字分组中其他商品的
关系" - pulling two items together changes their relation to the rest of the code bucket, so the loss
must also require the false positives to stay apart.

Two measured facts forced the second change (the **distance space**), and they are the reason the
codeword form could never express the owner's request:

* two items that share their L1 code have a **codeword distance of exactly zero**, so a
  codeword-distance hinge between them has **no gradient** (the codebook gradient is proportional to
  `c_anchor - c_negative = 0`) - "push the false positive away" cannot be written on the codewords;
* the module that decides *which* items share an L1 code is the **encoder**, and the codeword form
  never reaches it (rounds 1, 2, 4, 5: the encoder's pair gradient was exactly 0).

Round 6 therefore states the same-anchor ranking on the **item representation** (the encoder's
bottleneck): the loss pulls the anchor towards its behavioural partner and pushes it away from the
measured false positive, and the encoder - the module that owns the partition - receives the
gradient. The negatives are mined offline from the frozen TIGER checkpoint export and **refreshed
from the current partition every 500 epochs** with the same frozen rule, so they stay the items that
wrongly share the code *now*.

Index (`build_pairs287_r6.py`, manifest `pairs_hardneg.manifest.json`): 40,835 groups, 15,998
anchors, 32,666 training groups, 8,169 hold-out groups, 22,822 items touched; the mined
false-positive pool is **267.6 items on average** (median 231) - in the frozen TIGER each anchor's L1
bucket really does hold hundreds of unrelated items. The margin keeps the frozen rule with one
documented adaptation (`CONTRACT.md` addendum A): `m = |median(d_negative - d_positive)|` on the
frozen TIGER checkpoint's representation = **0.04035949200819747** (`MARGIN_287_R6.json`). The median
gap is **negative** (-0.0404): the false positives are *already closer* to the anchor than the
behavioural partner (only 17.6 % of the pairs already have the positive closer) - that is the
mechanism's premise, measured before the run.

Tests: 18 deterministic known-answer tests (the hand-computed hinge mean, the inverted order, the
satisfied hinge, the frozen margin, the gradient's direction on both sides, the term's isolation
from the codebooks, the index sampling, the rows/items mapping, the miner's exclusion rule, the
miner's determinism, the diagnostics and the index's validation) plus **8 mutations, all caught by a
named assertion** (`MUTATIONS.json`).

### The measured path and the formal run

3000 epochs, 717.704 s wall time, 0.2392 s/epoch mean (single card), final training collision 0.15890511245780292. The checkpoint-bound export (`item_sids.json`, sha `f1ee046ef8ca6090…`) evaluated with the frozen evaluator:

| metric | round 6 | round 1 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|---|
| P | **0.08819791430909313** | 0.08353209491808701 | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **PASS** |
| N | **0.010532514090149122** | 0.009391703948852751 | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | **FAIL** |
| C | **0.15886444055801852** | 0.15093342010005284 | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | **FAIL** |

Diagnostics: Hard ER 8.3739, L2 prefixes 7721, L3 refinement 2.1200 bits.

**Round verdict: **1 of 3 thresholds pass.****


## C3 — the round-by-round PASS table (one candidate, one checkpoint export)

| round | arm | the single main change | P | N | C | P ↑ | N ↓ | C ↓ | all three |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `TIGER_l1_pairrank_rq` | the supervised distance becomes the **selected L1 codeword** | 0.08353209491808701 | 0.009391703948852751 | 0.15093342010005284 | PASS | FAIL | PASS | no |
| 2 | `TIGER_sameanchor_pairrank_rq` | **same-anchor behavioural pairs** + the L1-scale margin | 0.08506415501662633 | 0.0101443625362726 | 0.1583763777606052 | PASS | FAIL | FAIL | no |
| 3 | `TIGER_ste_pairrank_rq` | the **straight-through gradient path** | 0.08088580929333727 | 0.009307922913861833 | 0.15601740757310778 | FAIL | FAIL | FAIL | no |
| 4 | `TIGER_l1margin_pairrank_rq` | round 1 + the **margin on the L1 scale** | 0.0862480196382249 | 0.010293984868672328 | 0.16415178752999549 | PASS | FAIL | FAIL | no |
| 5 | `TIGER_sameanchor_r5_pairrank_rq` | the **non-adjacent behavioural positive** | 0.08245268893957068 | 0.009581459175042285 | 0.1578883149631919 | PASS | FAIL | FAIL | no |
| 6 | `TIGER_hardneg_pairrank_rq` | the **hard-negative construction** in the item representation | 0.08819791430909313 | 0.010532514090149122 | 0.15886444055801852 | PASS | FAIL | FAIL | no |

Reference rows (frozen, re-verified in this round): R2 `9f6bf673…` = 0.0783613920855168 /
0.008752316503966327 / 0.1511367795989751; R3 `9d9f3d68…` = 0.08097285816257246 / 0.00901661164455637 /
0.1543498596819457. The frozen TIGER baseline, re-evaluated in this round with the same evaluator
(`results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json`): P **0.09253294799700552**, N
**0.011456922353416539**, C **0.16927644690283483** — it holds the highest P and the stretch P goal
(0.09253295) but also the highest N and the highest C, so the ranking term moves N and C down and P
down together; the issue's three thresholds sit **between** these reference rows, not on a line
between them. (The historical high-recall band 3.03–8.71 % belongs to other arms, not to the frozen
TIGER baseline, and remains a stretch goal, **not** a closure gate.)

## C4 — what is delivered, and what is deliberately not

- **Delivered:** one complete 3000-epoch candidate per round on the same frozen recipe, each judged on
  its **own** checkpoint export with the frozen #277 evaluator; the per-round evidence (the
  calibration and probe records, the trajectory, the export manifests, the checks, the tests and the
  mutation report), the frozen same-anchor index and its manifest, the frozen margin and its
  calibration record, the curvature-redundancy diagnostic, and the frozen #277 predictions for the
  candidates (diagnostic only, never a Recall).
- **Not delivered and not claimed:** any Stage3 run, any new real Recall, any replacement of the
  #277 predictor, and any claim that the curvature screen is an independent signal (it is largely a
  degree/common-neighbour scale proxy: the within-class Spearman(κ, degree sum) is 0.480–0.492).
- **Prospective clause of §4:** it concerns the SID → Recall prediction, which is a Stage3 matter and
  is forbidden here; the frozen #277 prediction of the best candidate is recorded as a diagnostic
  only. No unknown label was read in this round.

## C5 — effect sizes and the power statement

The three acceptance metrics are **population** quantities of the exported SID (not sample
statistics), so their differences carry no sampling error; the effect sizes against R2 are:

* round 1: ΔP = +0.00517 (+6.6 % relative), ΔN = +0.00064 (+7.3 %), ΔC = −0.00020 (−0.14 %)
* round 2: ΔP = +0.00670 (+8.6 %), ΔN = +0.00139 (+15.9 %), ΔC = +0.00724 (+4.8 %)
* round 3: ΔP = +0.00252 (+3.2 %), ΔN = +0.00056 (+6.4 %), ΔC = +0.00488 (+3.2 %)
* round 4: ΔP = +0.00789 (+10.1 %), ΔN = +0.00154 (+17.6 %), ΔC = +0.01302 (+8.6 %)
* round 5: ΔP = +0.00409 (+5.2 %), ΔN = +0.00083 (+9.5 %), ΔC = +0.00675 (+4.5 %)
* round 6: ΔP = +0.00984 (+12.6 %), ΔN = +0.00178 (+20.3 %), ΔC = +0.00773 (+5.1 %)

The **independent sample** that is still missing is not a triplet count but the number of
**independent Stage3 labels**: whether a finer L1 partition with a higher positive sharing translates
into a better Recall can only be measured on new Stage3 runs, which this issue forbids. The #277
predictor's own power analysis (#284) puts the SID-level prediction increment at ≈374 independent SIDs;
for the Recall consequence of these metric changes the required number is a future, separately
authorized measurement. No significance is claimed from the historical cohort.

## Closure

The issue's closure condition is **at least one complete 3000-epoch candidate that satisfies the three
frozen thresholds simultaneously on one real checkpoint export**, with the reproducible delivery.
Six complete candidates were run; **none met all three** (the best per axis: P 0.088198 in round 6,
C 0.150933 in round 1, and N never below 0.009308 against the 0.008752 ceiling). The issue therefore
**stays OPEN**; the measured root cause, the pre-registered next legal change and the full per-round
evidence are in `VERDICT.md`; every C is recorded with its evidence path and sha256 in `COMMENT_002.md`.
