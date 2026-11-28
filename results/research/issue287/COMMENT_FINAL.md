## Issue #287 — final verdict: the acceptance criteria are **not met**; the issue stays open

Six complete 3000-epoch candidates were trained and each is judged on its **own** checkpoint export
with the frozen #277 evaluator. The frozen thresholds are P > 0.08097285816257246, N <
0.008752316503966327, C < 0.1511367795989751, all three on one and the same export.

| round | arm | the single main change | P | N | C | thresholds met |
|---|---|---|---|---|---|---|
| 1 | `TIGER_l1_pairrank_rq` | the supervised distance becomes the **selected L1 codeword** | 0.08353209491808701 | 0.009391703948852751 | **0.15093342010005284** | P, C |
| 2 | `TIGER_sameanchor_pairrank_rq` | **same-anchor behavioural pairs** + the L1-scale margin | 0.08506415501662633 | 0.0101443625362726 | 0.1583763777606052 | P |
| 3 | `TIGER_ste_pairrank_rq` | the **straight-through gradient path** (the encoder joins the loop) | 0.08088580929333727 | 0.009307922913861833 | 0.15601740757310778 | none |
| 4 | `TIGER_l1margin_pairrank_rq` | the **margin calibrated on the L1 scale** | 0.0862480196382249 | 0.010293984868672328 | 0.16415178752999549 | P |
| 5 | `TIGER_sameanchor_r5_pairrank_rq` | the **non-adjacent behavioural positive** | 0.08245268893957068 | 0.009581459175042285 | 0.1578883149631919 | P |
| 6 | `TIGER_hardneg_pairrank_rq` | the **hard-negative construction** in the item representation | **0.08819791430909313** | 0.010532514090149122 | 0.15886444055801852 | P |
| ref | frozen TIGER (no ranking term) | — | 0.09253294799700552 | 0.011456922353416539 | 0.16927644690283483 | P |
| ref | R2 (`pairrank_enc`) | — | 0.0783613920855168 | **0.008752316503966327** | 0.1511367795989751 | — |
| ref | R3 (`pairrank_cover`) | — | 0.08097285816257246 | 0.00901661164455637 | 0.1543498596819457 | — |

The six rounds trace a **monotone trade-off**, not a Pareto improvement over R2/R3: the family's best
P (round 6) comes with N and C above both ceilings; the only round that clears C (round 1) sits at
N 0.009392; the N ceiling (0.008752) was never reached (the closest is round 3's 0.009308).

### What was established (measured)

1. The L1 supervision reaches its intended module (round 1): the pair gradient reaches the L1
   codebook only (0.1586), the L1 assignments move (0.85 %), P and C improve on both reference arms.
2. The binding constraint is N, and it is structural: every mechanism that raises the behavioural
   co-sharing of the L1 code raises the random co-sharing with it (rounds 1, 2, 4, 5).
3. The partition is owned by the encoder: the STE path made the encoder gradient non-zero (0.0268)
   and the mid-training collision fell to 0.130–0.138, but the final-epoch export drifted to 0.156.
4. The curvature screen is largely a degree/common-neighbour **scale proxy** inside a structure class
   (Spearman 0.480–0.492) and is not claimed as an independent contribution.
5. **The owner's hard-negative construction works as a positive-pull mechanism but not as an N
   mechanism** (round 6): the premise is measured (only 17.6 % of the frozen pairs already had the
   behavioural partner closer, ranking accuracy 0.262 during training, margin 0.04035949200819747 =
   |median gap|), P rose to the family's best (0.088198), yet N rose with it (0.010533). The
   separation instruction is real - the false positives are pushed away in the representation - but
   the partition **re-equilibrates**: the Sinkhorn balanced assignment keeps the bucket sizes, so an
   item removed from the anchor's bucket is replaced by another.

### The next legal change (pre-registered, not run)

Because N is governed by the **assignment structure**, the next change is not another pair form but a
**bucket-size-aware separation**: the mined false positive is weighted by the size of the bucket it
would leave, so the term targets the imbalance instead of individual items. Its falsifiable
prediction: N < 0.008752 with P > 0.080973 and C < 0.151137 on one export, with the Sinkhorn
assignment rule untouched. It is outside this session's frozen contract and needs explicit
authorization for a seventh round.

### Delivery

- Evidence: `results/research/issue287/` (`REPORT.md`, `VERDICT.md`, `HYPOTHESES.json`, `PROGRESS.json`,
  `CONTRACT.md` with the owner's addendum, `EVAL_287.json`, the frozen margins and index manifests, the
  per-round records in `results/round*.json`, the frozen candidates).
- Code: `stage2_RQ-VAE/TIGER_hardneg_pairrank_rq/` (the round-6 candidate; the other five trees are
  archived locally under `stage2_RQ-VAE/_archive/` by the repo's exclude rule).
- Commits: `a5feb2ab`, `66bd8942` (pushed to `main`).
- Infra: FROZEN, drift 0, no Stage3 started, no new real Recall claimed.
