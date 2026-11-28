# Issue #287 — VERDICT

**Stage2-only.** No Stage3 was started, no new test was run, no new real Recall exists, and the
#277 predictor was neither replaced nor modified. The issue's closure condition is **one complete
3000-epoch candidate whose checkpoint export satisfies the three frozen thresholds simultaneously**:
P > 0.08097285816257246, N < 0.008752316503966327, C < 0.1511367795989751 (re-verified from the R2/R3
checkpoint exports with the frozen #277 evaluator, to the last digit).

## The verdict: the closure condition is **NOT met** by any of the seven rounds

Every number below is the frozen #277 evaluation of the round's **own** checkpoint export (never a
training-time export, never a stitched combination).

| round | arm | the single main change | P | N | C | thresholds met |
|---|---|---|---|---|---|---|
| 1 | `TIGER_l1_pairrank_rq` | the supervised distance becomes the **selected L1 codeword** | 0.08353209491808701 | 0.009391703948852751 | **0.15093342010005284** | P, C |
| 2 | `TIGER_sameanchor_pairrank_rq` | **same-anchor behavioural pairs** + the L1-scale margin | 0.08506415501662633 | 0.0101443625362726 | 0.1583763777606052 | P |
| 3 | `TIGER_ste_pairrank_rq` | the **straight-through gradient path** (the encoder joins the loop) | 0.08088580929333727 | **0.009307922913861833** | 0.15601740757310778 | none |
| 4 | `TIGER_l1margin_pairrank_rq` | the **margin calibrated on the L1 scale** | 0.0862480196382249 | 0.010293984868672328 | 0.16415178752999549 | P |
| 5 | `TIGER_sameanchor_r5_pairrank_rq` | the **non-adjacent behavioural positive** | 0.08245268893957068 | 0.009581459175042285 | 0.1578883149631919 | P |
| 6 | `TIGER_hardneg_pairrank_rq` | the **hard-negative construction** in the item representation | **0.08819791430909313** | 0.010532514090149122 | 0.15886444055801852 | P |
| 7 | `TIGER_bucketsize_pairrank_rq` | the **bucket-size-weighted separation** | 0.08593464370897821 | 0.009969683904863316 | 0.1538211249847481 | P |
| ref | frozen TIGER (no ranking term) | — | 0.09253294799700552 | 0.011456922353416539 | 0.16927644690283483 | P |
| ref | R2 (`pairrank_enc`) | — | 0.0783613920855168 | **0.008752316503966327** | 0.1511367795989751 | — |
| ref | R3 (`pairrank_cover`) | — | 0.08097285816257246 | 0.00901661164455637 | 0.1543498596819457 | — |

Read against the thresholds, the seven rounds trace a **frontier**, not a dominating point: the
family's best P (round 6, 0.088198) comes with N 0.010533 / C 0.158864; the family's best N (round 3,
0.009308) and best C (round 1, 0.150933) sit at P 0.080886 / 0.083532; round 7 dominates round 2 on
all three axes (P 0.085935 > 0.085064, N 0.009970 < 0.010144, C 0.153821 < 0.158376) and round 4 on
N and C, but the N ceiling (R2's 0.008752) is **13.9 % below** the family's best N and the C ceiling
(R2's 0.151137) is **1.8 % below** the family's best C - no round reaches R2's N, and only round 1
clears C.

## What each round established (measured, not asserted)

1. **The L1 supervision reaches its intended module** (round 1): the pair gradient reaches the **L1
   codebook only** (0.1586), never L2/L3 and never the encoder; the L1 assignments move (0.85 % after
   the first step); the export improves **P** (0.08353) and **C** (0.15093).
2. **The binding constraint is N, and it is structural** (rounds 2–7): every mechanism that raises the
   behavioural co-sharing of the L1 code raises the random co-sharing with it. The same-anchor
   behavioural pairs (round 2) confirmed it for the pair construction; the L1-scale margin (round 4)
   showed that a *selective* hinge (margin satisfaction 0.113 → 0.47) does not escape it either.
3. **The partition is owned by the encoder** (round 3): with the straight-through path the encoder
   gradient became non-zero for the first time (0.0268) and the mid-training collision fell to
   0.130–0.138, but the final-epoch export (the one the protocol keeps) drifted back to 0.156.
4. **The curvature screen is largely redundant** (C2): inside a structure class the curvature is a
   degree/common-neighbour **scale proxy** (Spearman 0.480–0.492 with the degree sum and 0.478–0.485
   with the common-neighbour count); it is not claimed as an independent contribution.
5. **The owner's hard-negative construction works as a positive-pull mechanism but not as an N
   mechanism** (round 6): the premise is measured (only 17.6 % of the frozen pairs already had the
   behavioural partner closer, ranking accuracy 0.262 during training) and P rose to the family's best
   (0.088198), yet N rose with it (0.010533). The separation instruction is real but the partition
   **re-equilibrates**: the Sinkhorn balanced assignment keeps the bucket sizes, so an item removed
   from the anchor's bucket is replaced by another.
6. **The bucket-size weighting moves the frontier but not across it** (round 7): weighting the
   separation by the size of the bucket it would leave did evening the buckets out - the largest
   bucket fell from **718** (frozen TIGER) to **460** at epoch 1000 and the mean from 270.2 to 222.6 -
   and the export improved on round 6 on both **N** (0.010533 → 0.009970, −5.3 %) and **C**
   (0.158864 → 0.153821, −3.2 %) at a 2.6 % P cost. The direction is confirmed; the magnitude is not
   enough: N is still 13.9 % above the ceiling and C 1.8 % above its ceiling.

## Hypotheses (see `HYPOTHESES.json` for the full register)

| id | status | reason |
|---|---|---|
| H56 (the mechanism acts on the L1 path) | SUPPORTED | the pair gradient reaches the L1 codebook (rounds 1/2/4/5) or the encoder (rounds 3/6/7); the L1 assignments move |
| H57 (all three thresholds on one export) | REFUTED | no round met all three (see the table) |
| H58 (#274's failure was the L2/L3 absorption) | SUPPORTED | the L1 supervision moved the failure to the negative sharing |
| H59 (the geometric labels are not the behavioural relation) | SUPPORTED | the behavioural pairs raise P and N together |
| H60 (the curvature adds information beyond the structure) | REFUTED (redundancy side) | within-class Spearman 0.48 with the degree sum / common neighbours |
| H61 (the chain-ALL signal is a proxy for the L1 coarseness) | SUPPORTED | the metrics move with the collision rate |
| H62 (a codebook-only ranking cannot lower N) | SUPPORTED | rounds 1/2/4/5: N rose with P |
| H63 (the same-anchor contrast lowers N) | REFUTED | round 2: N 0.010144 |
| H64 (the STE path lets the partition become finer) | REFUTED | round 3: the final export is coarser (C 0.156017) |
| H65 (the saturated margin made the term an unhinged contraction) | REFUTED | round 4: the selective hinge (0.0229) raised C to 0.164152 |
| H66 (the non-adjacent positive breaks the trade-off) | REFUTED | round 5: N improved 5.6 % against round 2 but P fell; 1 of 3 |
| H67 (the hard-negative construction lowers N) | REFUTED | round 6: P rose to 0.088198, N rose to 0.010533; 1 of 3 |
| H68 (the bucket-size weighting lowers N) | REFUTED | round 7: N fell 5.3 % and C 3.2 % against round 6 (the buckets evened out), but N 0.009970 > 0.008752 and C 0.153821 > 0.151137; 1 of 3 |

## The next change (needs a contract re-scope, not run)

The seven rounds measure the frontier of the **pair-level** mechanisms: the ranking can shape the
codebook geometry and the item representation, and the bucket-size weighting can *even the buckets
out*, but the **Sinkhorn balanced assignment keeps the residual imbalance**: the weights target where
the mass is, yet the mass re-equinbrates within the balanced assignment. The only lever left inside
the authorized area that can act on the bucket-size *distribution* directly is a **global usage-balance
term on the encoder's assignment** (an entropy/usage-deviation objective), which the frozen contract's
"no auxiliary loss" clause currently forbids. The measured numbers say what it must achieve: shrink
the largest buckets enough that `sum_k n_k^2 / N^2` falls below 0.008752 while the behavioural
co-sharing (P) stays above 0.080973 and the three-code collision (C) below 0.151137. Running it
requires the owner to re-scope the contract (a seventh-round extension is already used; this is an
eighth change and a different loss class).

## Closure checklist (skill §6.4)

1. Production code implemented: yes — seven candidate trees, each from the frozen baseline path with a
   documented single change; every per-file diff is in `REPORT.md`, and the finished trees are under
   `stage2_RQ-VAE/_archive/` (the round-7 tree is the active one).
2. Required tests executed: yes — the per-tree deterministic suites and the mutation tests (all caught
   by a named assertion); the machine gates (`entry`, `behavior_graph`) pass for every tree.
3. Required training completed: yes — seven complete 3000-epoch runs at seed 42 and global batch 4096,
   each judged on its own checkpoint export; the per-round records are in `results/round*.json`.
4. Infra boundary respected: yes — FROZEN; the protected-component snapshot (start vs close) shows
   added/removed/changed = 0; the assets were only read; the #277 predictor, the 124-row prospective
   file and the #284 candidate were not modified.
5. **The acceptance criteria are not met.** The issue therefore remains **OPEN**, with the
   reproducible evidence of all seven rounds, the measured frontier and the next change above. No
   Stage3 was started and no Recall is claimed.
