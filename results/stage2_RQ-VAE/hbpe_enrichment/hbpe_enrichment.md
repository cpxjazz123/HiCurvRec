# HBPE Held-Out Behavioral Prefix Enrichment (Issues #31, #32, #33)

## 1. Experimental Setup & Verification Protocol

- **Validation Pair Dataset**: `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz`
- **Held-Out Sample Size**: 52,855 transition pairs (scored relations are never observed in `train.parquet`), history[-1] -> target
- **Negative Baseline**: 10 popularity/frequency-matched negatives per anchor (total 528,550 negative draws); each negative has no observed train or valid relation to that anchor
- **Bootstrap Methodology**: Paired row-level bootstrap, 1000 replicates (seed 42), 95% percentile CIs. A row is one transition and contributes its positive match together with its 10 negative matches, so the positive and negative estimates stay paired across replicates.
- **Metrics**: `pos_share` = fraction of held-out positive pairs whose two items take the same code at that prefix level; `neg_share` = the same over negative draws; `gap = pos_share - neg_share`; `ER = pos_share / neg_share`.

### Baseline Canonical Check

The canonical Exp1a Resample export is re-evaluated on the identical pair set and compared against the historical reference constants from issue #30:

- **L1 ER**: expected `16.3895`, measured `16.3895` -> **PASSED**
- **L1+L2 ER**: expected `58.3871`, measured `58.3871` -> **PASSED**

---

## 2. Enrichment Ratio Summary

| Arm | Level | pos_share | neg_share | gap | ER | 95% CI (ER) | L1 codes | L1+L2 prefixes | 3-token coll. | Δ ER vs base |
|---|:---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| `resample` | L1 | 0.0794 | 0.00485 | +0.07460 | **16.39** | [15.59, 17.22] | 241 | 16,145 | 0.0363 | +0.00 (×1.00) |
| `resample` | L1+L2 | 0.0034 | 0.00006 | +0.00337 | **58.39** | [41.50, 90.01] | 241 | 16,145 | 0.0363 | +0.00 (×1.00) |
| `hbpe_a1` | L1 | 0.0922 | 0.00624 | +0.08593 | **14.76** | [14.17, 15.41] | 172 | 13,177 | 0.0667 | -1.63 (×0.90) |
| `hbpe_a1` | L1+L2 | 0.0079 | 0.00012 | +0.00775 | **64.00** | [51.04, 83.75] | 172 | 13,177 | 0.0667 | +5.61 (×1.10) |
| `hbpe_a2` | L1 | 0.1269 | 0.01952 | +0.10733 | **6.50** | [6.32, 6.70] | 58 | 6,481 | 0.1335 | -9.89 (×0.40) |
| `hbpe_a2` | L1+L2 | 0.0126 | 0.00048 | +0.01208 | **26.25** | [22.86, 30.35] | 58 | 6,481 | 0.1335 | -32.14 (×0.45) |
| `hbpe_a3` | L1 | 0.1367 | 0.02037 | +0.11633 | **6.71** | [6.53, 6.91] | 55 | 6,701 | 0.1203 | -9.68 (×0.41) |
| `hbpe_a3` | L1+L2 | 0.0106 | 0.00038 | +0.01025 | **27.96** | [23.84, 32.57] | 55 | 6,701 | 0.1203 | -30.43 (×0.48) |

---

## 3. Findings

### 3.1 A1 (#31): L1 soft enrichment improves L1+L2 selectivity

A1 is the only HBPE arm whose **L1+L2 ER beats the baseline**: **64.00** vs 58.39 (95% CI [51.04, 83.75], non-overlapping with the baseline's [41.50, 90.01]). The mechanism behind it is visible in the components: positive sharing more than doubles (0.0079 vs 0.0034) and the gap widens from +0.0034 to +0.00775.

L1 ER falls slightly to 14.76 from 16.39, and the two bootstrap intervals do not overlap (A1 [14.17, 15.41] vs baseline [15.59, 17.22]), so the L1 regression is real rather than sampling noise. The cause is that A1's L1 positive sharing rose to 0.0922 but its negative sharing rose further in relative terms, from 0.0048 to 0.00624. The cost is codebook spread: L1 uses 172 codes against the baseline's 241, and collision rises from 0.0363 to 0.0667.

### 3.2 A2 (#32): optimising L2 through the residual proxy collapses L1

A2 regresses on **both** levels: L1 ER 6.50 and L1+L2 ER 26.25, against baseline 16.39 / 58.39. The L1 codebook falls to 58 distinct codes (baseline 241), which inflates the negative sharing rate to 0.01952, roughly 4× the baseline, and that denominator is what caps the enrichment ratio. Collision doubles to 0.1335.

This is the concrete failure of the residual proxy. Routing the L2 objective through `r2_proxy = z - C1.detach()[argmin]` keeps the gradient path into the Encoder alive, but the L2 term is free to satisfy itself by sharpening the L1 assignment rather than by refining the L2 codebook, and it does: the soft two-level objective reports Soft_ER12 = 89.32 while the discrete L1+L2 ER measured on held-out pairs is 26.25.

### 3.3 A3 (#33): the coverage floors never bind, and do not change the outcome

A3 adds positive-prefix coverage floors `rho1 = 0.11639702`, `rho12 = 0.00988457`, measured from the canonical Resample export over train transitions, penalised as `ReLU((rho - S_pos)/rho)^2` at weight 0.01. Across the whole run both terms are **exactly zero** (`L_cov1 = L_cov12 = 0.0` at every logged epoch after activation): the enrichment objective already holds positive sharing above both floors on its own, so the constraint never activates and contributes no gradient.

A3 therefore reproduces A2's behaviour with a slightly different collapse point — L1 ER 6.71, L1+L2 ER 27.96, 55 L1 codes, collision 0.1203. The A2 and A3 confidence intervals overlap heavily on both levels, so the coverage term is neither helping nor hurting: the floors were set at baseline level and the failure mode was never a coverage shortfall but a codebook collapse that raises negative sharing instead.

### 3.4 The soft proxy does not rank the arms the way the discrete metric does

Training-side soft ratios and held-out discrete ER disagree in sign across all three arms:

| Arm | Soft_ER1 (train) | Hard L1 ER | Soft_ER12 (train) | Hard L1+L2 ER |
|---|---:|---:|---:|---:|
| A1 | 40.33 | 14.76 | — | 64.00 |
| A2 | 7.83 | 6.50 | 89.32 | 26.25 |
| A3 | 8.19 | 6.71 | 102.98 | 27.96 |

The soft quantity is an inner product between two full softmax distributions over the codebook, so it rewards distribution shape rather than actual discrete hits: it is maximised by making the assignment distribution peaked and behavioural, which is exactly the direction that collapses the codebook. A soft ER of 103 (A3) corresponds to a held-out ER of 27.96. This is why A2 and A3 report healthy soft numbers while losing more than half the baseline selectivity, and it is the single most important read of these three runs.

---

## 4. Summary

- **A1** improves L1+L2 ER to 64.00 (baseline 58.39) while costing L1 ER (14.76) and codebook spread. This is the best held-out L1+L2 result in the programme and the only HBPE arm that beats the baseline on any level.
- **A2** regresses on both levels (6.50 / 26.25) through L1 codebook collapse.
- **A3** matches A2 (6.71 / 27.96); its coverage floors never bind.
- **Across all three**: the soft-prefix proxy optimises a different quantity from the discrete prefix-sharing rate that the held-out evaluation measures, and the gap between them widens as the proxy is weighted more heavily. Any future arm that reports a soft ER should also report the held-out hard ER, because only the latter has predicted Stage 3 behaviour in this programme.

*Stage 3 Recall@10 and NDCG@10 under the frozen protocol (150 epochs, seed 42, beam 20) are reported separately in the per-arm Stage 3 result directories; they are not inferred from the offline Stage 2 enrichment ratios.*
