# Counterexamples: where Hard Prefix ER stops predicting recommendation quality

Every model below is retained. Nothing is dropped for being inconvenient, and
each row is a case where a single Stage 2 reading would mislead model selection.
Ranks are 1 = best among the 38 models.

| case | model | Recall@10 (rank) | L1 ER (rank) | L1 Gap | L1 distinct | 3-token collision |
|:--|:--|---:|---:|---:|---:|---:|
| best_recall_among_bottom_half_ER | `TIGER_FORK_RQ-VAE/prefix_sib` | 0.061613 (7) | 3.5064 (35) | +0.09947 | 175 | 0.2272 |
| lowest_recall | `metric_cone_arms/A_euclid_metriccone_s42` | 0.032191 (38) | 1.0347 (38) | +0.00446 | 39 | 0.6354 |
| highest_gap_models | `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | 0.053814 (27) | 3.4705 (36) | +0.24796 | 67 | 0.8561 |
| highest_gap_models | `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw` | 0.052003 (31) | 4.4917 (34) | +0.18286 | 165 | 0.2493 |
| highest_gap_models | `TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1` | 0.055189 (25) | 10.8845 (12) | +0.08627 | 240 | 0.1039 |
| highest_gap_models | `TIGER_FORK_RQ-VAE/nce_euclid_group` | 0.059019 (18) | 6.8686 (28) | +0.08265 | 236 | 0.0508 |
| worst_collision | `TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare` | 0.046536 (35) | 3.1828 (37) | +0.07287 | 101 | 0.5985 |

## Reading these together

- **An ER filter would have discarded a strong model.** `TIGER_FORK_RQ-VAE/prefix_sib` reaches
  Recall@10 0.061613, rank **7 of 38**, while its
  L1 ER of 3.5064 ranks only **35 of 38**. Its L1 codebook uses
  175 codes and its three-token collision rate is
  0.2272 — it buys that recall with a crowded codebook, the same trade the
  R2/R3 rounds traced directly.

- **A Gap filter would have accepted a broken model.** `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` has the largest
  L1 Gap in the study (+0.24796) — it would clear the old train-pair
  threshold of 0.095 — yet its three-token collision rate is 0.8561 and its
  Recall@10 is only 0.053814 (rank 27 of 38). Gap is
  blind to how many unrelated items share the same triples.

- **ER is not sufficient either.** The highest-gap models above sit at L1 ER between 3.5 and
  10.9, and the best-recall model in the study
  (`TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample`, Recall@10 0.063633) ranks
  2 of 38 by ER — good, but not uniquely so. Ranking by ER is a reasonable
  heuristic, not a validated selector.

## Top-10 model screening

This measures the ability of a Stage 2 proxy to rank candidate models. It is not a
recommender metric.

| selector | P@10 | hits |
|:--|---:|---:|
| rank by raw L1 ER (no fitting) | 0.90 | 9/10 |
| LOFO M0 Ridge prediction | 0.50 | 5/10 |
| LOFO M1 Ridge prediction | 0.60 | 6/10 |
| LOFO M2 Ridge prediction | 0.10 | 1/10 |
| LOFO M3 Ridge prediction | 0.20 | 2/10 |

The raw-ER ranking beats every fitted model. Under Leave-One-Family-Out each family's
prediction comes from a model fitted without that family, so cross-family ordering can only be
as stable as the ER-to-Recall mapping between families — and the fitted models inherit that
instability while adding estimation variance on top. With five families this is a coarse test;
the gap should be read as 'these fitted models add nothing here', not as 'fitting is harmful
in general'.
