# Counterexamples for issue #48

Every model is retained. The question each row answers is what a single reading
would have got wrong. Ranks are 1 = best among the complete-case models.

| case | model | Recall@10 (rank) | ER (rank) | D2 | A_target | G_H1 |
|:--|:--|---:|---:|---:|---:|---:|
| best recall among the bottom half by ER | `TIGER_FORK_RQ-VAE/prefix_sib` | 0.061613 (7) | 3.5064 (35) | 0.2088 | 0.4512 | 0.1501 |
| lowest recall overall | `metric_cone_arms/A_euclid_metriccone_s42` | 0.032191 (38) | 1.0347 (38) | 0.0091 | 1.3573 | 0.0083 |
| largest target ambiguity A_target | `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | 0.053814 (27) | 3.4705 (36) | 0.1593 | 3.6217 | 0.2140 |
| highest raw ER | `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask` | 0.063563 (2) | 16.4346 (1) | 0.0425 | 0.0605 | 0.1588 |
| named: TIGER_RQ-VAE | `TIGER_RQ-VAE` | 0.053674 (28) | 6.4683 (29) | 0.0626 | 0.3660 | 0.1136 |

## Top-10 model screening

Candidate-model selection quality, not a recommender metric.

| selector | P@10 | hits |
|:--|---:|---:|
| rank by raw held-out L1 ER | 0.90 | 9/10 |
| LOFO M0 Ridge | 0.50 | 5/10 |
| LOFO M4 Ridge | 0.70 | 7/10 |
| LOFO M5 Ridge | 0.80 | 8/10 |
| LOFO M6 Ridge | 0.50 | 5/10 |
| LOFO M7 Ridge | 0.60 | 6/10 |
