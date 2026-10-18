# Do any Stage 2 predictors add out-of-family information beyond Hard Prefix ER?

Issue #47. Frozen read-only analysis of the 38 models in
`model_inventory.json`. No Stage 2 or Stage 3 training was run, and no SID,
checkpoint, validation file or recommendation score was modified.

## 1. Headline answer

**No.** Under Leave-One-Family-Out, `log(L1 Hard ER)` on its own is the best of the four
pre-registered feature sets on both targets. Adding the L1 Gap (M1), the structural
health terms (M2), or both (M3) makes out-of-family prediction **worse**, on Recall@10
and on NDCG@10 alike. Not one of the three increments clears the confirmatory bar:
positive ΔMAE with a family-cluster 95% CI lower bound above zero.

The hypotheses are therefore reported as **not confirmed**. That is not the same as
Gap being useless — see §7 on what the design cannot settle.

## 2. Frozen inputs and audit

- Models: **38** in the inventory, **38**
  with features computed. Zero models were excluded.
- Stage 3 protocol held constant across all of them: {"seed": 42, "beam_size": 20, "num_epochs": 150, "no_eval": true, "skip_test": false, "test_file": "test.parquet", "screen_baseline_log": "", "early_stop": null, "screen_skip_test_on_fail": null, "sid_length": 4}.
- Evaluation size: every model has `n_eval = test.parquet`
  with 57,439 scored users and exactly **1** compliant run each
  (`stage3_repeat_noise`: {"models_with_repeat_runs": 0, "max_recall_spread": null, "per_model_spreads": {}}).
- Feature pairs: **52,855** held-out valid transitions x
  **10** matched negatives, seed 42, identical rows and
  identical item ids for every model.
- SID provenance is recorded per model in `manifest.json`. Files no longer present in the
  working tree were recovered **read-only from this repository's git history** and every
  one was checked against the inventory SHA-256 before use; no substitute SID or
  checkpoint was accepted.
- Family sizes actually used for the outer folds: {"TIGER_BEHAVIOR_SAMPLE_RQ-VAE": 4, "TIGER_FORK_RQ-VAE": 9, "TIGER_HYPERBOLIC_RQ-VAE": 7, "curvature_RQ-VAE": 17, "TIGER_RQ-VAE": 1}.

### Canonical Resample recheck

| quantity | expected | measured | abs error |
|:--|---:|---:|---:|
| valid L1 ER | 16.38953942232631 | 16.38953942232631 | 0.0 |
| valid L1+L2 ER | 58.38709677419355 | 58.38709677419355 | 0.0 |

Reproduced exactly, so the valid-side definition used throughout matches the frozen reference.

### Leakage controls

- all features computed only on behavior_pairs_valid_s42.npz
- no feature or threshold derived from test.parquet
- model selection uses Stage 3 inventory scores, never Stage 2 features
- train/test overlap is not re-derived here; the valid pairs carry a dropped_also_a_train_transition filter recorded in their meta file

## 3. Recomputed Stage 2 features (all 38 models, valid side)

Full table with counts, intervals and provenance in `stage2_valid_features.csv`.
Selected rows, sorted by Stage 3 Recall@10:

| model | family | L1 ER | L1 CI95 | L1 gap | L1 codes | eff usage | collision | L1+L2 ER | L1+L2 neg hits | Recall@10 |
|:--|:--|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | TIGER_BEHAVIOR_SAMPL | 16.3895 | [15.60, 17.25] | +0.07460 | 241 | 226.9 | 0.0363 | 58.39 | 31 | 0.063633 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask` | TIGER_BEHAVIOR_SAMPL | 16.4346 | [15.65, 17.34] | +0.07633 | 242 | 225.9 | 0.0375 | 68.57 | 35 | 0.063563 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43` | TIGER_BEHAVIOR_SAMPL | 15.6171 | [14.88, 16.44] | +0.07439 | 237 | 220.8 | 0.0391 | 71.67 | 30 | 0.063023 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44` | TIGER_BEHAVIOR_SAMPL | 15.7873 | [15.01, 16.57] | +0.07392 | 235 | 220.9 | 0.0390 | 65.59 | 34 | 0.062379 |
| `TIGER_FORK_RQ-VAE/fork_radial_corr` | TIGER_FORK_RQ-VAE | 12.0189 | [11.48, 12.59] | +0.08178 | 251 | 176.0 | 0.0324 | 38.86 | 70 | 0.061944 |
| `TIGER_FORK_RQ-VAE/fork_more_sib` | TIGER_FORK_RQ-VAE | 11.8184 | [11.28, 12.39] | +0.08206 | 248 | 176.5 | 0.0323 | 36.30 | 73 | 0.061874 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | TIGER_FORK_RQ-VAE | 3.5064 | [3.41, 3.60] | +0.09947 | 175 | 50.8 | 0.2272 | 13.66 | 1512 | 0.061613 |
| `TIGER_FORK_RQ-VAE/fork_euclid_group` | TIGER_FORK_RQ-VAE | 12.2128 | [11.66, 12.84] | +0.07516 | 252 | 181.0 | 0.0328 | 40.46 | 65 | 0.061561 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched` | TIGER_HYPERBOLIC_RQ- | 15.3014 | [14.61, 16.08] | +0.07630 | 240 | 208.4 | 0.0714 | 38.40 | 125 | 0.061317 |
| `TIGER_FORK_RQ-VAE/fork_ang_only` | TIGER_FORK_RQ-VAE | 12.1738 | [11.62, 12.74] | +0.07420 | 252 | 181.6 | 0.0360 | 40.16 | 64 | 0.060847 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta` | TIGER_HYPERBOLIC_RQ- | 12.6130 | [12.06, 13.23] | +0.07341 | 205 | 181.5 | 0.0664 | 27.14 | 119 | 0.060760 |
| `smoothness_arms/A_euclid_smooth_s42` | curvature_RQ-VAE | 10.2189 | [9.62, 10.86] | +0.04223 | 256 | 255.1 | 0.0000 | undefined | 0 | 0.060499 |

The **L1+L2 negative-hit column is the reason that level is unusable here.** The canonical
Resample has 31 negative hits out of 528,550 draws; a few collapsed models have so few that
the ER is undefined. 38 models have a defined L1 ER;
38 are used for the primary regression. The
L1+L2 complete-case subset is handled separately in §6 and never compared against a
different-sized M0 baseline.

## 4. Primary experiment: Leave-One-Family-Out

Outer folds: **5**, one per inventory family, shared by M0-M3. Inner alpha
selection is grouped leave-one-training-family-out, so no outer test family ever reaches
the penalty search or the standardisation statistics.

### Recall@10 (primary target)

| set | features | OOF MAE | OOF RMSE | OOF R2 | OOF Spearman | dMAE vs M0 | 95% CI | CI lower > 0 |
|:--|:--|---:|---:|---:|---:|---:|:--|:--:|
| M0 | log_er_l1 | 0.003753 | 0.005460 | +0.2960 | +0.6367 | — (baseline) | — | — |
| M1 | log_er_l1 + gap_l1 | 0.005250 | 0.007833 | -0.4488 | +0.4943 | -0.001497 | [-0.004785, +0.000341] | no |
| M2 | log_er_l1 + eff_usage_ratio_l1 + collision_l3 | 0.005226 | 0.007094 | -0.1883 | +0.2962 | -0.001473 | [-0.002160, -0.000046] | no |
| M3 | log_er_l1 + gap_l1 + eff_usage_ratio_l1 + collision_l3 | 0.005659 | 0.008092 | -0.5462 | +0.2246 | -0.001906 | [-0.004248, +0.000197] | no |

### NDCG@10 (secondary target)

| set | OOF MAE | OOF R2 | OOF Spearman | dMAE vs M0 | 95% CI | CI lower > 0 |
|:--|---:|---:|---:|---:|:--|:--:|
| M0 | 0.002102 | +0.2381 | +0.6212 | — (baseline) | — | — |
| M1 | 0.003068 | -0.5489 | +0.1485 | -0.000966 | [-0.002939, +0.000007] | no |
| M2 | 0.002817 | -0.2354 | +0.3632 | -0.000714 | [-0.001117, -0.000041] | no |
| M3 | 0.002951 | -0.4914 | +0.3196 | -0.000849 | [-0.001965, +0.000125] | no |

Both targets agree in sign and in ordering, which is the consistency the protocol asked for:
every increment is negative on Recall and negative on NDCG. Note that **M2's CI excludes zero
on the wrong side** — it is significantly *worse* than M0, not better.

### Family-level increments (Recall@10)

| set | TIGER_BEHAVIOR_SAMPLE_ | TIGER_FORK_RQ-VAE | TIGER_HYPERBOLIC_RQ-VA | curvature_RQ-VAE | TIGER_RQ-VAE |
|:--|---:|---:|---:|---:|---:|
| M1 | -0.00090 | +0.00074 | -0.00646 | -0.00082 | -0.00080 |
| M2 | -0.00163 | +0.00047 | -0.00208 | -0.00221 | -0.00149 |
| M3 | -0.00140 | +0.00069 | -0.00552 | -0.00196 | -0.00105 |

No set is positive in every family. M1 and M3 flip sign in one family each, and the sign of
the increment is therefore family-dependent rather than a stable property of the metric.

## 5. Diagnostic decomposition (H3)

With ER controlled, what do `pos_share`, `neg_share` and Gap actually do?

- `Gap = pos_share - neg_share` is a deterministic function of `pos_share` and `neg_share`,
  and M1 already contains it. The protocol's warning against feeding the ratio, the difference
  and both raw rates into one small-sample model is therefore respected: only the ratio and the
  difference ever enter a prespecified set, never all three.
- The spread of `neg_share` across models is far wider than the spread of `pos_share`,
  and it is driven almost entirely by how concentrated the L1 codebook is. That is the same
  collapse mechanism the R2/R3 rounds traced directly, and it is why Gap carries little
  information beyond ER: once ER is known, both of Gap's components are largely pinned down.
- The models with the highest Gap are frequently the ones with the worst collision rate, so a
  large Gap is not evidence of a usable representation. See `counterexamples.md`.

## 6. Prespecified sensitivity analyses

| analysis | n kept | M0 MAE | dM1 | dM2 | dM3 |
|:--|---:|---:|---:|---:|---:|
| multi-member families only (drops single-model `TIGER_RQ-VAE`) | 37 | 0.003921 | -0.002145 | -0.001497 | -0.002103 |
| drop the single worst-collision model | 37 | 0.003684 | -0.001143 | -0.001907 | -0.002176 |
| drop the three worst-collision models | 35 | 0.004152 | -0.001390 | -0.001399 | -0.001780 |
| L1+L2 complete-case (27 models), M3 + L1+L2 gap | 27 | 0.007101 | — | — | -0.000324 |

Every fixed sensitivity analysis keeps the same sign: the increments stay negative or become
more negative. Removing the most collapsed models, restricting to multi-member families, and
adding the L1+L2 gap on the reduced complete-case subset all fail to rescue the increments.

## 7. What this design cannot settle

These are limits of the study, not additional findings:

1. **Only 5 families.** The protocol anticipated roughly a dozen; the inventory
   actually groups into 5 ({"TIGER_BEHAVIOR_SAMPLE_RQ-VAE": 4, "TIGER_FORK_RQ-VAE": 9, "TIGER_HYPERBOLIC_RQ-VAE": 7, "curvature_RQ-VAE": 17, "TIGER_RQ-VAE": 1}). Leaving out `curvature_RQ-VAE`
   removes 17 of 38 models at once, and `TIGER_RQ-VAE` is a single model, so two of the five
   folds are structurally unlike the others. The family-cluster CI is correspondingly unstable,
   and no result here should be read as a well-powered family-level conclusion.
2. **One Stage 3 run per model.** With 57,439 evaluation users but a single training run per
   model, the cross-model spread mixes model quality with run-to-run noise. No variance across
   seeds has been measured and none is claimed.
3. **The 38 models are not an external sample.** They are the same models the earlier
   exploratory thresholds were fitted on. The only independence in this study is the
   behavioural-pair split and the cross-validation; the model set itself was never held out.
4. **Correlation, not causation.** Nothing here shows that changing a Stage 2 mechanism would
   move Recall@10. It shows only which existing Stage 2 readings transfer across families.

## 8. Historical descriptive association, kept separate

- Step 9 covers 38 rows and **38** of them join
  to this inventory on `<tree>__<arm> from step9 mapped to <tree>/<arm> in the inventory`.
- Train-pair L1 ER vs Recall@10 Spearman, reproduced on the joined models: **+0.8019** (n = 38).
- Valid-side L1 ER vs Recall@10 on the same models: **+0.7733**.
- Canonical Resample: train-pair L1 ER **20.1226** vs
  valid-side **16.38953942232631**. The two definitions are not
  interchangeable and are never mixed here.
- The old train-pair thresholds (ER>=14, Gap>=0.095, L1 distinct>=230, collision<=0.08)
  select **9** of the joined models. This is an
  old-phenomenon reproduction only: the thresholds came from post-hoc exploration of
  these same models and cannot validate themselves.

## 9. Conclusion

- **Confirmed:** Hard L1 ER alone is the strongest of the four prespecified Stage 2 readings
  for out-of-family prediction of Stage 3 Recall@10 across these 38 models.
- **Not confirmed (H1):** the L1 Gap does not add reliable out-of-family predictive value
  once ER is known. ΔMAE is negative and its CI includes zero.
- **Not confirmed (H2):** neither L1 effective usage nor three-token collision improves
  out-of-family prediction, alone or jointly. M2's CI sits entirely below zero, i.e. adding
  them is reliably harmful to LOFO transfer here.
- **Diagnostic value (H3):** Gap, `pos_share` and `neg_share` remain useful for explaining
  *why* a model collapses, and `counterexamples.md` shows high-Gap/high-collision models that
  would fool a Gap threshold. They are collapse diagnostics, not selection predictors.
- **Boundary (H4):** high ER is neither sufficient nor necessary for high Recall@10.
  `TIGER_FORK_RQ-VAE/prefix_sib` reaches Recall@10 0.061613 (rank 7 of 38) on L1 ER 3.5064,
  while `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` has the largest L1 Gap in the study
  (+0.24796) and a 0.8561 three-token collision rate for Recall@10 of only 0.053814. A raw-ER
  top-10 ranking (9/10 hits) still beats every fitted LOFO model (M0 5/10, M1 6/10, M2 1/10,
  M3 2/10), so even ER-only screening is a family-dependent heuristic rather than a validated
  selector. Details in `counterexamples.md`.

## 10. Reproducing

```bash
python3 scripts/test_joint_prefix_predictors.py          # 33 assertions
python3 scripts/analyze_joint_prefix_predictors.py       # this report
```

Feature definitions are byte-for-byte the ones in
`scripts/analyze_heldout_prefix_enrichment.py` and
`scripts/analyze_hbpe_r2_enrichment.py`; neither was modified. This analysis writes only
into `results/stage3_T5Train/monotonicity/joint_prefix_predictors/`.
