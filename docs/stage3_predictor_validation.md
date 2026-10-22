# Stage3 historical predictor / mechanism diagnostics: 2026-10-22

**STATUS: implemented read-only diagnostic; NOT certified as an autonomous candidate-selection gate.**

## Provenance: two independent cohorts

Historical: 38 models, five canonical families, all evaluated on 57,439 Stage3 examples. The script cross-checks model IDs, family, Stage3 labels, 16-byte inventory SHA prefix against full audited SID SHA, and canonical frozen ER. Sources: model_inventory.csv, joint_prefix_predictors/stage2_valid_features.csv, and the CORRECTED history_hierarchy_predictors_fix48 per-model features and three-seed, history-length-20 probe scores.

DO NOT use history_hierarchy_predictors (without _fix48): #48's INVALIDATED.md documents held-out label leakage, row-indexing errors and faulty prior scores. #49 is the corrected successor.

#50's semantic-reconstruction report prose and its machine-readable summary disagree: its report claims M9 Recall MAE is unchanged at 0.003753, but its ablation_summary.csv says M9 = 0.004417 and M0 = 0.003753. These semantic scores are excluded until reconciled.

The independent #107 cohort is constructed by immutable manifest ID joins from results/harness3_runs/store/runs. Stage2-only rows are MISSING Stage3, never labelled as failures. Stage3-only rows without corroborating Stage2 ER are not forcibly patched from prose. Frozen Control is checked against exact original values. New Stage3 labels are NEVER used for historical regression fitting, feature selection or tuning.

## Recomputed historical five-family LOFO Ridge results

Outer leave-one-family-out; train-fold-only normalization; grouped inner CV for regularization; every comparator uses its own complete-case ER baseline.

| Features | Recall@10 OOF MAE | Delta MAE improvement vs ER on same cases | n |
|---|---:|---:|---:|
| log ER | 0.003753 | 0 | 38 |
| ER + G_H1 (corrected history-conditioned relative gain) | 0.002863 | +0.000890 | 38 |
| ER + D2 (conditional L2 discrimination) | 0.003169 | +0.000584 | 38 |
| ER + L1 positive-negative sharing gap | 0.005106 | -0.001353 | 38 |
| ER + effective usage + triple collision | 0.005035 | -0.001282 | 38 |
| effective usage + triple collision alone | 0.007324 | -0.003571 | 38 |
| ER + L1+L2 ER | 0.004397 | -0.000044 | 27 |

The ER-only historical benchmark exactly reproduces #47. The corrected #49 family-cluster bootstrap confidence intervals for the G_H1 and D2 improvements cross zero. There is NO prospectively confirmed feature improvement. The L1+L2 subset is smaller; never compare its scores directly with the full 38 baseline.

## #107 completely joined subset: nine experiments plus control (n=10)

| Diagnostic | Spearman rho with Recall@10 (n=10) |
|---|---:|
| Frozen L1 ER | +0.030 |
| L1 Gini | +0.030 |
| Mean L1-to-L2 branches | +0.394 |
| Simple, in-sample next-prefix sequence information excess | -0.139 |

No causal interpretation or reliable significance at n=10. Historical ER ranges from 1.03467 to 16.43458; EIGHT of these TEN new paired records exceed that maximum. Such historical-model predictions are out-of-distribution extrapolations, not calibrated Stage3 screening claims. The two cohorts must not be naively pooled.

## Deployment boundaries

The analysis is read-only: it never trains Stage2/Stage3, regenerates frozen evaluation, retries failed arms, or overwrites source evidence. Predictions must identify held-out family, cohort, ER OOD state, paired/unpaired status, and candidate uncertainty. In-sample sequence excess is descriptive, not a validated multi-item TIGER simulator. The fourth SID token is a collision extension and is not a semantic prefix token.

Run in a repository checkout (requires pandas/numpy/scipy/scikit-learn):

    python scripts/stage3_evidence_audit.py --out results/harness3_runs/stage3_evidence_audit
    python -m pytest scripts/test_stage3_evidence_audit.py -q

Reproducible outputs: historical_oof.csv, historical_lofo.csv, campaign_manifest_audit.csv, campaign_holdout.csv, campaign_mechanism_diagnostics.csv, evidence.json, and README.md. Until prospective new-family confirmation, DO NOT install any predictor as a hard Harness selection gate or claim it predicts improved Stage3.
