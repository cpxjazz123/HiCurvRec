### #261 analysis complete: correlations and limits

- 8/8 bundles checked with actual checkpoint/SID/archive/export/R2-recall hashes; 3000/3000 epochs; 57,439 prediction rows. Candidate source hashes pass. 100–102 have historical public-infra receipt drift only, recorded in `INFRA_FAULTS.md`.
- Full-table 28,391,948-triple hard-q / continuous-z evaluation completed for all eight with one independent 3-expert metric.
- Key test-R@10 associations (n=8): L1 positive +0.856 (CI +0.382..+0.974), L1 ER +0.839, L2 distinct −0.571, L3 collision −0.193, hard-q curvature micro-MAE −0.724, reconstruction L2 error −0.885, L1 codebook spacing −0.812. All exploratory; architecture confounding and small n are explicit.
- R06b/107 counterexample: 52.19% L2 code change and per-role 0.413–0.575, but test R@10 0.049600 vs R05 0.050349; code movement alone is not a recommendation gain.
- Missing items are listed with reasons in `MISSING.csv`; no NA was used for an executable item.
- Evidence: `results/research/issue261/{REPORT.md,VERDICT.md,METRICS.csv,CORRELATIONS.csv,SENSITIVITY.csv,FULL_METRICS.csv,TOKEN_GATE.json}`. No training.

Final commit/push SHA is appended below.
