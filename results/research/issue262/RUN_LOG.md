# Issue #262 run log

## 2026-11-26 — intake and freeze

- Read issue #262 from GitHub; saved the exact JSON in `ISSUE.json` (addresses
  `cpxjazz123/HiCurvRec#262`, OPEN, 3,560-character body).
- Classified as **analysis only** from the issue text: “不启动 Stage2/Stage3/TIGER
  训练,不增加种子或训练臂,不做 optimizer 更新,不修改已有机制、SID、公共 Infra、资产或历史结果”.
  No training or GPU job will be dispatched.
- Read #261's `METRIC_CONTRACT.md`, `MISSING.csv`, `REPORT.md`, `INVENTORY.json`,
  `CORRELATIONS.json`, `FULL_METRICS.json`; these are the baseline to audit, not
  independent truth.
- Found #261's `+0.958` row: it is
  `hard_q.strict_heldout.banks.strict_struct_test_shared_in.topology.jaccard_in.distance_spearman`
  (cross-arm Pearson with `test_recall@10`). It is a **cross-arm correlation of a
  within-arm distance/label Spearman**, not a shared-in fidelity value. The
  per-arm value is only about −0.039 to −0.075. This distinction is frozen in the
  contract and will be audited, not reused as a headline.
- Found #261 `MISSING.csv` claims Stage3 validation metrics have no
  identity-bound artifact. A counterexample exists for R01–R04:
  `results/research/issue256/rounds/{r01..r04}/recommendation_validation/` has
  `training=false`, `test_access=false`, the final checkpoint/SID identity and
  four prediction shards. These will be independently reduced and used only if
  they pass the identity and reduction checks. R05/R06 have no such record.
- Public-infrastructure/asset snapshot is taken before the analysis; no source,
  asset, checkpoint or SID will be edited.

## 2026-11-26 — measurements

- Identity audit completed: all eight local checkpoint/SID/test-final identities
  pass; the archive binding's historical public-infra drift for 100–102 remains
  reported, while candidate sources pass. R2 source tar is absent from the
  retained local R2 download; this is NA, not a reconstructed artefact.
- Stage3 validation correction: R01–R04 (`102–105`) have identity-bound
  recommendation-validation records with `training=false`, `test_access=false`
  and independently reduced `rank*.jsonl` shards. Their reported metrics match
  the reduction to ≤1e-10. #261's blanket NA is corrected; 100/101/106/107 remain NA.
- Full curvature: all eight arms evaluated on 28,391,948 valid frozen triples
  with exact coverage. Example 102: Pearson −0.0108, Spearman −0.1022,
  RMSE 2.2277, MAE 1.8815, matching #261's micro MAE to rounding.
- Quantisation/gradient/switch: all eight endpoint receipts complete; every
  state-dict checksum is unchanged and optimizer_steps=0. Actual 102 no-intervention
  replay differs from the frozen exported SID by 0.74% on both CPU and a
  read-only L40S replay; all switch values are therefore reported as interface
  probes, not as zero-baseline causal effects.

## 2026-11-26 — final results

- H1: raw target common-in +0.1338 over the degree/popularity negative;
  relation-matched +0.1070; both sides without direct/reverse/two-hop +0.0611
  (n=30,771). Status PARTIALLY_SUPPORTED: the raw target advantage is positive
  and degree-robust, but the target also carries direct/reverse/two-hop and
  prefix structure, and the paired logistic common-in coefficient is not
  positive once those controls are present.
- H2: #261 `+0.958` is a cross-arm Pearson of a **visible-graph** feature with
  mean 0.0041 on pairs whose true directed shared-in Jaccard mean is 0.2035.
  Recomputing with the directed bank label gives cross-arm Pearson −0.8682;
  the original headline is not shared-in fidelity. Status SUPPORTED.
- H3: complete 28,391,948-triple curvature per arm; lower curvature MAE is
  associated with higher test R@10 at n=8 (Pearson −0.724), but this is
  architecture/capacity-confounded and descriptive only.
- Tests: 8 issue-local known-answer tests pass. Public/infrastructure assets
  have identical start/end `path:sha256`; no training, optimizer step or SID
  rewrite was performed.



- `INFRA.md` frozen.
- `HYPOTHESES.json` pre-registered.
- Next: audit inventory/missing, then derived measurements.
