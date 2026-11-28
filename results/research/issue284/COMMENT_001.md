**Frozen checklist + INFRA_MODE (start of #284)**

- Acceptance C1..C5 frozen from the issue body (sha256 `26fa8b44…a168`, 3035 chars) in
  `results/research/issue284/RUN_LOG.md`; each C carries requirement and criterion.
- **INFRA_MODE = FROZEN** (issue states it explicitly); **training required: no** — "不重训
  TIGER/RQ-VAE/T5,不启动Stage2/Stage3、不运行新的最终test". Analysis-only.
- Reference commit #283 = `801048ca`; its `JOIN_SUMMARY`, `JOIN_TABLE.part1/2`, `MATCHED`,
  `MODEL_COMPARISON`, `VALIDATION_*`, `STATS_MANIFEST` are the inputs (sha256 recorded).
- Identity: 129 labels (120 strong / 9 weak; 13 high / 6 strong), 124 prospective SIDs; two new
  Stage3 results bind by exact SID+run identity in the R2 ledger (114 pairrank_enc 9f6bf6731e88…,
  115 codebook_budget_j 61f898498c19…) → time-postordered labels, not blind test.
- Candidates pre-registered: A = six #277 features; B = A + all1_chain (main hypothesis);
  C = A + ALL_norm (marginal-conditional residual diagnostic); D = A + all1_chain + cond_refine;
  inner-fold-only selection; BH over the three comparisons.
- No public tool/asset modified; `INFRA_SNAPSHOT_START` will be written before any statistic.
