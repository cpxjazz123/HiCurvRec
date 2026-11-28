**Frozen acceptance list + INFRA_MODE (start of #283)**

- Acceptance list C1..C5 frozen from the issue body (sha256 `87367c48106f705043db17003c9f8aae29f1f2b510dd976356e00d71bf412e0a`,
  2601 chars) in `results/research/issue283/RUN_LOG.md`; each C carries criterion, threshold, data and
  wording.
- **INFRA_MODE = FROZEN** — the issue body states it explicitly; no BUILD authorization sentence.
  Training required: **no** — "不重训TIGER/RQ-VAE/T5,不启动Stage2/Stage3、不运行新test", "任何情况下不启动Stage3".
  Analysis-only acceptance, all recomputable from existing artifacts.
- Assets referenced by hash (`assets/manifest.json`): 13 triplet shards 28,391,948 rows all valid;
  visible_forward `ac7155…`; index `a5eb29…`; #276 SID table/folds `4c9279…`.
- Protected-component snapshot written (`INFRA_SNAPSHOT_START.json`); no public tool/assets modified.
- Root-cause diagnosis and inherited hypotheses: `DIAGNOSIS.md`, `HYPOTHESES.json` (H28–H31 inherited
  from #282, new H32–H36). Algebra frozen: given pairwise marginals, the joint 5-partition has exactly
  one free parameter, ALL = P(a=m=b) (AM = p_am−ALL, MB = p_mb−ALL, AB = p_ab−ALL).
