### #261 freeze + inventory

- body sha256 `022ad858552a4d067b9a0e15abdff5d8f99e58404ad3d5574efe31cd59c38e9d`
- `INFRA_MODE=FROZEN`; issue is analysis-only and forbids new training.
- 8/8 bundles checked by reading local immutable Stage2/Stage3 artifacts and the R2 `recall.json` copies:
  - checkpoint, SID, archive binding, export manifest and R2 recall hashes PASS for 100–107;
  - 3000/3000 training epochs + `train_end` PASS;
  - prediction-shard row count = `n_eval=57,439` PASS;
  - Stage2 layer record SID hash PASS.
- candidate source/config hashes PASS for all eight. For 100–102 the historical `infrastructure_sources` receipt differs from current public tools (later infra commits); this is recorded in `INFRA_FAULTS.md` and `INVENTORY.json`, not repaired under FROZEN mode.
- full-table 3-expert curvature/reconstruction evaluation is running from frozen checkpoints; no optimizer update.

Evidence: `results/research/issue261/INVENTORY.json`, `INFRA_FAULTS.md`, `checks/`.
