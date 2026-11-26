# Issue #267 infrastructure record

- `INFRA_MODE=FROZEN`; issue body sha256 `fe40b2ba18899a41b2ef62a121a315923cb60f4cced22f2b233d255b5fb4301e`.
- Start snapshot: `INFRA_SNAPSHOT_START.json` (61 protected files, 0 missing).
- Old mechanism archived before the new arm: `stage2_RQ-VAE/TIGER_euclid_local_curve_supervision_rq`
  -> `stage2_RQ-VAE/_archive/TIGER_euclid_local_curve_supervision_rq_issue267_parent`.
- New candidate: `stage2_RQ-VAE/TIGER_relative_curvature_ranking_rq`, copied from the
  archived #265 source; only `ARM_NAME`, the relative term, its calibration wiring and
  mechanism tests differ.
- Reused read-only assets: `assets/manifest.json` (`cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`),
  `assets/v8_runtime_layout/targets.npy` (`8c0ff9f8c25b98ed31802cdb62cf921df6d3cf8bbf35ee2d3de9817d6cd0807a`),
  `stage1_GeneEmbedding/output/sentence_t5.npy`, frozen token cache
  (`4051b902934e72f0a06cd6bb657d67d7eec9c9a1c2304c10a31c848515492837`) and the
  #263 pretrained token/GNN dependency.
- Stage2 interpreter: `genrec_env_v2/bin/python3.9`; Stage3/Stage1 interpreter per CLAUDE.md.
- GPU layout: four A100-class devices visible, no running training process; launcher owns
  port 50202 and the four-rank DDP recipe.
- No public `tools/`, asset, baseline or Stage3 entry was modified.

## Archive and candidate

- Archived old top-level tree:
  `stage2_RQ-VAE/TIGER_euclid_local_curve_supervision_rq` →
  `stage2_RQ-VAE/_archive/TIGER_euclid_local_curve_supervision_rq_issue267_parent`.
- Candidate source copy: `stage2_RQ-VAE/TIGER_relative_curvature_ranking_rq`.
  Only `ARM_NAME`, relative term/calibration wiring and mechanism tests differ.
- Mutation check: `analysis/mutation_check.py`; all five mutations caught
  (`analysis/MUTATION_RESULTS.json`, `logs/mutation_check.log`).
- Candidate suite: `pytest -q tests` → 139 passed; relative-specific suite 7 passed;
  actual calibration smoke: 378 valid pairs, alpha ~0.3076 at formal init.

## End of run

- `verify_infra.py` compares all 61 protected paths against `INFRA_SNAPSHOT_START.json`.
  `INFRA_DIFF.json` → `status=PASS`, `changed=[]`, `missing=[]`; end snapshot
  `INFRA_SNAPSHOT_END.json`.
- R2 archive (CLAUDE.md §6b): `r2:genrec/110/` — 115 files, `recall.json` 23 keys,
  four-part layout (`<arm>/stage2_source.tar.gz` incl. `archive_binding.json` and
  `export_manifest.json`, `item_sids.json`, `rqvae_best.pth`, `recall.json`).
  Ledger entry recorded in `results/research/issue139/r2_ledger.json`; no credential
  written to any log or manifest.
