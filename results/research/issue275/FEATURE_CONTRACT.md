# Issue #275 — FEATURE_CONTRACT (frozen 2026-11-28 13:05, before any search)

## Label
* Unit: one SID sha256 (a checkpoint's export). All Stage3 records of that SID are kept.
* Primary label: mean `test_recall@10` over the records at the SID's majority `stage3_world_size`;
  other-protocol records are sensitivity only. `noise_range` = max−min over the used records.
* Labels come only from stored `test_final.json`; no value is imputed.

## Identity channels (no fabrication)
1. `stage3_identity`: the record itself carries `sid_sha256` + `stage2_checkpoint_sha256`.
2. `r2_ledger`: legacy record matched by name to `results/research/issue139/r2_ledger.json`
   (`pair.variant`/`pair.stage2_rel`), taking the ledger's recorded SID sha and R2 id.
3. `missing`: no SID file and no ledger match — excluded from every matched analysis, counted.

## Feature definitions (all recomputed with frozen evaluators; one definition each)
SID core queue (`Q_SID`, six features, all from `tools/issue252/layer_metrics.py` with the frozen
SID loader `tools.issue252.final_evaluation.load_sid` and `valid.parquet`):
* `l1_pos_hit` = `validation_metrics.l1_positive_hit_rate` (behavioural L1 positive shared rate).
* `l1_neg_hit` = `validation_metrics.l1_negative_hit_probability`.
* `l1_hard_er` = `validation_metrics.l1_hard_er`.
* `l2_distinct` = `catalogue_metrics.distinct_l2_prefixes` (capacity).
* `l3_collision` = `catalogue_metrics.l3_collision` = 1 − distinct triples / items.
* `l3_refine_bits` = `catalogue_metrics.l3_refinement_gain_bits` (conditional refinement).
No other collision definition is mixed in. Old `stage2_metrics` records carry a different
`valid_l1_*` definition and are **not** mixed into this table (they are only referenced for identity).

Vector queue (`Q_VEC` = Q_SID ∩ vector cache): adds
* `quant_error` and `effective_rank` from the frozen gate cache
  (`~/.cache/genrec-mcp/stage3__*.json`, `sid_quality.quant_error/effective_rank`), one definition
  each, keyed by the on-disk SID sha; PR(q) and PR(z) are not mixed (only the gate's effective rank
  of q is used; z-based PR is reported unavailable).
* `recon` = the run's final-epoch training `recon` from its `training_metrics.jsonl` (Stage2 MSE).
  Never mixed with `quant_error`.

Geometry queue (`Q_GEO` = Q_SID ∩ #271 geometry assets): adds
* `geo_mae`, `geo_spearman`, `geo_sign_acc` from `results/research/issue271/assets/geometry/*.json`
  matched strictly by `sid_sha256`; name-only matches are reported as sensitivity, never pooled.

## Queues (matched comparisons only)
* `Q_SID`: layer features present (54 of 62 label units).
* `Q_VEC`: Q_SID ∩ quant/rank (26).
* `Q_GEO`: Q_SID ∩ geometry (3) — declared **unidentifiable** for any increment claim.
Missingness per queue is reported; missing features are never filled with labels or means across
queues.

## Provenance
Every feature file records the evaluator path+sha256 and the input parquet paths+sha256
(`results/research/issue275/features/layer/<sid_sha>.json`, `layer_run.json`).
