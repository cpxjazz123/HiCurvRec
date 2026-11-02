# Interaction-strength-aware SID quality (Issue #130)

Observational, offline analysis: does the *strength* of observed item-item
interactions — not merely the existence of an edge — explain SID prefix
organisation and the held-out Stage 3 Recall@10? No Stage 2 loss, tokenizer,
codebook or Stage 3 training is touched.

## Declared construction (frozen in `config.json` before any Recall correlation)

* **Unit**: one training *context* (a row's `history` plus its `target`); an
  unordered consecutive pair counts once per context. `f(i)` is the number of
  contexts containing `i`. A distinct-user count is stored as a declared unit
  sensitivity (`distinct_users`, `distinct_user_activity` in `weight_audit.json`).
* **Primary strength**: `s(i,j) = c(i,j) / sqrt(f_i f_j)` (popularity adjusted);
  `log1p(c)` is the declared intensity alternative.
* **Weighted ORC**: weights enter only the lazy random-walk measures; the ground
  metric is the **unweighted** hop distance (the #128 control). The transport is
  computed exactly with a two-step max-flow identity, so no OT package is needed
  (`strength.w1_hop12`, validated against `quotient_curvature.transport`).
* **Quotient aggregation**: sum of item strengths over cross-group pairs;
  primary normalisation `exposure` (cell / `sqrt(M_a M_b)`, `M` the cross-group
  mass), `raw` reported alongside.
* **Budget**: 24 sampled quotient edges and 2 size-preserving null replicates per
  (arm, level) — the measured budget, declared here and in `config.json`.
  `nulls` for the retention/gap statistics: 60 label shuffles and 40
  activity-stratified shuffles per arm and level. A larger 48-edge / 4-replicate
  sensitivity was declared but **not run**; the config records that, and no
  larger-budget claim is made.

## Files

| file | contents |
|---|---|
| `strength.py` | reconstruction, strengths, weighted retention + nulls, strength bins, within/between gap, weighted quotient ORC, the max-flow transport |
| `run_measure.py` | the measurement runner; writes the per-row table and the weight audit |
| `run_stats.py` | univariate statistics with Holm, the nested A/B/C/D models with leave-one-series-out RMSE, the tier sensitivity, the figures and the report |
| `experiment_by_level.csv|json` | one row per (SID content SHA-256, Stage 3 run), every feature per level |
| `weight_audit.json` | counts, mass, activity, mapping hash, zero/positive edges, heavy-tail summary |
| `statistics.json` | the statistics and the nested comparisons |
| `REPORT.md` | Q1-Q4 answered separately, including negative and undefined outcomes |
| `scatter_*_L{1,2,3}.svg` | per-level strength-vs-Recall and weighted-ORC-vs-Recall scatter plots |
| `tests/test_strength.py` | 10 contract tests |

## Reproduce

```bash
python3 analysis_sid_recall/interaction_strength/run_measure.py   # the measurement
python3 analysis_sid_recall/interaction_strength/run_stats.py     # statistics, figures, report
python3 -m unittest discover -s analysis_sid_recall/interaction_strength/tests
```

## Limits (kept explicit, never filled in)

* The join is by SID content hash; the provenance tiers of #129 are carried over
  unchanged (`HASH_PROVEN` / `RECORD_HASH_MATCH` / `MISSING`).
* Where a level's within-prefix support is too small the row says
  `UNDEFINED_INSUFFICIENT_WITHIN_EDGES`; no zero is substituted. The L3 singleton
  trap is expected and is reported as such.
* The weighted ORC budget is 24 edges / 2 replicates; no 128-edge or 8-replicate
  claim is made for it.
