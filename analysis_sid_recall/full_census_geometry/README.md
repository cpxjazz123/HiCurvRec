# Full-census local prefix geometry (Issue #132)

Re-does the #131 study **without sampling caps**: every occupied L1/L2/L3 prefix
group of every admitted SID is enumerated and its induced-subgraph structural
geometry is computed exactly. Offline analysis only; no Stage 2 loss, codebook or
Stage 3 job is touched.

## Census rules

* No `max_group_size`, no `groups_per_level`, no item-coverage filter.
  Singleton groups are handled analytically and count in every denominator.
* Structural metrics (size, internal edges, connected components and sizes,
  largest-component fraction, triangles, transitivity, cycle rank, volume,
  boundary edges, conductance) are **exact for every group**.
* Expensive metrics are never sampled:
  * four-point Gromov deviation is `EXACT` on every connected component with at
    most `census.FOUR_POINT_MAX` nodes and `BLOCKED_NOT_FULL_CENSUS` above it,
    with the blocked counts reported (`PARTIAL_BLOCKED` at group level);
  * the path detour is `BLOCKED_NOT_FULL_CENSUS` (exact all-pairs needs one
    full-graph BFS per item, ~2.7 million BFS for this table) and is not
    reported.
* Every metric carries its measured group and item coverage in
  `group_support_L1_L2_L3.csv`, so a metric defined only on part of the census
  is never presented as an all-group value.
* Provenance: a digest mismatch is `HASH_MISMATCH` and the pair is excluded, not
  relabelled. Counts are in `completeness.json`.

## Files

| file | contents |
|---|---|
| `census.py` | groups, analytic singletons, exact induced metrics, exact four-point enumeration, BFS/vectorised distances, nulls |
| `run_census.py` | measurement runner, checkpoint/resume (`_checkpoint.jsonl`), optional `--per-group-out` |
| `run_stats.py` | per-level correlations with Holm, partial controls, nested models, the #131 comparison, figures and the report |
| `census_by_level.csv|json` | per SID: support, coverage per metric, aggregates, null deltas |
| `group_support_L1_L2_L3.csv` | per SID and level: occupied/singleton/multi/>=4, eligible vs measured counts, histograms |
| `completeness.json` | cap settings, excluded pairs, metric status and blockers |
| `settings.json` | frozen settings |
| `statistics.json`, `figure_audit.json`, `REPORT.md` | results, figure coverage, the write-up |
| `tests/test_census.py` | 13 contract tests: accounting, >64-item groups, >300 groups, tree/cycle/triangle/disconnected/singleton fixtures, blocked large components, parent null, digest mismatch, figure honesty |

## Reproduce

```bash
python3 analysis_sid_recall/full_census_geometry/run_census.py --resume
python3 analysis_sid_recall/full_census_geometry/run_stats.py
python3 -m unittest discover -s analysis_sid_recall/full_census_geometry/tests
```

## Status of the expensive metrics

`completeness.json` is the authority: it records the four-point size rule, the
detour blocker and the null design (3 replicates × 3 shuffle kinds, structural
census recomputed on every group in each replicate).
