# Singleton-aware local geometry of SID prefix groups (Issue #131)

Diagnostic study of the *original* TRAIN behaviour graph inside the multi-item
groups that RQ-VAE prefixes actually form. A nonempty prefix is a valid address
even when it holds one item; the quotient-graph analyses (#129/#130) count such
a prefix as a node, so this directory reports **group support separately from
local geometry** and treats the L3 singleton trap as a first-class outcome.

## What is measured

* Support per level: occupied prefixes, singletons, multi-item groups, >=4
  groups, the size histogram and quantiles, item coverage by multi-item and by
  >=4 groups (group- and item-weighted).
* Local induced-subgraph metrics inside every eligible group (`2 <= |G| <= 64`,
  declared): four-point deviation (sampled, normalised by the component
  diameter; a tree gives zero), transitivity, cycle surplus per node, connected
  components and largest-component coverage, boundary edges, conductance, and a
  path-detour ratio on a declared subsample.
* Nulls: fixed-size global label shuffle, activity-stratified shuffle, and a
  **parent-conditioned** shuffle for L2/L3 that permutes children only inside
  their own parent and preserves the child-size multiset. Same estimator,
  eligibility rule and sample budget in real and null conditions.
* Hierarchy: parent-to-child maps for L1->L2 and L2->L3 and the split
  categories (several multi-item children / one multi-item plus singletons /
  only singletons).

## Files

| file | contents |
|---|---|
| `geometry.py` | groups, support audit, induced metrics, nulls |
| `run_geometry.py` | measurement runner, frozen settings |
| `run_stats.py` | univariate statistics with Holm, the nested A/B/C models with leave-one-series-out, the resolution-confound table, figures and the report |
| `group_support_L1_L2_L3.csv` | per SID and level support audit |
| `geometry_by_level.csv|json` | per SID and level geometry, null deltas/effects, hierarchy counts |
| `settings.json` | eligibility rule, sample budgets, seeds |
| `figure_audit.json` | defined points per figure; the test fails if a supported feature is drawn as empty |
| `REPORT.md` | Q1-Q5 answered separately with the measured/undefined status |
| `tests/test_geometry.py` | contract tests: singleton handling, monotonic occupied prefixes, non-monotone multi-group counts, induced exactness, connectivity, four-point on tree/cycle, boundary edges, parent-conditioned null, train-only isolation, hash join, no provenance relabelling, figure honesty |

## Reproduce

```bash
python3 analysis_sid_recall/local_prefix_geometry/run_geometry.py
python3 analysis_sid_recall/local_prefix_geometry/run_stats.py
python3 -m unittest discover -s analysis_sid_recall/local_prefix_geometry/tests
```

## Limits

* Budget: at most 300 eligible groups per (SID, level), 2 null replicates per
  shuffle kind, 8 four-point quadruples per group, 2 detour pairs per sampled
  group. Larger groups (>64 items) are reported as excluded, not measured.
* Standardised effects are left undefined when the null spread is below 1e-3.
* The join is by SID content hash; provenance tiers are those of #129
  (`HASH_PROVEN` / `RECORD_HASH_MATCH` / `MISSING`) and are never relabelled.
* The strict A2 subset is one mechanism series, so leave-one-series-out
  prediction is undefined there and is reported as such.
