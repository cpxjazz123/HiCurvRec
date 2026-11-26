# Issue #262 — frozen analysis contract

Frozen before any #262 metric computation (2026-11-26). The issue body is
`ISSUE.json` (sha256 in `INFRA_SNAPSHOT_START.json`). This is an analysis-only
issue: no Stage2/Stage3/TIGER training, no optimizer step, no parameter update,
no SID rewrite, no change to mechanisms, public infrastructure or frozen assets.

## Scope and identity

The eight completed R2 bundles 100–107 are inherited from #261. Their Stage3
`test_final.json` is the only recommendation-test result; validation records are
never replaced by test values. #261's metric contract and row identities remain
valid. Existing evidence is reused only where its population and definition match;
new files are below `results/research/issue262/` and are derived by the scripts in
this directory.

## Definitions

Let the frozen directed training graph be
`results/research/behavior_graph_directed/{edges.parquet,graph.npz,metadata.json}`:
`u -> v` means the ordered transition `i_t -> i_{t+1}` in a TRAIN sequence. The
undirected graph must never be used to claim a direction.

For an item `i`, `P(i)` is its set of in-neighbours and `S(i)` its set of
out-neighbours. For an ordered pair `(u,v)`:

- `J_in(u,v) = |P(u) ∩ P(v)| / |P(u) ∪ P(v)|`, missing if the union is empty;
- `J_out(u,v) = |S(u) ∩ S(v)| / |S(u) ∪ S(v)|`, missing if the union is empty;
- `common_in` means the intersection `P(u) ∩ P(v)` is non-empty; `common_out`
  is defined analogously;
- `direct` means `u -> v`; `reverse` means `v -> u`; `two_hop` means `u -> k -> v`.

Held-out structural banks are read from the frozen `assets/manifest.json` and
`results/research/infra246/assets/pairs/`. Their pair universe, source graph and
features are audited separately: #261's `jaccard_in/out` fields were computed on
the visible graph even for strict banks whose pairs came from struct_val/test.
#262 therefore recomputes the pair label on the graph named by the bank, and
reports the visible-graph original as a separate diagnostic. The struct_test
bank is a structure-test population, not the recommendation test population;
any association with `test_R@10` remains exploratory.

## Curvature, quantisation and gates

- Hard `q` is the sum of the three actual codebooks at the exported SID indices;
  the fourth SID column is a collision counter and is not a code.
- The frozen cross-arm curvature prediction is the equal-weight three-expert
  (`-1,0,+1`) tangent-space distance, using the checkpoint's frozen
  `tangent_scale`; no scale is fitted on evaluation.
- Target curvature is the frozen `assets/v8_runtime_layout/targets.npy` integer
  proxy, read through the frozen shard validity masks. No target is regenerated.
- Full-table metrics: Pearson, Spearman, RMSE, MAE, sign accuracy and per-sign
  precision/recall/F1, plus MAE/RMSE by centre role and directed total-degree
  quintile. Macro/micro are both reported and never conflated.
- Per-level quantisation diagnostics use Euclidean distance in the actual
  codebook space: selected-vs-nearest, top1/top2 margin, normalised gap, residual,
  selected-code rank, and the checkpoint's saved quantisation weights. The
  training-time commitment/codebook values are reconstructed from the source
  registry; historical trajectories remain NA.
- Switch interventions are read-only: no-intervention export must reproduce the
  SID exactly; behaviour-off, curvature-off, uniform gate and Euclidean+uniform
  are separately recorded. A failed intervention is reported as unavailable,
  never replaced by a retrained effect.

## Gradients and historical validation

Endpoint gradients use one common fixed training batch (4096 catalogue items,
seed 42) and the source's own `combined_loss`; parameters are frozen and a
state-dict checksum is compared before/after. Endpoint gradient is not presented
as historical optimizer trajectory.

Stage3 validation metrics are accepted only when an identity-bound saved record
(`results/research/issue256/rounds/r01..r04/recommendation_validation/`) has
`training=false`, `test_access=false`, the same final checkpoint/SID and a
recomputable prediction shard. R05/R06 and 100/101 have no such record: NA.

## Statistical scope

n=8, exploratory. Pearson/Spearman are descriptive; Fisher intervals and
leave-one-out/leave-100/leave-107 sensitivity are shown when defined. No causal
claim, no multiple-comparison-adjusted discovery claim, no new model selection.
Missing values stay NA; never 0, never imputed.

## Acceptance

All legally computable #261 missing metrics that this issue names are filled;
truly unavailable items have a concrete attempted path. Every result is
recomputable from retained products or sufficient statistics. Test labels and the
recommendation test population are never used for training, parameter update or
mechanism selection. Public infrastructure and all frozen assets keep identical
`path:sha256`. `VERDICT.md` answers every acceptance item and records residual
limitations.
