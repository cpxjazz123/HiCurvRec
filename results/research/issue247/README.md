# Issue 247 assets — sectional curvature and node-local topology (v3)

All assets live in `results/research/issue247/assets/` and are described by
`manifest.json` (schema, dtype, shape, sha256, coverage, generation command, runtime, storage cost).
Rebuild/verify with one command:

```bash
bash tools/graph_infra/verify_v3.sh
```

The command re-runs the unit suites and then `python -m tools.graph_infra.prepare_v3`, which
**verifies** an existing manifest (it never silently overwrites one) and recomputes every hash,
re-derives the sampled triples, the reference distances and the curvature values, re-checks the
ID alignment, the split isolation, the directed two-hop membership and the TIGER collision.

Nothing here trains a model, defines a loss, computes PPR or touches the frozen v2 assets.

## Graph and splits

| Asset | Content |
|---|---|
| `item_ids.npy` | item index map (0..N-1), checked against the 768-d embedding rows |
| `directed_edges.npz` | visible directed edges: source, target, weight, reciprocal flag |
| `directed_out_within2.npz` / `directed_in_within2.npz` | sparse directed one/two-hop neighbourhoods of the visible graph |
| `manifest.json` entries `visible_forward`, `visible_reverse`, `split_users`, `held_out_support` | frozen v2 graph and split assets (read-only reuse) |

Splits (frozen in #246, re-verified here): visible 45,951 users / struct_val 5,744 / struct_test 5,744,
pairwise intersection empty, union complete. Every hidden edge is used for evaluation only.

## Sectional curvature (GraphMoRE Appendix A.1)

`sectional_triples.npz` — one triple per eligible centre:

* `triples` int64 `[T,3]` = (centre m, neighbour b, neighbour c), b < c, both adjacent to m;
* `references` int64 `[256]` — the fixed seeded reference set (nested prefixes 16/64/256 used by the sensitivity study);
* `value` float64 `[T]` — the A.1 statistic averaged over the reachable references, **NaN where no reference is reachable**;
* `effective_references` int64 `[T]` — how many references actually contributed;
* `valid` bool `[T]` and `invalid_reason` — the empty-sample mask, never zero-imputed;
* `endpoint_distance` float64 `[T]` — g(b,c) (1 or 2);
* `directed_graphlet` int64 `[T]` — canonical 3-node directed class of the triple;
* `neighbor_pair_population` int64 `[N]` — how many unordered neighbour pairs each centre has (population, not sample);
* `reference_contributions` float64 `[T,R]` — the per-reference parallelogram residuals.

`sectional_summary.npz` — per item: `count`, `valid`, `mean`, `std`, `negative`, `zero`, `positive`
(proportions), `quantiles` `[N,5]` (0, .25, .5, .75, 1).

`reference_distances.npy` float64 `[256,N]` — the reference-to-node shortest-path distances
(unweighted, on the undirected projection of the visible graph).

Statistic (paper Eq. A.1): `kappa(m;b,c) = mean_a [ d(a,m)^2 + d(b,c)^2/4 - (d(a,b)^2 + d(a,c)^2)/2 ]`.
Units: squared hops. This is a sectional statistic on the **undirected projection** of the visible
directed graph — it is *not* a directed sectional curvature, and *not* a Ricci curvature.

## Node-local topology

`topology_features.npz` — per item: `degree_out`, `degree_in`, `reciprocal_degree`,
`family_counts` `[N,16]` (exact 3-node family counts), `family_distribution`,
`family_valid`, `sampled_orbit_counts` `[N,K]`, `sampled_orbit_distribution`, `sampled_orbit_valid`.
`orbit_schema.npz` — the `(graphlet, orbit)` keys of the orbit columns.

These are **deterministic hand-computed topological features**, not learned embeddings.
Orbit columns count the centre's orbit participation in the *sampled* 3-node instance bank
(#246 sample: 2,000 centres); they are not full-population orbit totals.

Multi-scale ego: `TrainingAssets.ego(item, radius)` returns the exact induced radius-1 or radius-2
subgraph from the visible CSR with `root_local`, `hop`, `role_out`, `role_in`, `features`,
`node_mask` and `edge_mask`. The sampled v2 ego assets remain available under their own names
(`ego_full_1hop`, `ego_sampled`).

4-node instances (`motif_instances4`, v2) are a **sample**: 500 centres, first 12 sorted neighbours,
at most 10 triples per centre. Coverage, sampling mechanism and the resulting bias are recorded in
`coverage.json`; no unbiased population estimate or confidence interval is claimed.

## Curvature distribution summary

`coverage.json` holds the reference sampling rule, the triple sampling rule, the zero tolerance,
the per-reference-count sensitivity table (mean/max absolute change and sign-change fraction between
nested reference sets), the disconnected-graph estimand, the runtime of the distance computation and
the storage cost.

## Interfaces

```python
from tools.graph_infra.interfaces_v3 import TrainingAssets, HeldOutEvaluation
train = TrainingAssets(repo)              # training features only; cannot read evaluation targets
block = train.ego(item_id, radius=2)      # exact induced ego subgraph
held  = HeldOutEvaluation(repo)            # strict held-out targets, requested explicitly
```

`TrainingAssets.load` refuses any asset whose declared use is `evaluation`, and
`HeldOutEvaluation.load` refuses anything that is not a `strict_` evaluation bank.

## TIGER baseline

`tiger.json` — the Stage3-read `item_sids.json` with its sha256, prefix-3 collision and duplicate
share, per-level codebook usage/entropy in bits, prefix sharing per structure bank, and the
separately identified checkpoint and NPY. Historical recall and reconstruction error are `NA`:
no immutable historical SID hash binding to a completed recall was established, and the frozen
checkpoint belongs to another SID generation.
