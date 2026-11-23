# VERDICT — delivered, pending, unavailable (issue #245)

Acceptance criterion for this issue is the infrastructure itself: production code, frozen data,
independent verification. Recall improvement is explicitly **not** required, and the issue forbids
starting any training.

## Delivered (usable now, no model needed)

| item | evidence |
|---|---|
| shared package, 10 modules | `tools/graph_infra/`, all imports resolve from the repo root |
| 102 tests across 9 suites | all passing (`REPORT.md` §5) |
| 17 mutations, 17 caught | `results/research/infra245/MUTATIONS.md` |
| directed graph + visible/eval split + counts matching the canonical graph | `assets/graph/`, `test_isolation.py` |
| ego subgraphs with explicit centre index, hop tags, induced edges, batching masks | `assets/ego/`, `test_topology.py`, `test_isolation.py` |
| typed motifs with degree strata and overlap counts | `assets/motifs/`, `test_graphlet.py` |
| pair banks with undirected leak blocking, eval-side banks, frozen probe | `assets/pairs/`, `test_pairs.py`, `test_isolation.py` |
| distance cache, full-vs-subgraph naming, PPR with its own unit | `assets/distance/`, `test_graph_distance.py` |
| unified metric caliber on the frozen TIGER SID | `baseline_tiger.json` |
| strict join for future model numbers | `assets.join_scores`, `test_assets.py` |

## Pending (needs something this pass must not provide)

| item | what it needs |
|---|---|
| reconstruction error, `D_geo`, `S_G`, gate values, expert scales | a candidate model forward pass; the contract fixes only the probe and the join |
| relation-strength Spearman on the banks | the same model output, joined through `assets.join_scores` |
| degree-preserving structural null | a separate rewiring pass; absent on purpose, stated in the report |
| curvature on the visible-only graph | #236 used the same graph version, so reuse is valid; a visible-only variant is a new asset version if a future experiment needs it |

## Unavailable / rejected

| item | why |
|---|---|
| `sids_for_hgrec.npy` as "the TIGER SID table" | it contradicts `item_sids.json` (17 shared triples of 20,425 / 21,873) |
| the §63 record's `l3_collision` value | irreproducible from its own fields; the gate caliber (0.1693) is used instead |
| Stage3 recall as a comparison in this pass | historical numbers were produced under older calibers and would not be a like-for-like baseline |
| any "graphlet-specific geometry works" claim | nothing was trained; that claim requires a Stage3 evaluation, which is out of scope here |

## What later experiments must do

1. import `tools/graph_infra/` — never copy geometry, graph or metric code into a new issue directory;
2. read `assets/manifest.json`, verify it, and use the named banks rather than resampling;
3. keep the experiment's own curvature weights, temperatures and scales as its own constants;
4. attach model outputs only through `assets.join_scores`, and report `subgraph_pass_fraction` and
   `center_all_pass_fraction` side by side;
5. if the shared interface turns out to be wrong, fix the package and re-run its suites first — then
   re-verify the affected assets before any candidate uses them.
