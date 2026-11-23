# Issue252 frozen contract — phase1, 2026-11-24

## Authority, goal and gates

Owner issue252 explicitly authorizes resolving #251's graph-identity conflict, performance engineering and one formal Stage2/Stage3 pair. Original issue body SHA `3e9b8ca4018cd9d0a43648efc42cd6a12ef27bc892fdb981590030bea37849eb`; all14 checkboxes are copied verbatim to ACCEPTANCE.json. Completion requires graph admission PASS, inclusive >=30 consecutive four-card epochs mean<=0.5seconds, formal Stage2 exactly3000 epochs, corresponding legal SID, formal Stage3 exactly150 epochs/final test, R2 archival and source-bound independent reconciliation. Short validation never substitutes for formal completion.

No formal run may start before graph/maths/gradient/DDP/entry and the performance gate pass. No changed768/32 dimensions, three256-entry books, original reconstruction/commitment/codebook loss, hard-assignment/Sinkhorn protocol, frozen optimizer recipe, global batch4096, seed42, formal epoch budgets or Stage3 recipe. No PPR/teacher/extra baseline/multiseed/pretraining/metric-selected reruns. Candidate remains TIGER_gnn_attention_sectional_rq; old #251 results remain immutable historical evidence, not credentials for changed source.

## Phase1: graph reconstruction and named admission

Inputs: original train.parquet rows `(user:int64, history:list<int64>, seen_history:list<int64>, target:int64)`; global item IDs0..24586. Reconstruction is independent of production graph builders: keep the longest `history+[target]` for each user, enumerate ordered adjacent distinct transitions, and count occurrences as int64 edge weights. Do not count every prefix row as a separate full session. Recommendation valid/test inputs must never contribute edges.

The #246 deterministic sorted-user permutation uses NumPy seed246001 and ratios80/10/10. The training graph contains visible users only; struct_val and struct_test are evaluation only. Expected user counts45951/5744/5744, disjoint complete union57439. Graph comparison must match support AND each integer weight; all-train support323633, visible support261132, canonical-only62501, visible-only0, unexplained residual0. Separately classify val-only/test-only/both missing-edge support and transition multiplicity on already-visible edges; do not confuse support difference with total weight difference67275.

Pure audit APIs to test before implementation:

- `reconstruct_edges(records, allowed_users=None) -> Counter[(source,target), int]`.
- `classify_difference(full, visible, struct_val, struct_test) -> dict` with disjoint val_only/test_only/both counts and unexplained residual, visible_only count and total transition-weight difference.

These APIs run on int64 IDs/weights. Empty users/graphs return empty counters/zero categories; repeated transitions retain multiplicity; self-loops contribute no edge; shorter prefix rows cannot double-count transitions; equal-length conflicting sequences fail rather than silently selecting one. Exact integer equality is required, no floating tolerance.

Case A is provisionally supported by the existing #246/#248 provenance: canonical includes intentionally held-out structural users. Independent raw-data reconstruction must establish it before admission changes. If true, reuse all correct graphs/ego/features/full-target/banks by immutable hash; no graph or target rebuilding. A new declarative named contract under assets/ may bind `infra246-visible-v2` to those existing files. It must bind raw train, split, graph/weights, audit algorithm/version, isolation report and graph-derived training dependencies. Candidate-local adjacency building, unregistered graph paths, mixed graph families, evaluation inputs in training and stale/mutated artifacts remain failures. Existing canonical contracts for unrelated candidates are not silently relaxed. Mutable metadata alone is not authority: the registered contract and its expected digest must be pinned in the gate source.

Graph admission must be executable from the stdlib MCP server; heavy raw-data auditing may use the Stage2 interpreter and then an immutable source/input-bound audit receipt. Cache hits must still verify graph/contract/input identity. A changed legal graph registration/source must invalidate prior entry credentials; a corrupted/mixed/leaking graph must never inherit PASS from cache.

## Phase1: timing and profiling

Historical0.4s claims require recorded scope: actual world size, batch, data, seed, hardware/source/config evidence and timer boundaries. Missing provenance is reported missing, not inferred from current defaults. Audit >=30 consecutive legacy epochs and their timer sum, median/P95 and available timestamp/cumulative window wall.

Current baseline profiling uses actual four ranks/global4096 through launcher.launch, not the old one-card smoke or256-item microstep. Record initialization separately; then distinguish data read, ego assembly/transfer, GNN, Attention, five-geometry assignment costs, Sinkhorn, reference distances, triple statistic, backward and communication. For each report wall/CUDA cost, calls, peak memory and repeated populations. Profiling instrumentation overhead cannot be used as the steady-state timing oracle.

Final speed proof: one continuous >=30-epoch window after initialization, synchronized only at justified window boundaries, actual wall-clock total divided by completed epochs. Include data loading, all forward/backward/optimizer/communication and amortized periodic structure updates/recomputation. Report mean, median, P95, full window wall, initialization/export cost and predicted3000-epoch cost. Hard gate mean<=0.5seconds; no timing exclusions that hide structural work, no smaller data/batch/epoch, no frozen learned outputs.

## Mechanism and prospective estimator changes

Until an explicit phase2 amendment is written BEFORE production edits, mathematical production semantics remain #251's frozen two full induced ego tokens, trainable shared directional GNN, five independent Queries/shared K/V/shared score, two distinct Softmax axes, mixed fixed-curvature hard costs and same-cost soft backward surrogate. Fixed equal-expert symmetric metric supervises actual hard codeword sums; it does not read trainable gate weights. Full objective is centre-normalized Huber of exact full-component A.1 with frozen target scale; lambda uses a real initial encoder-gradient ratio and stays fixed.

Issue252 permits a prospectively specified stochastic/chunked/periodic training estimator. Any phase2 amendment must define item/triple/reference probabilities, complete reference population, coverage/normalization, sampling state/DDP semantics, hard-q and soft-gradient relationships, zero/empty/coincident/boundary cases, tolerances and explicit differences from the full objective. Nonlinear sampling/staleness cannot be called unbiased without proof. Full28391948-row supervision and full-reference final evaluation stay available; no sampling rebuild of existing target assets. Every optimization receives a red regression, independent numerical/gradient/hard-ID comparison and a real-loss update proof. No estimator or approximation is selected from recommendation results.

## Independent references / paper comparison

| Module | Primary definition | Independent reference | Choice |
|---|---|---|---|
| Split graph | #246 CONTRACT and raw Stage0/graph source | Raw per-user transition Counter, not graph_build import | Preserve visible/struct-val/struct-test and recommendation-time isolation |
| Attention | #251 CONTRACT and PAPER_COMPARISON (Vaswani et al., scaled dot-product attention §3.2.1) | Scalar hand Q/K/V expectations in test_attention | Five Queries, shared K/V/score; token and expert axes remain distinct |
| Geometry | #251 common tangent contract and paper comparison | Lorentz/ambient-sphere identities, not production geometry import | Fixed{-3,-1,0,1,3}, scale and zero-curvature limit unchanged |
| Curvature | GraphMoRE arXiv2412.11085v1 AppendixA.1 | Complete-reference direct enumeration vs distance moments; independent NumPy full evaluator | Symmetric distance-induced proxy, not intrinsic/directed curvature |
| Hard quantization | Actual frozen TIGER layer._indices and original Sinkhorn | Hand balanced case and explicit codeword decoding | Preserve all original hard assignment rules/capacity |
| Training estimator | To be prospectively frozen after phase1 profiling | Exact finite-population enumeration/known answers before optimization | Current full objective is the reference, never silently replaced |

## Provenance and final verification

All production/test/config/assets revisions invalidate stale affected evidence. Bind final runs to path:SHA, actual recipe, worker PIDs, run_id and output identities. Full q versus continuous z evaluation, reconstruction and strict banks remain separate; historical TIGER geometry/recall without same-generation provenance is NA. Final archive must contain the exact Stage2 source, SID, checkpoint and full Stage3 recalls/log stream used by that run, never a similarly named substitute. Credentials are read only by the existing uploader and never emitted.

## Phase1: exact archive generation

Retain the existing `results/research/issue139/upload_stage3.py::stage(arm, staging, tree=None)` and R2 ledger/uploader rather than adding a competing archive convention. Before copying/uploading, compare the final Stage3 record's `sid_sha256` and `stage2_checkpoint_sha256` to the actual same-run Stage2 files; mismatches raise ValueError, never select another similarly named generation. Stage3 records must emit those identities without changing any frozen training/evaluation parameter. Source must be the matching immutable Stage2 run tree; a caller-supplied tree is not permission to bypass its identity.

Exactly four first-level entries: mechanism source directory (including a reproducible `stage2_source.tar.gz` with train_rqvae.py, model and all imported source/config), item_sids.json, rqvae_best.pth, recall.json. No fifth Stage3 weights entry. recall.json retains every original final-test key and embeds the complete HG_Rec.log and training_metrics.jsonl plus their hashes without overwriting an existing recall key. Missing logs/final test or crossed SID/checkpoint/source fail closed. Existing unrelated remote archives are not deleted. Correct archive uploads are verified by remote object hashes and the existing ledger, not command success alone. R2 credentials never enter logs, manifests, comments or commits.

## Phase1 lossless runtime layouts and final evaluator, frozen before implementation

New static layouts may change representation only, not graph support, target rows or weights. Reuse the existing frozen one-hop nodes/edges, visible CSR, standardized features and full curvature shards by leaf hash. New `assets/v8_runtime_layout/` may contain: (a) bit-packed visible directed adjacency, preserving every support bit; (b) full two-hop closed-ego node populations and root positions, equal to the existing EgoStore on every item; (c) a memory-mappable per-centre contiguous table of every valid frozen `(b,c,target)` row, with int32 global node IDs, float64 target values, int64 centre offsets and no resampling, averaging or target recomputation. Empty/isolated populations and zero-valid-row centres remain explicit. Duplicate supervision occurrences must remain duplicate rows. Integer/layout comparisons are exact; targets must match the original stored float64 values bit-for-bit. Root IDs, shape/dtype, bytes, hashes, input leaves, generator sources and graph identity belong in the manifest. No trainable GNN output or model parameter may be stored in these caches.

Before using a layout, deterministic tiny tests must cover directed edges, empty/isolated nodes, root placement, repeated triples, invalid-row exclusion and lossless reconstruction. The all-item/full-table audit must compare to existing frozen populations/targets without rebuilding those assets. GPU runtime sampling is a separate prospective phase2 amendment, not authorized implicitly by a lossless layout.

Final evaluation uses only the final same-run checkpoint, exact export manifest and archive binding; independently decodes the first three SID columns into the actual three codebook sums, computes actual encoder/decoder outputs, and reuses `tools/issue251/{evaluation,full_evaluation}.py` rather than creating another metric convention. Every valid frozen triple and every original component reference remains included for both hard-q and continuous-z. Report macro/micro MAE, balanced signs, strict held-out structure/prefix diagnostics, reconstruction, occupancy and collision separately. Same-generation historical TIGER metrics only; unavailable geometric/recall provenance is NA. The evaluator must reject crossed checkpoint/SID, reordered/missing global IDs, malformed code columns and missing final-run identity. A short validation checkpoint cannot be presented as the formal final result.

