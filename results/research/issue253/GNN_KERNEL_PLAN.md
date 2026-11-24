# C10 — full-ego numerical oracle and prospective kernel design

## Status and ordering

This is test/reference preparation only. No production file, immutable #252 source, existing asset, or training configuration was changed. No tests, build, lint, formatter, GPU diagnostic, or training were run in this phase. A read-only CPU census of existing v9 row pointers supplied the degree counts below; it did not construct or persist any asset.

The legitimate historical performance RED is #252's actual formal inclusive mean **0.640580804297318 s/epoch** for epochs31–3000, versus issue253's expected **≤0.5 s/epoch** (approximately0.4 target). Sources: `results/research/issue252/RECONCILIATION.json`, `PROFILING.md`, `REPORT.md`. This is not a failed numerical test and not a measured three-expert baseline. Do not create a source-text or synthetic timing assertion claiming it demonstrates a new kernel's benefit.

Required next order, controlled by Main:

1. Execute initial frozen contract/RED checks and obtain a **real four-GPU, three-expert, original serial kernel, separate dual-branch profile** before any union/kernel optimization. Keep global4096/world4/six steps and the original stochastic sampler/quantization populations.
2. Authorize production changes explicitly. Introduce only the #253 modules described below, keeping #252 infrastructure byte-identical.
3. Run numerical equivalence on every rank, then isolated kernel timings, then complete same-source four-rank profiling/inclusive windows. The kernel microbenchmark cannot stand in for the complete mechanism's performance acceptance.

## Prepared files and independent oracle

- `tools/issue253/ego_reference.py`: imports only NumPy/PyTorch/standard library, never production modules. Explicitly enumerates directed induced edges into dense adjacency with multiplicity, derives independent incoming/outgoing full-row normalization, and evaluates the uncomposed input projection, both live GNN layers and whole-ego mean. Packed synthetic layout construction is independent of `local_csr`, `batch_operators`, `pack_synthetic`, and the proposed kernel.
- `tools/issue253/test_ego_kernel.py`: eight executable `unittest` cases load the actual candidate `model/structure.py` and follow its selected `ego_kernel`/`directional_means` imports. No test imports a not-yet-existing optimized module or substitutes the reference for a GPU production path. CPU mode exercises the sparse path only; sanctioned diagnostic entry must set `KernelEquivalenceTests.production_device` to its `cuda:LOCAL_RANK` before constructing the suite. CUDA test setup refuses an entry without launcher child-rank context.

Coverage:

1. Hard-coded directed mean/adjoint answer for edges0→1,0→2,2→1, plus isolate; expected input adjoint first coordinate `[6.5,22.5,7,0]`.
2. Independent dense values and exact transpose input gradients in float64/float32 on zero-, short-, long-, repeated-degree rows and repeated ego IDs; supplied output-buffer behavior for CUDA raw row kernels. Degrees straddle8/32/64/128 and include257/513/1049; another incoming star has1049 edges. A row can have1050 after adding the incoming-star edge: this is deliberately not a truncation ceiling.
3. Empty batch, empty ego, root-only isolate and independently unused incoming/outgoing output gradients.
4. Noncontiguous32-wide input value/gradient checks, plus central finite differences on both directions and an isolate through a nonlinear objective.
5. Live first-affine composition, second directional aggregation/affine/tanh and all-node readout: values, feature input gradients, and all ten actual GNN parameter gradients against the uncomposed dense implementation.
6. Complete weak-closed1/2-hop induced egos including non-root internal edges, self-loop, root-only isolate, all masks, repeated/order-restored item IDs and chunk boundaries. Compare every cached node/edge in both directions against explicit induced enumeration. Changing each current parameter tensor must immediately change tokens and agree with a fresh dense evaluation; no learned-output cache across optimizer steps.
7. Central finite differences for every GNN parameter tensor, including second affine/readout and first projection.
8. Dedicated float64 degree3/7 regression in both directed orientations, with hard-coded `10/21` transpose answer; no promotion of float32 reciprocals can satisfy this oracle.

The low-level repeated-edge fixture tests multiplicity arithmetic. It does not relax frozen asset uniqueness: `local_csr` currently rejects duplicate directed support, and production v9 remains unchanged. Repeated items/rows are legal and their gradients accumulate separately. Existing inherited `test_sparse_structure.py`, `test_sparse_adjoint.py`, `test_static_means.py`, and `test_sync_free_batch.py` remain required regressions, not replaced by this suite.

Frozen tolerances: float64 GNN values/input/parameter gradients `atol=1e-9, rtol=1e-8`; float32 `atol=1e-5, rtol=1e-4`; central finite differences `atol=1e-6, rtol=1e-5`, epsilon1e-5. TF32 is explicitly disabled in GPU checks. IDs, edge/node inventories, masks, duplicate restoration and output shapes are exact comparisons. No relaxed shared-branch tolerance is borrowed for the GNN kernel.

Inspection-only risks, not executed failures:

- **[INFERENCE]** #252 `ego_kernel.reciprocal_degrees` converts degrees to float32 unconditionally. Float64 input adjoints involving degree3/7 can therefore exceed the frozen float64 tolerance. Retain the regression; only #253 may fix it by computing reciprocal degrees in the live input dtype, with this precision correction explicitly recorded.
- **[INFERENCE]** #252 raw `row_means` assumes contiguous row-major input; the new noncontiguous test may expose it. New #253 address arithmetic must use host-known input/output strides, not allocate a contiguous copy to hide the problem.
- New numerical tests are not labelled observed RED/PASS until actually executed. The mathematically correct serial reduction is not expected to fail merely because it is serial.

## Existing immutable workload: measured row-pointer census

Read-only inspection used the Stage2 Python interpreter and `np.load(..., mmap_mode='r')`; `np.diff`/`np.bincount` ran in CPU chunks of1,000,000 rows. Quantiles are nearest rank from the exact integer degree histogram. No column arrays, graph builder, asset writer, GPU, or test runner was invoked.

|ego/direction|rows|edges|P50/P90/P95/P99/max degree|zero|1–8|9–32|33–64|65–128|129+|
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
|1/in|535051|1716410|1/7/13/38/982|145275|345573|37116|5358|1479|250|
|1/out|535051|1716410|1/7/13/37/1049|144903|345096|38163|5326|1337|226|
|2/in|37077307|513330917|6/32/51/123/982|1211363|21649350|10581535|2366213|928103|340743|
|2/out|37077307|513330917|6/32/51/118/1049|1446019|21082448|10893662|2438238|907061|309879|

These are full static layouts, not claims about the exact rank-local optimizer-step distribution. Four-rank profiling must also record actual requested node/edge totals per branch/hop/rank and waiting/peaks. Large-ego counts are never an excuse to cap an ego or discard a high-degree row.

## Fixed reduction design, after authorization

New ownership:

- `tools/issue253/ego_kernel.py`: vector/chunked full-row primitive plus existing host-sized geometry interface.
- `tools/issue253/sparse_adjoint.py`: exact paired transpose autograd, importing only the new kernel for dynamic reductions. Preserve the tested CPU CSR path and its higher-order sparse behavior.
- `tools/issue253/static_features.py`: existing immutable six-feature means/cache semantics, importing the unchanged read-only #252 layout consumer. No extra asset or initialization work without later measured net savings.
- Candidate `model/structure.py`: migrate relevant imports to #253 and keep existing runtime methods/state_dict/parameter names, layer count and readout semantics. `_compiled_affine_layer` must import `tools.issue253.cuda_math.compile_dense`, whose same-API implementation/cache namespace belongs to the geometry owner, not the historical #252 cache. Keep `tools/issue252/{ego_kernel,sparse_adjoint,static_features,ego_operators}.py` unchanged; the v9 producer hash and executed #252 source remain valid.

Keep `row_means(row_ptr,col_idx,x,geometry,out=None,scale=None)` and `batch_geometry(layout,items,device)` call contracts. Geometry retains `row_ego`, `ego_base`, `ego_start`, `layout_rows`, `nodes`; shapes are host-known, no GPU→CPU shape reads. `directional_means(x,layout,geometry,operators=None)` and `StaticFeatureMeans(layout,features,chunk=256).gather(layout,ids,plan=None)` retain their external contracts. New internal `reciprocal_degrees` receives an explicit dtype from `x`/gradient. Empty rows/batches must return correctly shaped zero/empty tensors without a zero-grid launch or invalid memory read.

Proposed fixed compile-time tiles for the existing width32:

- One program owns one batch-local row, `BLOCK_FEATURES=32`, `num_warps=4`.
- Zero degree writes exactly zero.
- Degree1–8: masked **8×32** load and reduction; this covers roughly65% of all1-hop rows and57–58% of2-hop rows in addition to zeros.
- Degree9–32: masked **32×32** reduction; P90 for2-hop is32.
- Degree>32: **128×32** masked chunks, summing all chunks before one final full-degree division. Largest observed1049 requires9 chunks, not1049 serial edge iterations. P99 fits one128-edge chunk; the long tail is preserved in full.
- Choose these branches on the row degree already loaded on-device within the same launch, rather than host bucketing or per-step CPU sorting. There is no autotuning of semantics or degree cap. If compiled branch register pressure makes short rows slower, measure and reconsider the fixed dispatch implementation without changing row arithmetic.
- `tl.sum(..., axis=0)` accumulates in float64 for double inputs, float32 for float inputs. Never use fp16/bfloat16/TF32, approximate messages, fast-math reciprocals, or tensor-core precision shortcuts. Use exact dtype division; carry masked edge values as zero. Read live strides from tensor metadata and include them in pointer arithmetic. Returned/out-buffer dtype and shape remain identical to input.

For incoming support `Ain[t,s]=1/din(t)` and outgoing support `Aout[s,t]=1/dout(s)`, the exact reverse is:

`grad_x = Ain^T grad_in + Aout^T grad_out`.

Using opposite supports avoids constructing/sorting a transpose:

- Incoming-gradient contribution at node `s`: gather over **outgoing** neighbors `t`, sum `grad_in[t]/din(t)`.
- Outgoing-gradient contribution at node `t`: gather over **incoming** neighbors `s`, sum `grad_out[s]/dout(s)`.

The optional `scale` mode multiplies each **gathered column's** gradient by its original-direction reciprocal degree and **does not divide by the current reversed row's degree**. Dividing by a reversed row's own degree is a different operator. Normal forward mode divides exactly once by FULL original row degree, never per chunk, nonzero count after masking, tile width or partial degree. Handle either unused output and empty ownership without assuming `grad_out` exists. Save only necessary graph-local live activations/topology; no detached global activation cache or across-step reuse.

## Affine/readout fusion decision

First implement/review the reduction and exact adjoint alone while keeping current PyTorch second affine and `tanh`/readout. This isolates numerical and wall-clock attribution. A second-affine/readout fusion is optional, not a promised speedup:

- A row-local fused second affine may consume current `Wself,Win,Wout,bself`, compute all32 outputs and save needed activations for true gradients. Bias exists only on the self transform. No frozen weights or serialized learned intermediate.
- A fixed-order per-ego segmented mean or partial-sum readout can avoid global atomics, but must include every row, divide by full ego node count and handle empty egos. It cannot add persistent scheduling assets unless real inclusive savings exceed construction/load/transfer costs.
- Do not adopt a hand-written fused backward until all ten parameter gradients and input gradients, finite differences, changed-parameter checks and the unchanged real reconstruction/structural-loss module checks pass. Keep precision-sensitive affine operations outside Triton unless IEEE float32/float64 equivalence is demonstrated; no TF32 `dot` masquerading as exact fusion.
- Only retain a fusion with positive measured end-to-end gain including its saved-state memory and backward. There is no requirement to fuse purely for naming compliance if the full-row reduction is the useful exact optimization.

No new static asset is proposed now. Reuse the existing complete v9 integer layout, original six-feature asset and already-correct six-feature neighbor means. If later evidence justifies persistent bucket metadata or means, create a genuinely new version under `assets/`, register sources/input hashes/schema/coverage/bytes/command/purpose in the manifest, reload-verify it, and retain it only with measured net benefit. Do not rebuild v9 or mutate its producer source.

## Exact later commands and diagnostic API

None of these commands was run during preparation. Run CPU numerical checks now or after authorization as Main directs:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m unittest tools.issue253.test_ego_kernel -v
```

Numerical GREEN repeats that same command after authorized edits, plus the candidate's full existing suites through the main-owned suite runner. A failure must retain its actual traceback; do not relax tolerances to make it green.

The following **prospective entrypoint names are the implementation contract for new main-owned launchers**, not files created or executable in this phase:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue253.launch_steady_profile
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue253.launch_kernel_validation
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue253.launch_kernel_benchmark
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue253.launch_benchmark
```

All launchers must call `tools.research.stage2.launcher.launch`, world4/port50202, not bare CUDA unittest or direct `torchrun`. Paths/modes are source constants, not CLI/env hyperparameter overrides. Main may integrate these exact modes into its existing new launcher design before execution; record any resulting command-name change in the execution evidence.

- `launch_steady_profile`: first capture the three-expert original-kernel/original-dual-branch baseline with source-bound mode constants; later capture optimized identical world4/global4096/six-step spans. Record both directions, both hops, feature gather, first affine, second aggregation/affine, pooling, union if adopted, both quantization branches, attention/geometries, backward, optimizer, NCCL, waits and peaks. Nested spans overlap and are never added.
- `launch_kernel_validation`: candidate entry sets local CUDA device, sets this suite's `production_device`, runs all eight cases on each of four ranks with zero skips, synchronizes before closing its diagnostic receipt and writes rank-specific `kernel_equivalence.json` data. Bind actual candidate/imported kernel/reference/test/asset SHA identities, PIDs/rank/world/run_id/device, per-case status and observed value/gradient error maxima. Exceptions are failures, not skipped GPU checks. It never trains or exports.
- `launch_kernel_benchmark`: benchmark original #252 serial primitive and new #253 primitive in the same diagnostic process on the same immutable layouts and actual recorded rank-local ID populations from the baseline; learned inputs/weights are identical, never use historical weights to initialize a formal candidate. Fixed source constants:20 warmups,100 measured repetitions; CUDA events plus enclosing synchronized wall time, both hops/directions, forward+backward and complete GNN/token path, dtype32 (dtype64 separately as correctness context), per-rank row/edge counts/peaks. Include assembly/transfer costs separately and in enclosing walls; exclude compilation only with separately recorded compile time. Do not include per-iteration test assertions/dense oracle/synchronizations in the timed kernel body. Serial input must be contiguous if its historical stride assumption requires that; separately test the new stride API.
- `launch_benchmark`: complete actual mechanism, original two rank-local quantization populations and full unchanged sampler; consecutive≥30 inclusive epochs, preferably a fixed60-epoch diagnostic as in #252, not a formal arm. Record nearest-rankP95 (`sorted[ceil(.95*n)-1]`), mean/median, all six steps/epoch, waits, periodic work and initial/export costs separately. Expected≤0.5, target~0.4; only complete inclusive≤0.8 permits formal dispatch if the expected target remains unmet. Later reconcile actual formal windows too.

Acceptance requires independent values/input/parameter gradients within frozen tolerances and positive inclusive measured benefit, not a guessed speedup from operation counts. Microbenchmark evidence is attribution only; no single-rank or tiny-fixture timing is four-rank epoch evidence. No cross-rank dynamic GNN reuse, changed quantization population, graph truncation, lowered precision or sampler alteration is part of this design.

## Initial CPU execution diagnosis (parent-run evidence)

Main executed the integrated45-test suite; `INITIAL_NUMERICAL_GREEN.log` records17 assertion failures. Sixteen failure subtests belong to this C10 suite (two dtype adjoints and seven feature/first-layer gradient variables per dtype); the independent geometry intervention failure is outside this ownership. No child rerun was used to confirm the report.

Cause: **oracle-to-production adapter defect, not a change to GNN mathematics**. The low-level fixture intentionally includes edge0→1 twice plus2→1 and1→2. Dense adjacency correctly retains multiplicity, but converting the normalized dense matrix with `.to_sparse_csr()` coalesces the two0→1 occurrences into one weighted support entry. That violates the existing paired-CSR adjoint contract: each stored occurrence must have a uniform reciprocal full-row degree, and reverse support must retain the same occurrences. `_transpose_pair` obtains full degree from pointer differences, not the sum of nonuniform coalesced weights.

Minimal deterministic reproduction, by exact arithmetic: directed occurrences `[(0,1),(0,1),(2,1),(1,2)]`, one feature, upstream incoming gradient at node1 equal1. True `Ain[1]=[2/3,0,1/3]`, hence input gradient `[2/3,0,1/3]`. Dense-to-CSR coalesces that row into two entries with values2/3 and1/3; the production adapter then reconstructs transpose weights from stored degree2 and coalesced opposite support, returning `[1/2,0,1/2]`. The two reported failing input-gradient rows1077/1079 are exactly nodes0/2 of this fixture in the mixed batch. The affine test's failing feature rows1081–1083 are the same repeated-edge ego; only pre-aggregation parameter gradients fail because the second-layer forward values remain correct.

Fix the test adapter, retaining all repeated edges and all original tolerances: independently enumerate a paired occurrence-CSR with repeated columns and per-occurrence reciprocal FULL degree. Keep the independent dense multiplicity oracle unchanged. Frozen production assets reject duplicate support, so no production fix or historical source mutation is required for these CPU failures. Add explicit paired-CSR versus dense-value checks in the adapter. The separate float64 degree3/7 regression remains intact; its CPU PASS does not prove the still-unexecuted CUDA reciprocal implementation correct.

Applied only to `ego_reference.py`/`test_ego_kernel.py`: new independent `occurrence_operators` builds sorted paired CSR directly from explicit edge occurrences; the adapter now verifies exact stored occurrence counts and dense values before invoking production. No production edits, tolerance changes, fixture removals, test runs, or GPU work. Main owns the narrowed rerun of the two previously failing C10 methods.

Parent follow-up: `NUMERICAL_FIX_GREEN.log` records13 narrowed tests passing, including the two previously failing C10 methods. Main extracted arithmetic-identical `model.structure._second_affine` and `_pool_readout` helpers solely for distinct profiler spans, and migrated `_compiled_affine_layer` to the issue253 compiler. Preserve these helper boundaries in later reduction work. Any optional affine/readout fusion must explicitly adapt equivalent profiler boundaries before adoption; do not silently collapse measured spans. The three-expert baseline still uses the original serial kernel and separate dynamic branches; performance implementation remains unauthorized.

## Authorized implementation after actual baseline and CUDA RED

Main supplied completed baseline evidence before authorizing this phase. Three-expert/original-serial-kernel/separate-branch diagnostic `issue253_benchmark_3872457_580922477` measured inclusive epochs31–60 mean0.9118503465s, median0.5979033862s, nearest-rankP95=2.9670059s; actual periodic evaluation at50 took7.402s and remains included. This fails the0.8s continuation ceiling. Instrumented warmed epoch2 and uninstrumented epoch3 are attribution only, not replacement throughput evidence.

Actual four-rank CUDA RED `issue253_kernel_validation_3888447_580934726` ran eight C10 methods on every rank with12 assertion failures/zero errors per rank. `rank_0.log` confirms the previously inferred defects: degree3/7 double adjoint returned0.4761904925107956 versus0.47619047619047616; strided inputs had96/128 wrong values in both dtypes. Float64 feature/first-layer gradients also failed. These observations supersede the earlier inspection-only risk status. No child rerun was used.

Authorized code is now in new `tools/issue253/{ego_kernel,sparse_adjoint,static_features}.py`; candidate `model/structure.py` imports them. Fixed8/32/128 edge tiles retain complete support, normalize once by the full row degree, use explicit strides and float32/float64 accumulators, and perform float32 division with `tl.div_rn`. Kernel floating-point fusion is disabled; there is no reduced-precision matrix multiplication. Reciprocal-degree tensors use the incoming gradient dtype, including float64, and reverse aggregation keeps the original-direction degree. An unused directional output is not materialized and its transpose is not computed. Empty tensors avoid zero-grid launches.

The static cache remains detached six-feature-only with identical construction/gather semantics; initialization slice offsets now come from the existing host offsets, not device scalars. Existing v9 layout and host-transfer readers remain unchanged. No assets were built, rewritten, or added. Candidate `_gather_features`, `_second_affine`, `_pool_readout`, first affine and issue253 compiler boundaries are untouched. No second-affine/readout fusion or new learned cache was introduced.

Associated inherited `tests/test_sparse_adjoint.py` and `tests/test_static_means.py` now exercise issue253 helpers, including their module-level cache-construction interception, instead of accidentally validating historical252 helpers. All original fixtures, assertions and numerical tolerances are preserved. Parent owns CPU/CUDA suites and the common four-rank inclusive profiling/benchmark after integration with its separate same-step union change. This implementation phase ran no tests/builds/formatters/GPU work, and makes no measured speedup or GREEN claim.
