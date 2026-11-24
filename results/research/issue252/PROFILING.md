# Issue252 phase profiling

Historical timing: HISTORICAL_TIMING.json. Original actual four-rank diagnostic: BASELINE_PROFILE.json, with source_snapshot.tar.gz hash and normalized native device-time units. Optimized intermediate actual four-rank diagnostic: OPTIMIZED_PROFILE_INTERMEDIATE.json and raw per-rank reports.

Both diagnostics executed one real full-sized optimizer step, zero complete epochs; neither proves epoch throughput. Nested inclusive spans overlap and must not be summed. Original phase allocator peaks were not measured: only recorded rank-wide upper bounds are available. The optimized run records each inclusive phase allocator peak. Sparse logical matrix numel is not allocated coordinate storage or computational work.

| Scope / phase | Calls/rank | CPU dispatch/wall seconds (min–max) | CUDA stream seconds (min–max) | Peak allocated GB (max) |
|---|---:|---:|---:|---:|
| initialization / Attention | 2 | 0.988053–0.988053 | 0.987253–0.987253 | 5.413 |
| initialization / Decoder | 1 | 0.085463–0.085463 | 0.085516–0.085516 | 5.269 |
| initialization / Encoder | 2,4 | 1.904956–1.929449 | 1.905034–1.929489 | 5.415 |
| initialization / GNN | 4 | 1.551140–1.551140 | 1.560361–1.560361 | 5.610 |
| initialization / assignment_floor | 3,9 | 0.847471–0.998242 | 0.847038–0.998465 | 5.450 |
| initialization / assignment_guards | 24 | 0.852445–0.852445 | 0.852867–0.852867 | 5.450 |
| initialization / codebook_initialization | 1 | 5.472030–5.618485 | 5.472008–5.618476 | 0.219 |
| initialization / ego_static_operator_assembly | 4 | 0.917362–0.917362 | 0.917454–0.917454 | 5.524 |
| initialization / embedding_read | 1 | 0.023320–0.262658 | 0.023384–0.262725 | 0.000 |
| initialization / five_geometry_assignment_cost | 6 | 1.148196–1.148196 | 1.148275–1.148275 | 5.451 |
| initialization / five_geometry_distances | 8 | 1.150644–1.150644 | 1.150806–1.150806 | 5.450 |
| initialization / gradient_calibration | 1 | 10.080423–10.080423 | 10.080385–10.080385 | 5.610 |
| initialization / hard_Sinkhorn | 1 | 0.002011–0.002011 | 0.002029–0.002029 | 5.213 |
| initialization / sampled_structural_forward | 1 | 0.086024–0.086024 | 0.086040–0.086040 | 5.610 |
| initialization / sampled_triple_statistic | 1 | 0.005542–0.005542 | 0.005557–0.005557 | 5.450 |
| initialization / scale_calibration | 1 | 1.110082–1.110706 | 1.110114–1.110740 | 0.219 |
| initialization / soft_allocation | 6 | 0.013034–0.013034 | 0.013142–0.013142 | 5.450 |
| initialization / structural_sample | 1 | 0.001064–0.001064 | 0.001088–0.001088 | 0.119 |
| initialization / train_parquet_read | 1 | 3.955348–4.206193 | 3.955428–4.206207 | 0.000 |
| training_step / Attention | 2 | 0.027748–0.046083 | 0.005135–0.015697 | 6.267 |
| training_step / Decoder | 1 | 0.009183–0.010648 | 0.009207–0.010675 | 6.124 |
| training_step / Encoder | 2 | 0.001379–0.001845 | 0.001420–0.001886 | 6.269 |
| training_step / GNN | 8 | 0.187109–0.261586 | 0.231527–0.309810 | 7.471 |
| training_step / assignment_floor | 6 | 0.081342–0.084552 | 0.081482–0.084691 | 6.303 |
| training_step / assignment_guards | 24 | 0.062498–2.972962 | 0.062956–2.973140 | 6.303 |
| training_step / backward | 1 | 1.190589–1.198806 | 1.196239–1.206837 | 8.494 |
| training_step / data_read | 1 | 0.012163–0.013020 | 0.012196–0.013054 | 5.352 |
| training_step / ego_static_operator_assembly | 4 | 0.037512–0.057057 | 0.038456–0.058088 | 6.413 |
| training_step / five_geometry_assignment_cost | 6 | 0.039307–0.070760 | 0.039898–0.070878 | 6.305 |
| training_step / five_geometry_distances | 8 | 0.041272–0.073193 | 0.042751–0.073685 | 6.304 |
| training_step / hard_Sinkhorn | 1 | 0.003007–0.003180 | 0.003030–0.003203 | 5.907 |
| training_step / optimizer | 1 | 0.333534–0.333675 | 0.333578–0.333721 | 5.059 |
| training_step / sampled_structural_forward | 1 | 0.125675–0.136870 | 0.125693–0.136888 | 6.571 |
| training_step / sampled_triple_statistic | 1 | 0.006046–0.006296 | 0.006066–0.006315 | 6.303 |
| training_step / soft_allocation | 6 | 0.015807–0.016450 | 0.015930–0.016573 | 6.304 |
| training_step / structural_sample | 1 | 0.001136–0.001467 | 0.001158–0.001489 | 5.356 |

Original bottleneck: rank0 rebuilt3,204 egos in one optimizer diagnostic (78.3M expanded nodes,1.072B edges,32.97GB NumPy construction); full-catalogue forward repeated88.95s and backward100.87s are nested costs. The raw full-curvature path processed1,307,338 rows /966 centres, streamed twice, with full-reference distance work. Exact persisted CSR and reference-moment algebra remove this reconstruction/decompression from the hot loop; stochastic structural loss is explicitly a different declared objective.

Optimized cold-step rank0 assignment-guard time contains2.944s NCCL waiting while other ranks first load device-resident static layouts; rank0 has warmed them during calibration. Cold first optimizer allocation and profiler replay likewise cannot be treated as a steady-state lower bound. Every rank retains the original learned full-ego GNN and actual structural gradient computation.

Historical wide_l1 epochs1001–1030: mean0.367733s, median0.347s, nearest-rank P950.479s; role_Q epochs970–999: mean0.260733s, median0.256s, P950.260s, cumulative window7.928s. The old timers exclude post-loop communication/evaluation; GPU/executed-source evidence is missing. They do not certify this mechanism.

Final-source60-epoch benchmark with inclusive epochs31–60, actual epoch50 full evaluation, all-rank structural coverage and a recomputed mean<=0.5s remains required. No formal run has started.

## Current exact execution, before diagonal-mask fusion

Actual inclusive60epoch diagnostic `issue252_benchmark_669470_578127856` FAILED: epochs31–60 window32.3996938094s, mean1.0799897936s, median0.9042086266s, P951.7783592449s; realeval50 included, initialization28.2181958472s separate. Projected3000epoch time3239.9693809450s is an extrapolation, not a completed formal run.

Actual warmed profile `issue252_steady_profile_689572_578144683` completed3epochs/18optimizer steps on all4 ranks, no formal artifacts. Instrumented epoch2=1.7381–1.7464s; uninstrumented epoch3=0.9454–1.1899s, allocated peak9.640–9.897GB.

| Nested phase / epoch | Calls/rank | CPU dispatch/wall seconds | CUDA stream-span seconds |
|---|---:|---:|---:|
| assignment guards | 144 | 0.299418–0.312363 | 0.284823–0.297480 |
| full static operator assembly | 24 | 0.172066–0.174762 | 0.177186–0.179907 |
| soft allocation | 36 | 0.099015–0.103255 | 0.099669–0.103944 |
| GNN | 24 | 0.074387–0.075708 | 0.222414–0.228264 |
| backward | 6 | 0.261937–0.271898 | 0.389811–0.477752 |

These spans overlap and must not be added. Native `aten::fill_` calls numbered11596–13951/rank. Inspection identified36 separation checks ×256 separate self-diagonal writes=9216 avoidable tiny writes per epoch. A strided diagonal view can mask the identical entries with one write per block, preserving every off-diagonal value, cdist result and refusal threshold. CPU production regressions first observed256 writes against a1/3-write bound (RED), then passed the exact known minima, duplicates, partial blocks, empty/singleton and bounded-write assertions after fusion. `SEPARATION_MASK_RED.log` / `SEPARATION_MASK_GREEN.log` are the evidence; actual inclusive remeasurement is still required.

## Final formal run and authorized bound

The earlier sections are historical optimization checkpoints. The original
0.5-second target was **not met**. `CONTRACT.md` records the owner's prospective
0.8-second limit and separate last-level-only Sinkhorn decision; neither is
represented as compliance with the original unamended recipe.

Inclusive wall-clock results, independently recomputed from raw boundaries:

| Window | Epochs | Wall seconds | Mean seconds/epoch | Median | P95 |
|---|---:|---:|---:|---:|---:|
| Admission benchmark 31–60 | 30 | 20.066302 | 0.668877 | 0.617600 | 0.801633 |
| Formal run 31–60 | 30 | 20.164583 | 0.672153 | 0.623867 | 0.776458 |
| Formal run 31–3000 | 2970 | 1902.524989 | 0.640581 | 0.606950 | 0.643008 |
| Formal run 1–3000 | 3000 | 1944.579127 | 0.648193 | 0.607219 | 0.648559 |

The immutable formal generation is `issue252_formal_r2`; all 3000 consecutive
boundaries exist. Windows include gaps and periodic work between boundaries,
not just the sum of the logged optimizer timers. The first formal epoch includes
22.891136 seconds of cold execution/compilation and is retained in the full-run
row. The admission-window prediction was 2006.630166 seconds for 3000 epochs;
the actual trainer's total elapsed record is 1979.702 seconds. This is now an
observed completion, not an extrapolated completion claim.

Initialization has its own recorded 28.908401-second interval. Checkpoint/export
completion occurs 9.398303 wall seconds after the last epoch boundary. These
intervals use their recorded origins and are not added to manufacture an exact
decomposition of the trainer total. Source: the immutable run's
`result/logs/training_metrics.jsonl`, plus the pre-training benchmark referenced
by `TRAINING_ADMISSION.json` (original bytes preserved in `admission_history/`
under SHA256 `e93d46e65ec5ad43d1e216739cc98e12a67de783c44aa12e5767780ef3fdedff`).

