# Issue252: completed single-run experimental result

## Decision

**Negative result.** The completed `TIGER_gnn_attention_sectional_rq` experiment does not meet the repository's adoption threshold. It provides no matched single-geometry or graphlet-specific counterfactual, so it does **not establish** “mixed curvature is more suitable” or “different graphlets require different geometric biases.” This is not a general disproof of mixed curvature either. No score-driven retraining, seed search or extra baseline was performed.

| Final next-item metric | Observed |
|---|---:|
| R@5 | 0.03278260415397204 |
| R@10 | 0.0492870697609638 |
| NDCG@5 | 0.021616071827379253 |
| NDCG@10 | 0.02692583040624563 |

The strict acceptance condition is `test_R@10 > 0.065`; it is not met. That threshold is **not** presented as a provenance-verified historical TIGER score. The evaluated population is 57,439, distributed across four shards as 14,360 / 14,360 / 14,360 / 14,359. Independent reduction of per-user zero-based target ranks reproduces all four reported metrics within 1e-10; it does not reuse the trainer's shard accumulators. Beam width 20 is not an instruction to invent a reported Recall@20 metric: the frozen protocol reports cutoffs 5 and 10.

## Exact experiment and source ownership

- Stage2 immutable generation: `results/stage2_RQ-VAE/_runs/issue252_formal_r2/TIGER_gnn_attention_sectional_rq/`, exactly 3000 epochs, four ranks, global batch 4096, seed 42. The checkpoint selection is `final_epoch`.
- SID SHA256: `037766704565c5143ab721fd5ca5f61bbba3e800baa9dd561d3ae5601987cff2`.
- Stage2 checkpoint SHA256: `0ca5cd32c307460cad7ecd72734538dff2ca3427f64286063ece8c290f8e28f0`.
- Re-encoding from that checkpoint agrees on all 24,587 exported triples, with zero differing items. The fourth export counter does not enter any three-code statistic.
- Stage3: `logs/Amazon_2023_Instruments/Nov-24-2026_15-34-06/` under the matching variant directory; exactly 150 completed epochs, four ranks, global batch 16384, seed 42, beam 20, `NO_EVAL=True`, `SKIP_TEST=False`. Final test and train_end exist.
- Exact archive: `r2:genrec/100/`, association key `TIGER_gnn_attention_sectional_rq/037766704565`. The existing uploader validated source/SID/checkpoint/run identities, uploaded the four top-level entries and verified every uploaded file's SHA256 by downloading it. The ledger contains 100 files; recall.json preserves every final-test key and the complete HG_Rec.log/training_metrics.jsonl contents. Credentials are not part of the record.

The original pre-training admission bytes remain under `admission_history/e93d46e65ec5ad43d1e216739cc98e12a67de783c44aa12e5767780ef3fdedff.json`. Later diagnostics and public verification generations are separate evidence, not backdated training prerequisites. The Stage2 MCP status remains idle for the canonical arm because this formal generation was dispatched through the sanctioned CLI launcher and immutable workspace, not through an MCP launch-state record; its completion is proven by that workspace's actual checkpoint and full metrics stream.

## SID and geometry measurements

The persisted SHA-keyed layer record predates reading the candidate's Stage3 score. Three-code collision is **0.07284337251393014** (`1 - 22796/24587`), L2 distinct prefixes 9130, all three codebooks occupy 256 codes. The 0.0721519502 collision printed inside the final training epoch belongs to the live training diagnostic, not the checkpoint-bound final exported table; the latter is authoritative here.

Recommendation-validation L1 ER is 6.8457096954, from positive hit rate 0.05329131775 and exact legal uniform-negative expectation 0.007784630099. It is an enrichment ratio, not an embedding reconstruction error. L3 refinement is `H(C3|C1,C2)=2.0295648441` bits. The removed legacy Conditional G3 definition is unavailable and remains **MISSING**; the separately named conditional log enrichment is not claimed to be its replacement.

The full CPU geometric remeasurement covers 28,391,948 valid frozen triples and 24,326 centres **for each representation**, using every original component reference, with no sampling or scale fit on evaluation. This is the GraphMoRE A.1 discrete curvature proxy, not an intrinsic sectional curvature measurement.

| Full-table diagnostic | Hard SID decoded q | Continuous encoder z |
|---|---:|---:|
| Curvature macro MAE | 1.9442331968024757 | 1.951155439337128 |
| Curvature micro MAE | 2.612150043503379 | 2.6194174266757386 |
| Sign balanced accuracy | 0.3428875918991509 | 0.34230780265475036 |
| Reconstruction MSE | 0.00008840025082127364 | 0.00008917173965848479 |
| Degree-matched held-out hop-distance Spearman | 0.19211125408305219 | 0.19397677841035002 |

Hard-q negative/zero/positive sign recalls are 0.0745125404 / 0 / 0.9541502353. Low collision therefore does not imply faithful preservation of negative and positive curvature. The final average gate assigns about 70.27% to the Euclidean expert; that observational mean cannot establish the benefit of mixing or a causal graphlet-type effect.

Each representation crosschecks 334,050,193 distance values against the independent ambient implementation. Maximum absolute discrepancies from production are 6.2088e-11 and 2.4835e-10, within the declared 1e-9 tolerance. Full report SHA256 is `e832369d59ea9d8d31a917b274de81afbe38262c1027f274e0b469e8bfbfa3ba` (41,864,252 bytes, kept on shared storage); `FINAL_EVALUATION_SUMMARY.json` is the committed compact record.

Only the frozen historical TIGER SID is compared: SHA256 `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059`, three-code collision 0.16927644690283483. Historical geometry/reconstruction/Recall without a matching generation binding remain NA. No different checkpoint is substituted and TIGER is not retrained.

## Protocol qualifications

1. The original 0.5-second performance requirement was not met. The owner's 0.8-second amendment and last-level-only Sinkhorn change are recorded in the frozen `CONTRACT.md`. Actual formal inclusive epochs 31–60 average 0.6721527828 s; epochs 31–3000 average 0.6405808043 s. See `PROFILING.md` for timing origins, initialization/export and the complete inclusive windows.
2. Levels 1/2 use floor-constrained nearest-code assignment with a softmax surrogate; only level 3 retains Sinkhorn. This arm is not evidence for the original “baseline balanced assignment plus only a changed cost matrix” claim at all three levels (§19h).
3. The manual layer record was written after Stage3 had started, although its definition was frozen earlier and its values were persisted before reading any candidate Stage3 score. It is not falsely described as a pre-launch measurement.
4. The public machine-verification calls were completed during post-run reconciliation. The original real four-rank one-epoch/math/reference/mutation/performance credential existed before formal dispatch; a later public receipt is not substituted for that historical fact.
5. The formal Stage2 JSONL records actual mode, seed, run/source/asset identities and all epoch metrics, but do not contain rank worker PID fields. Later diagnostic worker PIDs must not be reported as the exited formal workers' PIDs.

These qualifications limit compliance/causal claims; none changes the observed held-out score or authorizes pretending the requested positive research conclusion was obtained.

## Reproduction entrypoints

- `python -m tools.issue252.final_evaluation`: explicit immutable final generation, complete geometry/reconstruction/held-out evaluation, no model training or SID rewriting.
- `python -m tools.issue252.reconcile_results`: independent per-user Recall reduction, per-centre curvature aggregation, full SID counts, completed epoch sequences and exact R2/source binding.
- The interpreter is the project Stage2 Python 3.9. Raw data and registered assets remain at their frozen paths/hashes; generated full distributions and repeated diagnostic generations are not Git source files.
