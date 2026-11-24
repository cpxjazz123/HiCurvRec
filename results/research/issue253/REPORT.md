# Issue253: three fixed curvature experts {-1, 0, +1} with the complete 2-hop ego

## Decision

**Negative result for adoption.** The completed `TIGER_three_expert_attention_rq` experiment does not meet the repository's adoption threshold: `test_R@10 = 0.046431866850049616` against the required `> 0.065`. Under the identical frozen Stage3 protocol the same four metrics are slightly below the five-expert #252 arm (`R@10 = 0.0492870697609638`). No score-driven retraining, seed search, extra arm or baseline was run.

| Final next-item metric | #253 (three experts) | #252 (five experts, same protocol) |
|---|---:|---:|
| R@5 | 0.0314246417939031 | 0.03278260415397204 |
| **R@10** | **0.046431866850049616** | 0.0492870697609638 |
| NDCG@5 | 0.02080348571822262 | 0.021616071827379253 |
| NDCG@10 | 0.02563402906712319 | 0.02692583040624563 |

The 57,439-user test population is recomputed independently from the trainer's original per-user target ranks (agreement to 1e-17), not from the trainer's own shard accumulators.

What the single arm does and does not support, stated against the hypothesis that motivated it:

- **Mixed curvature is not shown to be "more suitable" in general.** This arm keeps a mixed-curvature geometry (three experts, a learnable per-item gate over {-1, 0, +1}), but by epoch 3000 the gate has converged to an almost purely Euclidean assignment (mean weights `[0.055, 0.884, 0.060]` for the hyperbolic, Euclidean and spherical experts; epoch 60 was `[0.605, 0.271, 0.125]`). Its Stage3 quality is below the five-expert arm's and far below the adoption threshold. One arm cannot establish the general claim.
- **"Different graphlets need different geometric biases" is neither confirmed nor refuted.** The arm makes the bias learnable per item rather than per subgraph type, and the gate's per-item weights vary; there is no per-graphlet-type counterfactual arm in this issue, so the specific graphlet-typed claim remains untested.
- **What the same-scope geometry measurement does show** (one run per arm, observational): under the identical three-expert fixed metric the three-expert arm has a lower macro curvature MAE (2.0879 vs 2.1173), a higher curvature-sign accuracy (0.3627 vs 0.3403), a higher negative-curvature recall (0.1857 vs 0.1512) and a higher held-out structure correlation (degree-matched Spearman 0.2026 vs 0.1690; two-hop 0.1690 vs 0.1263); the five-expert arm keeps a lower micro MAE (2.3444 vs 2.3601), a lower distance distortion (0.5840 vs 0.6647) and a lower reconstruction MSE (8.840e-05 vs 9.855e-05). Neither curve recovers zero curvature.

The performance target of the issue was not reached either: the inclusive four-rank mean is **0.6396 s/epoch** on the acceptance benchmark and **0.6032 s/epoch** over the 3000 formal epochs, so the 0.4 s/epoch target and the 0.5 s/epoch expectation are **not met**, while the 0.8 s/epoch ceiling that #252 already accepted **is** met.

## 1. Frozen scope (C1–C5)

| Item | Value |
|---|---|
| Candidate | `stage2_RQ-VAE/TIGER_three_expert_attention_rq` — an independent copy of the frozen `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` tree; #252's `TIGER_gnn_attention_sectional_rq` (now in `stage2_RQ-VAE/_archive/`) is the historical reference and was never retrained |
| Input / latent / codebook | 768-d input, 32-d latent, three levels of 256 codewords — unchanged |
| Assignment protocol | #252's actual hard floor (`assignment_floor.json`: `changed=false`, `family=euclid`, `model/layers.py`); only L3 keeps the Sinkhorn assignment, the earlier levels take the floor-constrained nearest codeword with the softmax soft backward surrogate. **This arm therefore makes no three-level balanced-assignment claim** |
| Training recipe | global batch 4096, world size 4, seed 42, 3000 epochs, port 50202 — unchanged |
| Experts | exactly three fixed curvatures `CURVATURES = (-1, 0, 1)` in `model/geometry.py`; every five-expert constant was removed from the production path (C11) |
| Curvature supervision | equal-weight average over the three fixed experts, independent of the trainable gate. This **is** a change of #252's five-expert fixed supervision metric |
| Ego graphs | complete directed 1-hop and 2-hop induced ego per item; no downsampling, no truncation, no frozen GNN, no full-graph GNN |
| Gate | three independent trainable queries (orthogonal init), shared K/V and score MLP, masked attention over the two valid tokens plus a null slot, then a softmax over the three experts. No expert is hard-wired to a subgraph type |
| Forbidden | PPR, four-node full graphlet, teacher pre-training, retraining TIGER, any change of 768/32 or the three-code protocol |

The gate assigns a **learned** weight per expert per item. The fixed supervision metric uses all three experts with equal weight. The trainable gate and the fixed metric are separate by construction.

## 2. Engineering work (C6–C11)

### C6/C13 four-rank measurement and optimisation

Two four-rank profiles with the same measurement scope (CUDA-event span and wall clock, overlapping spans never summed) plus the inclusive 60-epoch benchmark:

| Measurement | Pre-optimization source | Optimized source |
|---|---:|---:|
| Inclusive mean, epochs 31–60 | 0.9119 s/epoch | **0.6240 s/epoch** |
| Inclusive median | 0.5979 s | 0.5794 s |
| Inclusive p95 (nearest-rank) | 2.9670 s | 0.7638 s |
| Periodic full-catalogue epoch 50 | 7.407 s | 1.667 s |

The optimized source differs from the baseline in exactly three files (`model/structure.py`, `model/mechanism.py`, `model/estimator.py`; `train_rqvae.py` and `model/geometry.py` are byte-identical) — see `OPTIMIZATION_DELTA.json` and the archived baseline source tarball.

### C7 exact same-step dynamic sharing

The reconstruction population and the structural auxiliary population are merged into one item-id union; the GNN token/gate forward runs **once** per step on the union and is gathered back into the two branches in their original order. The equivalence against "two independent branch executions of the original GNN" is measured on the real four-rank entry: hard IDs identical (exact), hard `q` identical (exact), loss max abs difference `0.0`, gradients within `atol=2e-05, rtol=2e-4`. Each branch keeps its own quantization population, floor, Sinkhorn, loss normalisation and gradients — no codeword assignment was merged.

### C8 static scheduling and GPU residency

Fixed-seed, global-step and rank-deterministic curvature sampling tables, static indices and the existing six-dimensional neighbour means are prepared ahead of time; the sampling hash, order, coverage and fillers are unchanged. Duplicate `np.unique` calls, repeated index construction and GPU→CPU shape synchronisations were removed; host→device copies use pinned staging buffers.

### C9 four-card responsibilities and balance

The data-parallel split is unchanged: every rank computes its own complete ego, reconstruction and auxiliary work. No single-item GNN was split across cards and no card was dedicated to GNN. Measured per rank: unique items per token call ≈ 1,270 (1,024 reconstruction + 256 auxiliary), 1-hop ≈ 28 k nodes / 100 k edges, 2-hop ≈ 2.0 M nodes / 27 M edges; `rank_local_populations_preserved = true`. The measured per-rank workload differs (the ego degrees differ), which is reported rather than hidden.

### C10 per-ego kernel

The #252 per-ego Triton kernel is reused; the row-wise serial edge traversal is replaced by a legal vectorised/blocked reduction, the direction aggregation and the second-layer affine/readout are fused, and static metadata is precomputed. Values and input/parameter gradients are verified against an independent CSR/dense reference; the neighbourhood extent, the normalisation and the per-step parameter update are unchanged. No low-precision or approximate scheme is used.

### C11 three-expert hygiene

`CURVATURES = (-1, 0, 1)` is the only expert set in the production path; the gate, mixed cost, fixed distance, surrogate, independent evaluator, checkpoint schema and every diagnostic were converted from five to three, with no residual `1/5` weight or five-expert constant.

## 3. Verification and performance (C12–C14)

### C12 TDD, reference, mutation and public entry

| Suite | Tests | Failures | Errors | Skipped |
|---|---:|---:|---:|---:|
| deterministic | 178 | 0 | 0 | 0 |
| reference | 11 | 0 | 0 | 0 |
| mutation | 2 | 0 | 0 | 0 |

The suite covers the independent three-geometry formulas, the zero-curvature limit, the two normalisations, the three-expert attention, the hard `q` as the three codebooks, effective gate intervention changing the code selection, the real sectional loss → selection cost → Q/K/V/score/GNN gradient path, the absence of a bypass around the fixed metric, sampling identity with #252, the same-step shared gradient equivalence, the absence of graph leakage, complete 2-hop coverage and the checkpoint-bound export.

Public machine admission on the final source: `entry` **PASS** (source-bound real four-rank Trainer epoch with an exact checkpoint SID export) and `behavior_graph` **PASS** (raw train split, graph, audit source and training dependencies). Both were re-run against the final source before the formal dispatch.

### C13 performance

| Quantity | Value |
|---|---:|
| Inclusive mean, epochs 31–60 | 0.6396 s/epoch |
| Inclusive median | 0.5901 s |
| Inclusive p95 (nearest-rank `ceil(0.95·N)`) | 0.8210 s |
| Target 0.4 s/epoch | **not met** |
| Expected ≤ 0.5 s/epoch | **not met** |
| Authorised ceiling 0.8 s/epoch | **met** (`passed=true`; the ceiling is measured on the inclusive mean) |
| #252 formal steady state | 0.64058 s/epoch (this arm: 0.999x) |
| Initialisation | 30.8 s |
| Total wall clock of the 60-epoch diagnostic | 90.1 s |
| Peak allocated memory | 8.85 GB |
| Optimiser steps | 6 per epoch per rank; 1,440 in the measured window |
| Periodic full-catalogue epoch 50 | 1.67 s (7.41 s in the pre-optimization source) |

The original performance target is **not** reached and is reported as not met. The arm runs the complete mechanism — full 1-hop and 2-hop ego, all three experts, trainable gate, floor and Sinkhorn at L3 — at 0.6396 s/epoch, inside the 0.8 s/epoch ceiling that #252 already accepted. Nothing was dropped, frozen or downsampled to reach the number, and the periodic work is included in the inclusive window rather than subtracted. The p95 of 0.821 s is one epoch that carries the periodic full-catalogue evaluation; the ceiling is measured on the inclusive mean, as the admission oracle defines it.

### C14 gate, clipping and per-card diagnostics

| Item | Value |
|---|---|
| Gate coverage | population 1,024; per-expert mean 0.3357 / 0.3397 / 0.3246; maximum row-sum error 1.19e-07 |
| Gate at epoch 60 | `[0.5097, 0.3301, 0.1602]` — no collapsed expert |
| Gradient norm before clipping, epoch 60 | 0.002996 |
| Per rank | 18 optimiser steps over 3 profiled epochs; profile window wall 1.60–1.61 s; peak allocated 8.57–8.65 GB; communication CUDA-stream span 0.047–0.054 s |
| Rank-local populations | preserved |

## 4. Formal experiment (C15–C17)

### 4.1 Stage2 — the reported formal run

| Item | Value |
|---|---|
| Immutable run | `results/stage2_RQ-VAE/_runs/stage2_787483d6ef9842d1a7c4593a61ae925c/` |
| Budget | 3000 epochs, 4 ranks, global batch 4096, seed 42, port 50202, `checkpoint_selection=final_epoch` |
| Wall clock | 1843.5 s (30.7 min) |
| Stage2 checkpoint SHA256 | `ecc37a67753bee1bf08c26692f05f87e1601a2970c2adfb707ea4e644e62e72d` |
| Exported SID SHA256 | `24c17aee0818373fbb4c76d175170a6b207d6e09a29faa398b140acd84bae2e3` |
| Export proof | `tokens_match=true`, `same_l3=1.0`, `differing_items=0`, `original_differing_items=274`, `rewritten_from_checkpoint=true`; the trainer's own table is kept as `item_sids.trainer.json` |
| Archive binding SHA256 | `e0aed038ed6e1d6bb11ba4b9b833dc32207896d9d6c2c1dab27098658524e046` |
| Three-code collision | 0.07162321552039697 (22 826 distinct triples / 24 587) — inside the 0.40 Stage3 gate |
| Layered SID record | `results/stage2_RQ-VAE/TIGER_three_expert_attention_rq/stage2_metrics/24c17aee….json`, sha256 `7b243d6e…`, written **before** any Stage3 score was read |

The checkpoint's own export is reproducible in a fresh process (identical `tokens_sha256`), and the framework's export gate rewrote the table to that checkpoint-owned export, as its policy requires.

### 4.2 Full geometry over all 28,391,948 lawful triples

Ten cells, each over the complete frozen triple table with 0 exclusions, the same targets, the same reference set and one common scale (`s_ref = 0.08433422446250916`, taken from the archived #252 checkpoint, never fitted on the evaluation).

| Cell | Macro MAE | Micro MAE | Sign accuracy | Negative / zero / positive recall | Distance distortion (abs / rel) |
|---|---:|---:|---:|---|---|
| **#253 hard q, three experts** | 2.087941 | 2.360080 | 0.362655 | 0.185682 / 0 / 0.881497 | 0.6647 / 0.3486 |
| #253 hard q, five experts | 2.041674 | 2.499896 | 0.339020 | 0.148948 / 0 / 0.896264 | 0.6338 / 0.3356 |
| **#252 hard q, three experts** | 2.117253 | 2.344406 | 0.340347 | 0.151192 / 0 / 0.894906 | 0.5840 / 0.3078 |
| #252 hard q, five experts | 1.944233 | 2.612150 | 0.298239 | 0.074513 / 0 / 0.954150 | 0.6578 / 0.3504 |
| #253 continuous z, three experts | 2.094012 | 2.364204 | 0.362011 | 0.184841 / 0 / 0.880880 | 0.6662 / 0.3493 |
| #252 continuous z, three experts | 2.123783 | 2.350669 | 0.338135 | 0.148260 / 0 / 0.894308 | 0.5842 / 0.3079 |

The #252 rows are a **re-measurement of the archived #252 generation with this arm's evaluator**; the five-expert rows reproduce #252's own published values exactly (macro 1.944233196802475, micro 2.6121500435033793), which is the cross-check that the unified scope is the same measurement.

Under the identical three-expert fixed metric the three-expert arm has a lower macro curvature MAE (2.0879 vs 2.1173), a higher sign accuracy (0.3627 vs 0.3403) and a higher negative-curvature recall (0.1857 vs 0.1512); the five-expert arm has a lower micro MAE (2.3444 vs 2.3601), lower distance distortion (0.5840 vs 0.6647 absolute) and a lower reconstruction MSE (8.840e-05 vs 9.855e-05). Zero curvature is never recovered by either arm. This is an observational comparison of two single runs; it is **not** a causal proof that three experts beat five.

Reconstruction MSE (hard q / continuous z): #253 `9.855251e-05` / `9.803003e-05`; #252 `8.840025e-05` / `8.917174e-05`.

Independent held-out banks (strict test split, no probe trained, no labels passed to the model), Spearman between the geometry's hop distance and the frozen graph's hop distance:

| Cell | degree-matched bank | two-hop bank |
|---|---:|---:|
| **#253 hard q, three experts** | **0.202629** | **0.169000** |
| #253 hard q, five experts | 0.202908 | 0.181563 |
| #252 hard q, three experts | 0.169042 | 0.126290 |
| #252 hard q, five experts | 0.192111 | 0.153039 |
| #253 continuous z, three experts | 0.201429 | 0.165462 |
| #252 continuous z, three experts | 0.170660 | 0.125087 |

Under the identical three-expert metric the three-expert arm keeps more of the held-out structure than the five-expert arm does (0.2026 vs 0.1690 degree-matched, 0.1690 vs 0.1263 two-hop). This is again one run per arm: it is consistent with, and does not prove, a mechanism-level advantage of three experts.

Only the frozen historical TIGER SID is referenced (sha256 `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059`, three-code collision 0.16927644690283483); its geometry and Recall have no matching generation binding and stay **NA**. TIGER was never retrained.

### 4.3 Stage3 — the frozen protocol, one run

| Metric | #253 (three experts) | #252 (five experts, same protocol) | Threshold |
|---|---:|---:|---:|
| test R@5 | 0.0314246417939031 | 0.03278260415397204 | — |
| **test R@10** | **0.046431866850049616** | 0.0492870697609638 | **> 0.065 → not met** |
| test NDCG@5 | 0.02080348571822262 | 0.021616071827379253 | — |
| test NDCG@10 | 0.02563402906712319 | 0.02692583040624563 | — |

The evaluated population is 57,439 test users, 150 completed epochs, four ranks, beam 20, `NO_EVAL=True`, `SKIP_TEST=False`, seed 42, and the frozen trainer hash `54ca6bfbcd55a42e718660dd0004370a41a5e1c2c5038a86bbfc84f32b963690`. The Stage2 checkpoint, SID, export manifest and archive binding are the exact generation named above (sha256 `ecc37a67…`, `24c17aee…`, `ed58ef99…`, `e0aed038…`).

**The adoption threshold `test_R@10 > 0.065` is not met** (0.04643), and every one of the four metrics is slightly below the five-expert #252 arm under the identical protocol. No score-driven retraining, seed search or extra arm was performed.

### 4.4 R2 archive

`r2:genrec/101/`, association key `TIGER_three_expert_attention_rq/24c17aee0818`, 99 files: the bound Stage2 source (`stage2_source.tar.gz`, `archive_binding.json`, `export_manifest.json` and every executed source/config), `item_sids.json`, `rqvae_best.pth` and `recall.json` carrying every key of the final test record plus the complete `HG_Rec.log` and `training_metrics.jsonl`. Every uploaded file's SHA-256 was re-verified by downloading it, and the ledger entry `101` records the number, the key and the digests. Credentials are not part of the record.

### 4.5 Independent recomputation and performance reconciliation

`results/research/issue253/RECONCILIATION.json` (sha256 `9d07b3fc68af901e133ab3b1082f1ff030091bf231dd5ae45edc6f91c0c9e5f7`) re-derives, without importing the candidate model and without reusing the trainer's accumulators:

| Quantity | Value |
|---|---|
| Recall from the original per-user ranks | R@5 0.0314246417939031, R@10 0.046431866850049616, NDCG@5 0.02080348571822261, NDCG@10 0.0256340290671232 (n_eval 57,439) |
| SID catalogue | 24,587 items, 10,386 distinct L2 prefixes, 22,826 distinct L3 triples, collision 0.07162321552039697, L3 refinement gain 1.6997085207343527 bits, all three codebooks 256/256 occupied |
| Reconstruction from the bound arrays | hard q MSE 9.855251e-05 (mean L2 0.238335), continuous z MSE 9.803003e-05 (0.237306) |
| Formal-run timing, all 3000 epochs | mean 0.6032 s/epoch, median 0.5699, p95 0.6271, total window 1809.55 s, inter-epoch work 5.03 s |
| Calibration actually used | reconstruction gradient 1.8867168e-04, curvature gradient 5.7616029, frozen lambda 3.2746386e-05, tangent scale 0.08433422446250916, target scale 2.000375270843506 |
| Scientific threshold | `test_R@10 > 0.065` → **not met** |

The 60-epoch acceptance benchmark (0.6396 s/epoch mean) and the 3000-epoch formal timing (0.6032 s/epoch mean) agree to within 6%; both are inside the 0.8 s/epoch ceiling and both are above the 0.5 s/epoch expectation.

## 5. Protocol qualifications

- The fixed curvature-supervision metric changed from #252's five experts to this arm's three (equal weight over {-1, 0, +1}), independently of the trainable gate. The comparison therefore always states which metric produced the number; MAE values under different metrics are never compared directly.
- Only L3 keeps the Sinkhorn-balanced assignment; the earlier levels take the floor-constrained nearest codeword. This arm makes **no** three-level balanced-assignment claim.
- The three-code SID protocol, the 768/32 dimensions, the three 256-code codebooks, the 3000-epoch Stage2 budget, the 150-epoch Stage3 protocol, the seed and the global batch are unchanged.
- The Stage2 trainer's in-memory export and the checkpoint's own export differ on 274 of 24,587 items; the framework rewrote the exported table to the checkpoint's own table (the table Stage3 read), kept the trainer's table beside it, and the checkpoint's export is reproducible in a fresh process.
- The shared GNN forward is verified by the assignment **objective** rather than by bit-identical hard codes: float32 batch composition changes the last bits of the accumulated costs, and the capacity-constrained assignment is a global matching, so a float-level perturbation can move a few items off their own argmin. The measured objective difference is ~2.5e-6 relative; the reconstruction hard IDs are still required to match exactly.
- The Stage2 run that is reported is `stage2_787483d6ef9842d1a7c4593a61ae925c`. An earlier dispatch (`stage2_6e4324cd326e45eba067d30295f8e4bd`) completed 3000 epochs but was abandoned because a post-dispatch evidence-chain defect (the export-manifest and contract-path defects described in the issue comments) required the training-admission receipt to be regenerated; that dispatch is diagnostic evidence of the launch path only and is not the reported result.
- No PPR, no four-node full graphlet, no teacher pre-training, no retraining of TIGER, and no change of the 768/32 dimensions or the three-code protocol.


