# Issue256 R02 frozen contract — native-block module-budget refresh

Frozen before reference/tests or inherited mechanism production patch. Issue body SHA256:32a2764070e8502926965d355a3ef6ff7619d45881abd0ffa575cb7cef73c5a8. Candidate TIGER_periodic_budget_prefix_rq, sole formal run issue256_r02_formal. Parent R01 evidence commit6cd32f054f139dccb8eee03f49afb08e59c50fa9; executed source commit0cdfec9b024e568b5b08d96e777033844d66635e. R01 was fully completed and archived before this design. Fresh source origin is only baseline/TIGER_RQ-VAE; initially all15 source/config files matched its hashes and only ARM_NAME was changed. No checkpoint/cache/result is inherited.

## 1. One principal intervention and evidence

Refresh the identical R01 global module-norm budget at each existing native evaluation-block boundary, rather than freeze its first value for all3000epochs. No new cadence parameter: frozen EVAL_INTERVAL=50. Anchor epochs are1,51,101,...,2951, exactly60 events. Within each50-epoch block the scalar is frozen. The first calibration is identical to R01; no change to original/curvature loss, group partition, scale, sampler, assignment, representation, optimizer, data, LR, clipping, seeds, capacity or evaluation protocol.

Support is training-side, not a choice fitted to test items: final actual four-rank global VJPs exceed the initial-only budget in13/17groups; maximum9.0859271222(Query0); global sharedScore normalized original-dot-combined=-0.6988326877109153. Evidence final_global_gradient/GLOBAL_FINAL_GRADIENTS.json in R01, SHA2562f8ab0d79bb61ea4989fe44c8c2f8ca981f6cdd2d1b00117134663afe94571ee. Counterexample: local Key/Query0 ascent disappears after global averaging. Root cause of low Recall remains unknown. Full external validation, not test error rows, observes41397 of54128 misses with zero matching L1 prefix; this identifies the error stage but does not prove gradient causality. Alternative explanations include Euclidean gate concentration, distance scales, sample weighting and AdamW preconditioning.

The existing50-epoch period is reused without tuning. It limits calibration age to49epochs, not the2999-epoch initial-only horizon. It does not prove an intervening-step bound. If inclusive performance fails, repair execution without changing this mathematical definition or hiding its cost; do not silently change cadence.

## 2. Exact mathematical and data contract

Use R01 CONTRACT.md SHA256aa644b16eb9a378d48384eb72eb618eac8b90efe4790baad2e2fe7b2f2a54156 §§1–5 for all inherited formulas, paper comparisons, tensor shapes, boundary domains and tolerances, except once-only timing in its §1/§2 is explicitly replaced here. The full inherited #255 contract remains SHA256a47efec99b0dff2e03adcfdbb113af925693927736382630e5a1d496de7eb7d5. These are references to immutable definitions, not permission to import a production implementation into the independent oracle.

At anchor epoch e, after DistributedSampler.set_epoch(e-1) and model.train(), and before its first optimizer step, peek exactly the actual first per-rank1024-item batch without advancing global/DataLoader/sampler RNG. Reuse the exact first-step structural sample draw((e-1)*len(loader),rank,world4,track=False). Initial e=1 remains the existing initialization calibration. For e>1 refresh iff (e-1) mod EVAL_INTERVAL=0. Reject invalid nonpositive/noninteger epoch/interval rather than create an ambiguous schedule. Calibration uses training data only.

For O=actual original reconstruction+commitment+codebook and C=(C1+C2+C3)/3, compute per-parameter actual VJPs o_r,p and c_r,p. Average vectors with SUM/world before any norm, then concatenate the same17 disjoint natural coupled groups: encoder; GNN.input; six GNN0/1 self/incoming/outgoing groups; sharedKey/Value/Score; Query0/1/2 rows; codebook0/1/2. Decoder is C-independent and excluded.

At each anchor, a_g=||mean_r o_r,g||2 and b_g=||mean_r c_r,g||2 in float64 reductions. Solve the same one-variable program lambda_e=max{lambda>=0:lambda*b_g<=a_g for all active g}=min_(b_g>0) a_g/b_g. Store positive finite float32 stop-gradient buffer, broadcast/verify all ranks, train L=O+lambda_e*C until the next anchor. No EMA, tuned factor, smoothing, monotonic cap, fallback zero, or alternative loss. All current groups are recomputed; stale optimizer gradients are not used.

Boundary rules: b=0 imposes no bound; a=b=0 inactive; a=0,b>0, absent/nonfinite VJP, no active group, NaN/negative norms or nonpositive/nonfinite float32 coefficient fail closed. No epsilon hiding degeneracy. Norm reference relative2e-6/absolute1e-12; active budget relative2e-6; rank coefficient agreement exact. Calibration changes no parameter/optimizer/scheduler or structural coverage counter. Retained hard codebook allocation buffers may follow their existing deterministic first-forward behavior; actual fresh anchor proof must evaluate the same loss path/sample/state, not a synthetic surrogate.

Cauchy-Schwarz proves o_g dot(o_g+lambda_e*c_g)>=a_g(a_g-lambda_e*b_g)>=0 at that anchor for raw Euclidean gradients. No theorem is claimed for the next49epochs, AdamW directions, noncollapse, semantic quality or Recall. Once-frozen tangent/level/graph target scales stay unchanged across all anchors.

Catalogue[24587,768] float32, global item IDs int64, per-rank batch[1024,768], world4/global4096. Original768->32/three256-word books, three{-1,0,+1}spaces, full directed1/2-hop/in/out+null tokens, shared trainable GNN, per-level Cross-Attention, same mixed cost, same hard selected-code gradient, same cumulative-prefix three Huber terms, all original RQ-VAE losses. L1/L2 floor-constrained nearest and L3 Sinkhorn are the explicitly inherited actual protocol; no claim of all-level baseline Sinkhorn. Hard export must match the checkpoint on all24587 items and exact fourth collision digit.

## 3. Paper / official source cross-check

|Module|Paper position|Official implementation|This round choice|
|---|---|---|---|
|Three experts|GraphMoRE arXiv2412.11085v1 Eqs4–6|RingBDStack/GraphMoRE models.py frozen blob5352bddb1b8a4eb6c32f6585af956b477eb836bd|Unchanged inherited mixed tangent/codeword costs; not three Riemannian encoders|
|Directed GNN/Cross-Attention|GraphMoRE representation module; Vaswani1706.03762 §3.2.1|Official modules do not specify our full per-level directed ego queries|Preserve owner-required four complete ego tokens and shared trainable GNN|
|Curvature proxy|GraphMoRE AppendixA.1 / Eqs9–10|Official alignment differs from frozen graph-triple Huber proxy|Same biased sampled training estimator and independent full-reference common_fixed3 diagnostics|
|RQ-VAE|TIGER residual semantic-ID construction|Read-only repository baseline|Same capacity/original loss and inherited actual floor/Sinkhorn allocation|
|Periodic norm budget|This contract §2, elementary norm inequality|No official GraphMoRE/TIGER equivalent claimed|Only cadence changes from R01, derived from frozen50-epoch evaluation block|

Sources: https://arxiv.org/html/2412.11085v1 ; https://github.com/RingBDStack/GraphMoRE/blob/main/models.py ; https://arxiv.org/abs/1706.03762 . Existing frozen paper audit is reused, not a claim of a new faithful paper reproduction.

## 4. Independent oracle, mutations and pretraining predictions

Independent reference uses SciPy HiGHS LP rather than production closed-form minimum; cadence oracle enumerates arithmetic-progression anchor epochs rather than production modulo. It imports no production code. Hand cases hardcode anchors1/50/51/100/101/2951/3000, coefficient bottlenecks, inactive and degenerate groups. Real trainer path must exercise epoch51: at least60 full diagnostic epochs with actual event epochs[1,51], current-state independent global budget/weight verification, RNG/batch/sample identity, unchanged parameters and all original/structural module routes. One-epoch entry alone cannot establish refresh correctness.

At least five specified-assertion mutations across numeric/structure/routing categories remain required, with new cadence off-by-one, skipped-refresh and stale-weight counterexamples. A crash/import failure is not a catch. First red occurs on current inherited once-only production path, then green after the refresh implementation. Full candidate and inherited relevant tests run on final source. Four-rank real-loss VJPs/updates/hard-prefix/export, numerical boundaries, actual machine entry and behavior_graph must PASS before formal training.

Preregistered predictions (separate, no aggregate success claim):
1. Every observed anchor has all17 active budgets<=1+2e-6, exact rank agreement and a freshly solved maximal positive coefficient. Formal event epochs exactly list(range(1,3001,50)). Failure is implementation/contract failure, not a negative scientific result.
2. Final actual same-definition four-rank gradient max weighted C/O falls below R01's9.0859271222, and global sharedScore has nonnegative original-dot-combined. These are falsifiable mechanism hypotheses, not guaranteed between anchors.
3. Exact final collision<0.4 and catalogue reconstruction MSE no worse than parent255's0.00018233678135234408. Full per-layer macro/micro/sign metrics and prefix/control rates reported even if mixed/worse.
4. Full fixed external recommendation validation R@10 exceeds R01's0.057643761207541916, with a smaller L0-prefix miss fraction among misses than41397/54128. No validation-based checkpoint selection or extra seeds.
5. Sole complete fixed-protocol final test_R@10>0.065. If not, retain the valid negative round, archive it and continue a distinct training/validation-evidence-driven hypothesis; no same-mechanism score rerun.

R5 measures at least30 consecutive inclusive four-card steady epochs31–60, including periodic evaluation50 and refresh51, communication, full-data GNN/2-hop/structural loss and optimizer. Authorized mean<=0.8s, desired0.4–0.5s; report per-card/module and refresh costs. No data pruning/freezing or changed global batch.

## 5. Immutable infrastructure, execution and result protocol

Reuse the109 registered assets and fixed train/valid/test hashes in Issue256 INFRA.json. Manifest SHA256cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b. No static asset is missing or rebuilt. Full graph supervision remains train-visible only. No #254 test error rows, recommendation-test labels/features or structure-test bank participates in this design/calibration.

Source starts from the exact15-file baseline copy and only ARM_NAME, then an inventoried inheritance patch reconstitutes R01's required mechanism; do not copy an old tree as the origin. All changed inherited/new paths and hashes go to LINEAGE.json. Historical source/diagnostic records remain frozen. New round tooling has its own identity; no global monkeypatch or compatibility shim rewires R01.

After all admission: one immutable3000epoch/world4/global4096/seed42 Stage2 via launcher port50202. Finalizer must require60 correctly timed calibration records and bind final coefficient to the checkpoint, not assume one initial event. Exact cold SID, collision<0.4 and persisted layer metrics precede actual Stage3 MCP150epoch/beam20/fulltest. Run the same full57439 validation checkpoint and independent raw-artifact reductions. Complete exact four-piece R2 archive with all logs and source closure. Freeze/report test accesses, repeated-test exploratory bias, lack of untouched independent confirmation. Do not change evaluation/SID vocabulary rules or select another checkpoint. Record actual commands/exit codes/source hashes at every gate.
