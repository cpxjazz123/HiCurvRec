# Issue252 final verdict

**COMPLETED — negative experimental result, under the recorded owner amendments.** This is not a claim that the original unamended 0.5-second/full-Sinkhorn specification or every operational recording requirement was met without deviation. The qualifications are explicit in `REPORT.md`, not erased by later receipts.

Issue body SHA256 remains `3e9b8ca4018cd9d0a43648efc42cd6a12ef27bc892fdb981590030bea37849eb`; the 14 original criteria remain verbatim in `ACCEPTANCE.json` and `CHECK_3.md`. Performance and assignment amendments were recorded before the completed formal run in `CONTRACT.md` (SHA256 `395ed683321b276701361490036e69228cf5ca41ea360f5a685b55e656a23513`).

## Criterion reconciliation

| Criterion | Actual result and evidence | Judgment |
|---|---|---|
| C1 Raw graph provenance/isolation | Canonical 323633 versus visible 261132 directed edges. Difference 62501 = val-only 31100 + test-only 31241 + both 160; visible-only/unexplained 0. Transition-weight residual 0; strict held-out/train overlap 0. Graph audit E1. | PASS |
| C2 Legal named graph/assets | `infra246-visible-v2`, independently checked against raw train and disjoint user splits. Existing assets reused by hash; new static runtime indices/operators persisted under assets/v8/v9 and registered, not a rebuilt behavioral teacher. Contract E2; asset manifest E3. | PASS |
| C3 Real public admission | Final mounted MCP `verification_check` returned entry PASS and behavior_graph PASS, bound to real four-rank epoch, actual checkpoint/SID agreement and current source inventory. E4/E5. Public invocation is post-run and is not backdated to formal dispatch. | PASS, timing qualification |
| C4 Historical timing and actual profiling | Historical 0.4-second timers distinguished from this mechanism; real four-rank CPU/CUDA phase spans, initialization, repeated work, allocator peaks and timing units recorded. Nested spans are not summed. `PROFILING.md` / `HISTORICAL_TIMING.json` / frozen profile evidence. | PASS |
| C5 Remove redundant computation without freezing learning | Static graph layouts, exact per-ego directional means/adjoints and fused execution replace repeated expansion/CSR assembly. Actual GNN, shared K/V/score, five queries, encoder/decoder and books have finite real-loss gradients and nonzero optimizer updates. E4/E7. | PASS |
| C6 Freeze estimator and preserve full evaluation | Training uses the declared stochastic structural estimator; no claim that it equals the entire full-reference objective. All original 28391948 valid triples remain stored and are fully measured for hard q and continuous z. CONTRACT / full evaluation E8. | PASS |
| C7 Inclusive performance | Original <=0.5 s target NOT MET. Owner-amended <=0.8 s bound met: final-source 30-epoch inclusive window 0.6676147555 s mean, 0.6150472206 median, 0.8196680934 P95. Formal epochs 31–60 mean 0.6721527828; initialization/export and inclusive overhead separately reported. E4/E6 / PROFILING. | PASS under explicit amendment |
| C8 Independent optimization/gradient/SID/DDP checks | Final generation: 193 deterministic + 11 reference tests, 2 mutation suites covering 15 designated numeric/structural/routing mutations; no failures/errors/skips. Actual four-rank validation covers module updates, VJPs, switches, intervention, identical completed rank parameters and exact export. Levels 1/2 no longer claim the original balanced Sinkhorn rule; only level 3 retains it, by the recorded owner change. E4/E7. | PASS under explicit amendment |
| C9 Formal Stage2 | Exactly 3000 consecutive formal epochs, four ranks, global batch4096, seed42, final_epoch checkpoint and complete loss/gradient/timing/gate/memory stream. Only one completed formal generation, `issue252_formal_r2`; failed starts remain preserved. E6/E9. Formal rank PIDs are not present in its JSONL and are not fabricated from later diagnostic PIDs. | PASS for completed run; recording limitation disclosed |
| C10 Exact SID/full diagnostics | All24587 checkpoint re-encodings agree. Exported three-code collision0.0728433725; occupied books256/256/256; complete hard-q/continuous-z curvature, sign, reconstruction and held-out/prefix measurements exist. Legacy G3 remains MISSING, not imputed. E6/E8/E9. Layer recording was after Stage3 launch but before reading its score, as disclosed. | PASS for measured diagnostics; timing limitation disclosed |
| C11 Fault repair/regressions | Deleted-module preflight/analysis references, stale generation bindings, graph lifetime/transfer/collective/runtime issues have concrete regressions. Final repairs additionally cover >64KiB JSONL identity loss, wrong manual-SID generation, intake versus checkpoint manifest, frozen Recall cutoffs and unkeyed historical R2 entries. Red/green logs and original failures preserved; no score-selected recovery. Final complete suites E4. | PASS |
| C12 Stage3/final test/R2 | Exactly150 completed epochs under the frozen four-rank T5 protocol, final test on57439 rows, exact SID/run/checkpoint binding, full logs and checkpoint retained. Four-entry R2 bundle100 uploaded and all100 files hash-verified. E6/E10. Scientific threshold not met; no formal retraining. | PASS, negative scientific result |
| C13 Provenance-safe historical TIGER | Only frozen historical SID SHA025ad8bd… used. Its pure SID collision is recomputed; geometry/reconstruction/Recall lacking matching-generation proof remain NA. No TIGER retraining or crossed checkpoint. E6/E8 / REPORT. | PASS |
| C14 Independent reconciliation/delivery | Per-user Recall and per-centre curvature independently reduced; all completed epoch sequences, exact source/input identities, SID counts and R2 association checked. RECONCILIATION command exited0. Production code and evidence committed/pushed at `25a5164c246fa096e9703f3e3616c9c83a28ad94`, with local/remote main SHA equality observed. This verdict's metadata publication is identified by the final GitHub closing comment. E6. | PASS; closing transaction anchors publication |

## Evidence identities

Paths below are repository-relative unless explicitly part of the immutable run. E4 also contains the complete production/asset inventory, per-test source hashes, executed command/output hashes and four actual-rank report descriptors; E6 contains exact Stage2/Stage3 stream, final-test, shard and evaluator hashes.

| ID | Artifact | SHA256 / association |
|---|---|---|
| E1 | assets/v7_graph_contract/audit.json | `0918afe35cddf209441fff4fe80a558a2d562e14027326d124ec2f51000acb1b` |
| E2 | assets/v7_graph_contract/contract.json | `d61a4ca54f6e7ea29331652aceae1b0d9495e215c44fb8415ae507ed52e38687` |
| E3 | assets/manifest.json | `8c58d34521181925df2e66dbe01f64b7150fb56a1eab028f59e8b4832c381ae6` |
| E4 | results/research/issue252/TRAINING_ADMISSION.json | `f7d24eb30580fd1246b071c1520023ef64d94aa7e7d86092db8e3d2a5976e015` |
| E5 | results/research/issue252/PUBLIC_VERIFICATION.json | `a1c3a4252f407a0af3f913d97afe8038bd482d073da60030bdb188b19b0bf25c` |
| E6 | results/research/issue252/RECONCILIATION.json | `d6cf1cd6520eb06e36bfe8803468c41835bb89de25315a38635e31d3bdf752e8` |
| E7 | results/research/issue252/MODULE_CHECKS.json | `97fbb5e0aff5e2315acb60fb9bbd4dac1627156e6ab8341b4124575454e9605f` |
| E8 | results/research/issue252/FINAL_EVALUATION.json | `e832369d59ea9d8d31a917b274de81afbe38262c1027f274e0b469e8bfbfa3ba`; compact summary committed |
| E9 | immutable run result/rqvae_best.pth; result/item_sids.json | checkpoint `0ca5cd32c307460cad7ecd72734538dff2ca3427f64286063ece8c290f8e28f0`; SID `037766704565c5143ab721fd5ca5f61bbba3e800baa9dd561d3ae5601987cff2` |
| E10 | results/research/issue139/r2_ledger.json, entry100 | `TIGER_gnn_attention_sectional_rq/037766704565`; source tar `253944a0bd36499fbfc3cbf9235d173f1888f41c157061855e2b5f78fe03fe6a`; recall `a91228ea43d7b398bcf227ee2e9e59e30d2a29556b15f4a2bdde92366c1e2ea7` |

Final public identity: candidate digest `4aeae9dec8f3cbbf4ee4b78fc10191e09a5d12cc5bf96180c881a334daad7a85`, checks fingerprint `4fae512fc97c`, graph identity `c6fd4a78ef03e1374dcb9d701163e29e9e25237c617fd987733760aaabe0231b`.

## Mandatory verification mechanisms

- Frozen mathematics/data/edge-domain and paper/official-implementation crosscheck: CONTRACT.md; no post-score retuning of scales or loss weights.
- Independent reference: candidate tests/reference.py and tools/issue251/full_evaluation.py use independent geometry paths; full evaluation crosschecks every distance block against production. The declared stochastic estimator has its own frozen reference/known-answer checks.
- Deterministic answers and normal/boundary/degenerate inputs: candidate tests/test_known_answers.py and final deterministic receipt,193 tests.
- Designated mutation assertions: MUTATIONS.md plus the final mutation receipt's15 caught cases across numeric/structural/routing categories, not crashes or import failures counted as catches.
- Actual module forwards, real training-loss VJPs and parameter updates: E7, derived from the exact current four-rank diagnostic reports; the prior module proof is preserved under its SHA in module_checks_history/.
- Real DDP/entry: E4/E5, including equal four-rank completed parameter hashes and exact diagnostic checkpoint/SID alignment, not toy tensors or an empty suite.
- Independent conclusion recheck: E6 plus the complete full-table measurement E8; the actual CLI smoke succeeded and no scientific threshold was weakened to manufacture adoption.

## Scientific conclusion and scope

The independently recomputed final R@10 is **0.0492870697609638**, below the strict0.065 adoption threshold. The completed experiment is therefore **not adopted**. The single arm does not establish a mixed-curvature advantage or a graphlet-specific geometric necessity. Missing same-generation historical Recall is NA, and observational gate weights are not a causal counterfactual. Full numeric results and operational qualifications appear once in REPORT.md.

## Closing questions

1. Production implementation complete? **Yes** — exact final source/evidence commit above, not a reference-only skeleton.
2. Necessary checks actually executed? **Yes** — E4/E5/E7 and captured red/green/full-suite output; no skipped/zero-case pass.
3. Required formal training completed? **Yes** —3000/150 complete epochs, actual checkpoint/SID/final-test streams and verified archive.
4. Infrastructure reused/persisted correctly? **Yes** — existing registered assets reused by immutable hashes; genuinely new runtime assets are under assets/ and registered. No new behavioral target sampling asset or TIGER retraining.
5. Remaining experimental/implementation work? **No** — only publishing this verdict's metadata and the GitHub closing transaction remain. They are not a reason to rerun training, alter a threshold or reinterpret the negative result.

No unqualified “original protocol fully complied” claim is made: original speed/full-Sinkhorn requirements, public-admission timing, layer-record timing, missing formal PID fields and unavailable legacy G3 are all disclosed, with no fabricated replacement evidence. Final closure is allowed only after the metadata commit is pushed and main SHA equality is checked; its exact SHA is recorded in the closing comment rather than self-referenced inside this file.
