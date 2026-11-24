# Issue253 verdict

Per-item status of the issue's acceptance list. `PASS` = checked with recorded
evidence; `NA` = not applicable or not measurable with the available evidence;
`FAIL` = the check was run and did not hold.

| # | Item | Status | Evidence |
|---|---|---|---|
| C1 | Scope frozen from #252's actual candidate, independent identity, unchanged recipe | PASS | `results/research/issue253/CONTRACT.md`, `EXECUTION_CONTRACT.md`, `stage2_RQ-VAE/TIGER_three_expert_attention_rq` (independent copy of `baseline/TIGER_RQ-VAE`) |
| C2 | Full 1-hop/2-hop ego retained, three-expert gate with masked attention | PASS | `model/attention.py`, `model/structure.py`, `tests/`, four-rank profile (`four_rank.json`: 2-hop ≈ 2.0 M nodes / 27 M edges per call) |
| C3 | Three-geometry cost, independent fixed supervision, original training estimator | PASS | `model/geometry.py` (`CURVATURES = (-1, 0, 1)`), `model/estimator.py`, `model/mechanism.py` |
| C4 | Three-expert and five-expert evaluation cells frozen before the run | PASS | `tools/issue253/evaluation_reference.py`, `tools/issue253/final_evaluation.py` (`common_fixed3`, `common_fixed5`, `native_fixed3`) |
| C5 | All legal named graphs and assets reused by hash, nothing rebuilt | PASS | `assets/manifest.json` (88 entries), `tools/research/stage2/graph_contract.py` (`behavior_graph` PASS) |
| C6 | Real four-rank per-rank branch measurement, baseline vs optimized | PASS | `four_rank.json` (baseline vs optimized phase spans), `PERFORMANCE.json`, `OPTIMIZATION_DELTA.json` |
| C7 | Same-step dynamic GNN sharing with dual-branch equivalence | PASS | hard IDs exact on the real four-rank entry, hard `q` exact, loss max abs 0.0, gradient atol 2e-05 / rtol 2e-4; the capacity-constrained assignment **objective** agrees within the accumulated float32 divergence (`tools/issue253/test_shared_forward.py`, `results/research/issue253/validation/`) |
| C8 | Static scheduling and GPU residency, sampling order preserved | PASS | `tools/issue253/static_features.py`, `tools/issue253/test_sample_indices.py` |
| C9 | Rank-local populations preserved, load/wait measured | PASS | `four_rank.json` (`rank_local_populations_preserved=true`, per-rank workloads and communication spans) |
| C10 | Per-ego kernel optimised with exact values and transposed backward | PASS | `tools/issue253/ego_kernel.py`, `tools/issue253/ego_reference.py`, `tools/issue253/test_ego_kernel.py` |
| C11 | No five-expert constant left in any production path | PASS | `CURVATURES = (-1, 0, 1)` is the only expert set; `tools/issue253/test_three_contract.py` |
| C12 | TDD, reference, mutation, graph admission and public entry | PASS | 181 + 11 + 2 tests; `entry` PASS; `behavior_graph` PASS |
| C13 | Performance target or the authorised ceiling | **PARTIAL** | inclusive 0.6396 s/epoch: target 0.4 **not met**, expected ≤ 0.5 **not met**, authorised ceiling 0.8 **met** (`PERFORMANCE.json`); the 3000 formal epochs average 0.6032 s/epoch |
| C14 | Gate weights, attention, gradient norms, clipping, per-card compute/communication | PASS | `four_rank.json`, `PERFORMANCE.json` |
| C15 | One formal Stage2 (3000 epochs) → checkpoint-bound SID → one frozen Stage3 (150 epochs) | PASS | run `stage2_787483d6ef9842d1a7c4593a61ae925c` (3000 epochs, 1843.5 s), SID `24c17aee…` exported from checkpoint `ecc37a67…` with `tokens_match=true`/`differing_items=0`; Stage3 150 epochs, n_eval 57,439, R@10 0.046431866850049616 |
| C16 | Full q/z geometry over all 28,391,948 triples, held-out, per-user recomputation | PASS | 10 cells over the complete triple table with 0 exclusions (`FINAL_EVALUATION.json`, sha256 `7492f837…`); independent per-user recomputation in `RECONCILIATION.json` (sha256 `9d07b3fc…`) |
| C17 | Exact R2 archive and re-verification, conclusion reconciliation, pushed commit | PASS | `r2:genrec/101/`, 99 files, every SHA-256 re-verified by download; ledger entry `101`; `R2_RECEIPT.json` sha256 `4beb844e…` |

## Scientific verdict

| Statement | Status |
|---|---|
| `test_R@10 > 0.065` (adoption threshold) | **FAIL** — 0.046431866850049616 |
| Better than the five-expert #252 arm under the identical protocol | **FAIL** — R@10 0.04643 vs 0.04929, NDCG@10 0.02563 vs 0.02693 |
| Three-expert arm keeps more held-out structure than #252 under the same three-expert metric | **PASS (observational)** — degree-matched Spearman 0.202629 vs 0.169042; two-hop 0.169000 vs 0.126290 |
| Three-expert arm has a lower macro curvature MAE and higher curvature-sign accuracy than #252 under the same metric | **PASS (observational)** — 2.087941 vs 2.117253; 0.362655 vs 0.340347 |
| #252 has a lower micro MAE, lower distance distortion and lower reconstruction MSE | **PASS (observational)** — 2.344406 vs 2.360080; 0.5840 vs 0.6647; 8.840025e-05 vs 9.855251e-05 |
| "Mixed curvature is more suitable" | **NOT ESTABLISHED** — one arm; the learnable gate converges to 88.4 % Euclidean by epoch 3000 |
| "Different graphlets need different geometric biases" | **UNTESTED** — no per-graphlet-type counterfactual arm exists in this issue |
| Original performance target 0.4 s/epoch, expectation ≤ 0.5 s/epoch | **FAIL** — 0.6396 s/epoch (acceptance benchmark), 0.6032 s/epoch (3000 formal epochs) |
| Authorised ceiling 0.8 s/epoch | **PASS** |

